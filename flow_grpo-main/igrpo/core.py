"""The model-independent iGRPO equations from Ren et al. (ICML 2026).

Notation follows equations (5)--(12) of the paper.  Time is normalized to
``[0, 1]`` and broadcasts over all non-batch tensor dimensions.
"""

from dataclasses import dataclass
import math

import torch


def _batch_scalar(value: torch.Tensor, like: torch.Tensor, name: str) -> torch.Tensor:
    value = torch.as_tensor(value, device=like.device, dtype=torch.float32)
    if value.ndim == 0:
        value = value.expand(like.shape[0])
    if value.ndim != 1 or value.shape[0] != like.shape[0]:
        raise ValueError(f"{name} must be scalar or shape [batch], got {tuple(value.shape)}")
    return value.reshape(value.shape[0], *([1] * (like.ndim - 1)))


def policy_mean(
    x_t: torch.Tensor,
    velocity: torch.Tensor,
    t: torch.Tensor,
    delta_t: torch.Tensor,
) -> torch.Tensor:
    """Equation (5): mean of the single-step iGRPO policy kernel."""
    t_b = _batch_scalar(t, x_t, "t")
    dt_b = _batch_scalar(delta_t, x_t, "delta_t")
    if torch.any(dt_b <= 0) or torch.any(t_b - dt_b < 0):
        raise ValueError("iGRPO requires 0 < delta_t <= t")
    x0_prediction = x_t.float() - t_b * velocity.float()
    return (1.0 - (t_b - dt_b)) * x0_prediction


def exploration_std(
    t: torch.Tensor,
    delta_t: torch.Tensor,
    like: torch.Tensor,
    exploration: float,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Square root of equations (7)--(8): ``a*sqrt(t/(1-t))*sqrt(dt)``."""
    if not 0.0 < exploration < 1.0:
        raise ValueError("the paper constrains exploration coefficient a to (0, 1)")
    t_b = _batch_scalar(t, like, "t")
    dt_b = _batch_scalar(delta_t, like, "delta_t")
    safe_t = t_b.clamp(min=eps, max=1.0 - eps)
    return exploration * torch.sqrt(safe_t / (1.0 - safe_t)) * torch.sqrt(dt_b)


def policy_log_prob(
    action: torch.Tensor,
    mean: torch.Tensor,
    std: torch.Tensor,
) -> torch.Tensor:
    """Equation (10), averaged over latent dimensions as in FlowGRPO.

    Averaging (rather than summing) keeps PPO ratios numerically usable for
    high-dimensional image latents and matches this repository's convention.
    """
    std = torch.as_tensor(std, device=mean.device, dtype=torch.float32)
    if torch.any(std <= 0):
        raise ValueError("policy standard deviation must be positive")
    z = (action.float() - mean.float()) / std
    elementwise = -0.5 * (z.square() + 2.0 * torch.log(std) + math.log(2.0 * math.pi))
    return elementwise.mean(dim=tuple(range(1, elementwise.ndim)))


@dataclass(frozen=True)
class InstantTransition:
    action: torch.Tensor
    mean: torch.Tensor
    std: torch.Tensor
    log_prob: torch.Tensor


def instant_transition(
    x_t: torch.Tensor,
    velocity: torch.Tensor,
    t: torch.Tensor,
    delta_t: torch.Tensor,
    exploration: float,
    generator: torch.Generator | None = None,
    action: torch.Tensor | None = None,
) -> InstantTransition:
    """Sample (or rescore) the Gaussian kernel in equations (5)--(10)."""
    mean = policy_mean(x_t, velocity, t, delta_t)
    std = exploration_std(t, delta_t, mean, exploration)
    if action is None:
        noise = torch.randn(mean.shape, device=mean.device, dtype=mean.dtype, generator=generator)
        action = mean + std * noise
    log_prob = policy_log_prob(action, mean, std)
    return InstantTransition(action=action, mean=mean, std=std, log_prob=log_prob)


def collapse_to_x0(
    x_after: torch.Tensor,
    collapse_velocity: torch.Tensor,
    t_after: torch.Tensor,
) -> torch.Tensor:
    """Equation (11). Callers must compute ``collapse_velocity`` under no-grad."""
    t_b = _batch_scalar(t_after, x_after, "t_after")
    return x_after.float() - t_b * collapse_velocity.detach().float()


def group_advantages(
    rewards: torch.Tensor,
    group_ids: torch.Tensor,
    eps: float = 1e-4,
) -> torch.Tensor:
    """Equation (2), supporting groups split across arbitrary batch positions."""
    rewards = rewards.float().reshape(-1)
    group_ids = group_ids.reshape(-1).to(rewards.device)
    if rewards.numel() != group_ids.numel():
        raise ValueError("rewards and group_ids must contain the same number of items")
    result = torch.empty_like(rewards)
    for group_id in torch.unique(group_ids):
        mask = group_ids == group_id
        values = rewards[mask]
        result[mask] = (values - values.mean()) / (values.std(unbiased=False) + eps)
    return result


def ppo_grpo_loss(
    new_log_prob: torch.Tensor,
    old_log_prob: torch.Tensor,
    advantages: torch.Tensor,
    clip_range: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Clipped GRPO surrogate in equations (3)--(4)."""
    log_ratio = (new_log_prob.float() - old_log_prob.float()).clamp(-20.0, 20.0)
    ratio = log_ratio.exp()
    advantages = advantages.float()
    loss_unclipped = -advantages * ratio
    loss_clipped = -advantages * ratio.clamp(1.0 - clip_range, 1.0 + clip_range)
    loss = torch.maximum(loss_unclipped, loss_clipped).mean()
    metrics = {
        "ratio_mean": ratio.mean().detach(),
        "approx_kl": (0.5 * log_ratio.square()).mean().detach(),
        "clip_fraction": ((ratio - 1.0).abs() > clip_range).float().mean().detach(),
    }
    return loss, metrics


def gaussian_kl_same_variance(mean: torch.Tensor, reference_mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    """Per-example analytic KL to a reference kernel with identical covariance."""
    elementwise = (mean.float() - reference_mean.float()).square() / (2.0 * std.float().square())
    return elementwise.mean(dim=tuple(range(1, elementwise.ndim)))
