"""Model-independent equations for TAFS-GRPO.

The paper uses reverse velocity ``v=x0-eps`` and therefore ``x0=x+sigma*v``.
Diffusers SD3 predicts the opposite flow direction, so its equivalent is
``x0=x-sigma*model_output``. The convention is explicit to prevent a silent
sign error.
"""

from dataclasses import dataclass
import math

import torch


def _expand(value, like, name):
    value = torch.as_tensor(value, device=like.device, dtype=torch.float32)
    if value.ndim == 0:
        value = value.expand(like.shape[0])
    if value.ndim != 1 or value.shape[0] != like.shape[0]:
        raise ValueError(f"{name} must be scalar or [batch]")
    return value.reshape(value.shape[0], *([1] * (like.ndim - 1)))


def clean_prediction(state, model_output, sigma, convention="diffusers"):
    """Paper Eq. (4)/(13), adapted explicitly for model output convention."""
    sigma = _expand(sigma, state, "sigma")
    if convention == "paper":
        return state.float() + sigma * model_output.float()
    if convention == "diffusers":
        return state.float() - sigma * model_output.float()
    raise ValueError("convention must be 'paper' or 'diffusers'")


def gaussian_log_prob(value, mean, std):
    std = _expand(std, mean, "std")
    if torch.any(std <= 0):
        raise ValueError("annealing sigma must be positive")
    z = (value.float() - mean.float()) / std
    elementwise = -0.5 * (z.square() + 2 * torch.log(std) + math.log(2 * math.pi))
    # Match FlowGRPO's high-dimensional likelihood reduction.
    return elementwise.mean(dim=tuple(range(1, elementwise.ndim)))


@dataclass(frozen=True)
class AnnealedTransition:
    state: torch.Tensor
    mean: torch.Tensor
    std: torch.Tensor
    log_prob: torch.Tensor


def anneal_transition(clean, target_sigma, generator=None, state=None):
    """Full-refresh re-noising in Eqs. (6), (8), and (16)."""
    sigma = _expand(target_sigma, clean, "target_sigma")
    mean = (1.0 - sigma) * clean.float()
    if state is None:
        noise = torch.randn(clean.shape, device=clean.device, dtype=clean.dtype, generator=generator)
        state = mean + sigma * noise
    return AnnealedTransition(state, mean, sigma, gaussian_log_prob(state, mean, sigma))


def stage_group_advantages(rewards, group_ids, eps=1e-4):
    """Eq. (9): normalize each reward stage within each prompt group."""
    rewards = rewards.float()
    if rewards.ndim != 2:
        raise ValueError("rewards must have shape [trajectories, reward_stages]")
    group_ids = group_ids.reshape(-1).to(rewards.device)
    if group_ids.numel() != rewards.shape[0]:
        raise ValueError("one group id is required per trajectory")
    advantages = torch.empty_like(rewards)
    for group_id in torch.unique(group_ids):
        mask = group_ids == group_id
        values = rewards[mask]
        advantages[mask] = (values - values.mean(dim=0)) / (
            values.std(dim=0, unbiased=False) + eps
        )
    return advantages


def stage_grpo_loss(new_log_prob, old_log_prob, advantages, clip_range):
    """Eq. (11): average clipped objectives without collapsing stages."""
    if new_log_prob.shape != old_log_prob.shape or new_log_prob.shape != advantages.shape:
        raise ValueError("log probabilities and advantages must share [batch, stages] shape")
    log_ratio = (new_log_prob.float() - old_log_prob.float()).clamp(-20, 20)
    ratio = log_ratio.exp()
    raw = -ratio * advantages.float()
    clipped = -ratio.clamp(1 - clip_range, 1 + clip_range) * advantages.float()
    loss = torch.maximum(raw, clipped).mean()
    return loss, {
        "ratio_mean": ratio.mean().detach(),
        "approx_kl": (0.5 * log_ratio.square()).mean().detach(),
        "clip_fraction": ((ratio - 1).abs() > clip_range).float().mean().detach(),
    }


def gaussian_kl_same_variance(mean, reference_mean, std):
    std = torch.as_tensor(std, device=mean.device, dtype=torch.float32)
    elementwise = (mean.float() - reference_mean.float()).square() / (2 * std.square())
    return elementwise.mean(dim=tuple(range(1, elementwise.ndim)))
