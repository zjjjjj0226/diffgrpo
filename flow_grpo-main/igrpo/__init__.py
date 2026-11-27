"""Isolated iGRPO reproduction components.

This package intentionally does not monkey-patch or modify FlowGRPO.
"""

from .core import (
    InstantTransition,
    collapse_to_x0,
    gaussian_kl_same_variance,
    group_advantages,
    instant_transition,
    policy_log_prob,
    policy_mean,
    ppo_grpo_loss,
)

__all__ = [
    "InstantTransition",
    "collapse_to_x0",
    "gaussian_kl_same_variance",
    "group_advantages",
    "instant_transition",
    "policy_log_prob",
    "policy_mean",
    "ppo_grpo_loss",
]
