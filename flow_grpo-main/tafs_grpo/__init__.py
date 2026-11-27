"""Isolated TAFS-GRPO reproduction (arXiv:2602.01591v2)."""

from .core import (
    AnnealedTransition,
    anneal_transition,
    clean_prediction,
    gaussian_kl_same_variance,
    stage_group_advantages,
    stage_grpo_loss,
)

__all__ = [
    "AnnealedTransition",
    "anneal_transition",
    "clean_prediction",
    "gaussian_kl_same_variance",
    "stage_group_advantages",
    "stage_grpo_loss",
]
