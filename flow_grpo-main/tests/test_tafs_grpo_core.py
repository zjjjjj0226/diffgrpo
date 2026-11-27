import torch

from tafs_grpo.core import (
    anneal_transition,
    clean_prediction,
    stage_group_advantages,
    stage_grpo_loss,
)


def test_clean_prediction_conventions_are_opposites():
    state = torch.tensor([[2.0]])
    output = torch.tensor([[0.5]])
    sigma = torch.tensor([0.8])
    assert torch.allclose(clean_prediction(state, output, sigma, "paper"), torch.tensor([[2.4]]))
    assert torch.allclose(clean_prediction(state, output, sigma, "diffusers"), torch.tensor([[1.6]]))


def test_full_refresh_mean_and_rescore():
    clean = torch.tensor([[1.0], [2.0]])
    sigma = torch.tensor([0.75, 0.25])
    sampled = anneal_transition(clean, sigma)
    rescored = anneal_transition(clean, sigma, state=sampled.state)
    assert torch.allclose(sampled.mean[:, 0], torch.tensor([0.25, 1.5]))
    assert torch.allclose(sampled.log_prob, rescored.log_prob)


def test_advantages_are_normalized_per_stage():
    rewards = torch.tensor([[1.0, 10.0], [3.0, 14.0], [5.0, 20.0], [9.0, 22.0]])
    groups = torch.tensor([0, 0, 1, 1])
    result = stage_group_advantages(rewards, groups, eps=0)
    assert torch.allclose(result, torch.tensor([[-1., -1.], [1., 1.], [-1., -1.], [1., 1.]]))


def test_identity_policy_ratio():
    logp = torch.tensor([[-2.0, -1.0], [-3.0, -2.0]])
    advantage = torch.tensor([[-1.0, 1.0], [1.0, -1.0]])
    loss, metrics = stage_grpo_loss(logp, logp, advantage, 0.2)
    assert torch.allclose(loss, torch.tensor(0.0))
    assert torch.allclose(metrics["ratio_mean"], torch.tensor(1.0))
