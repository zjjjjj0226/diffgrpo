import torch

from igrpo.core import (
    collapse_to_x0,
    gaussian_kl_same_variance,
    group_advantages,
    instant_transition,
    policy_mean,
    ppo_grpo_loss,
)


def test_policy_mean_matches_equation_five():
    x = torch.tensor([[[[2.0]]]])
    v = torch.tensor([[[[0.5]]]])
    result = policy_mean(x, v, torch.tensor([0.8]), torch.tensor([0.1]))
    assert torch.allclose(result, torch.tensor([[[[0.48]]]]))


def test_rescoring_collected_action_is_on_policy():
    torch.manual_seed(7)
    x = torch.randn(3, 2, 2)
    v = torch.randn_like(x)
    t = torch.tensor([0.2, 0.5, 0.8])
    dt = torch.tensor([0.05, 0.05, 0.05])
    old = instant_transition(x, v, t, dt, exploration=0.2)
    rescored = instant_transition(x, v, t, dt, exploration=0.2, action=old.action)
    assert torch.allclose(old.log_prob, rescored.log_prob)


def test_collapse_stops_velocity_gradient():
    x = torch.ones(1, 1, requires_grad=True)
    velocity = torch.ones(1, 1, requires_grad=True)
    collapse_to_x0(x, velocity, torch.tensor([0.5])).sum().backward()
    assert x.grad is not None
    assert velocity.grad is None


def test_group_advantages_are_group_normalized():
    rewards = torch.tensor([1.0, 3.0, 10.0, 14.0])
    groups = torch.tensor([0, 0, 1, 1])
    advantage = group_advantages(rewards, groups, eps=0.0)
    assert torch.allclose(advantage, torch.tensor([-1.0, 1.0, -1.0, 1.0]))


def test_ppo_identity_ratio_and_zero_reference_kl():
    log_prob = torch.tensor([-2.0, -1.0])
    advantage = torch.tensor([-1.0, 1.0])
    loss, metrics = ppo_grpo_loss(log_prob, log_prob, advantage, 0.2)
    assert torch.allclose(loss, torch.tensor(0.0))
    assert torch.allclose(metrics["ratio_mean"], torch.tensor(1.0))
    mean = torch.randn(2, 3)
    assert torch.allclose(gaussian_kl_same_variance(mean, mean, torch.ones(2, 1)), torch.zeros(2))
