# TAFS-GRPO reproduction

Independent reproduction of **Know Your Step: Faster and Better Alignment for
Flow Matching Models via Step-aware Advantages** (arXiv:2602.01591v2).
Nothing in the existing `flow_grpo/`, `igrpo/`, original configs, or original
launchers is modified by this implementation.

## Paper-to-code map

| Paper component | Code |
|---|---|
| One-step clean prediction, Eq. (4)/(13) | `clean_prediction` |
| Full-refresh re-noising, Eq. (6)/(16) | `anneal_transition` |
| Gaussian transition policy, Eq. (8) | `gaussian_log_prob` |
| Stage-wise group advantage, Eq. (9) | `stage_group_advantages` |
| Stage likelihood ratio and objective, Eq. (10)/(11) | `stage_grpo_loss` |
| Algorithm 1 rollout/replay | `scripts/train_sd3_tafs_grpo.py` |

The sign convention is intentional. The paper defines reverse velocity
`v=x0-epsilon`, giving `x0=x+sigma*v`. Diffusers SD3 predicts the opposite
direction; this repository already uses `x0=x-sigma*model_output`. The config
therefore selects `velocity_convention="diffusers"` explicitly.

## Disclosed SD3.5-M setup

- 512px LoRA training; rank 32, alpha 64.
- AdamW, learning rate `3e-4`, weight decay `1e-4`.
- Four clean predictions at normalized effective noise coordinates
  `1.0, 0.75, 0.5, 0.25`.
- Three reward-bearing annealing transitions.
- Group/effective update batch 48 trajectories = 144 transition losses.
- Distributed micro-batch 24: 3 trajectories/GPU on 8 H100s, collected twice.
- 1500 optimizer iterations.

Run from the repository root:

```bash
bash scripts/single_node/tafs_grpo_sd3_geneval_8gpu.sh
```

Checkpoints are written only under:

```text
logs/tafs_grpo/sd35m_geneval/checkpoints/
```

## Reward reproduction boundary

For the GenEval composition experiment, the paper equally combines GenEval,
PickScore, CLIPScore, and HPS-v2.1 during training. This repository implements
the first three but has no HPS-v2.1 reward dependency or service. The supplied
config therefore runs an explicit equal-weight **three-reward subset**:

```python
{"geneval": 1/3, "pickscore": 1/3, "clipscore": 1/3}
```

It is algorithmically faithful but is not claimed to reproduce the paper's
reported `80.84` (4 NFE) or `82.37` (8 NFE) GenEval values. Exact reproduction
requires adding HPS-v2.1 and using the authors' still-unreleased code/model
details. The paper also evaluates checkpoints at 1024px, while training is
512px.

## Validation

Core equation tests:

```bash
python -m pytest tests/test_tafs_grpo_core.py -q
```

Before a long run, verify that all three reward backends are available. GenEval
uses the repository's external reward server; PickScore and CLIPScore download
their Hugging Face checkpoints. Pre-download them to avoid distributed startup
races.
