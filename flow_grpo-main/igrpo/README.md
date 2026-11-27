# iGRPO reproduction (isolated from FlowGRPO)

This directory reproduces the algorithm in **iGRPO: Fast Online RL for Flow
Matching Model with Instant Reward** (Ren et al., ICML 2026). It only adds new
files. Existing `flow_grpo/`, `config/grpo.py`, and `scripts/train_sd3.py` are
not imported as mutable code and are not modified.

## Method-to-code map

| Paper | Implementation |
|---|---|
| Eq. (2), group-relative advantage | `group_advantages` |
| Eq. (5), one-step policy mean | `policy_mean` |
| Eq. (7)-(9), exploration kernel | `exploration_std`, `instant_transition` |
| Eq. (10), action likelihood | `policy_log_prob` |
| Eq. (11), stop-gradient collapse | `collapse_to_x0` plus the no-grad second transformer call |
| Eq. (12), instant reward | reward call immediately after VAE decoding |
| Eq. (3)-(4), clipped GRPO | `ppo_grpo_loss` |
| Base-model reference KL | `gaussian_kl_same_variance` and disabled LoRA adapter |

The training entry point is `scripts/train_sd3_igrpo.py`; experiment configs
are in `config/igrpo.py`.

## Instant-reward feasibility probe (run this first)

Before training, validate that the two one-step predictions produce images
that CLIP can score meaningfully. The probe constructs reachable intermediate
states from one deterministic 40-step SD3 ODE trajectory, rather than treating
independent Gaussian noise as an arbitrary intermediate state.

```bash
python scripts/probe_igrpo_instant_reward.py \
  --prompt "three red cubes on a wooden table" \
  --times 0.8 0.6 0.4 0.2 \
  --candidates 4 \
  --output-dir logs/igrpo/instant_reward_probe
```

Outputs are:

- `instant_reward_grid.png`: columns are timesteps; rows are re-noised actions.
- `candidate_*.png`: the 16 individual post-action collapse images.
- `scores.csv`: requested/actual timestep, delta, and CLIPScore for every cell.
- `report.json`: prompt, seed, settings, and all measurements.

Because scheduler shifting means requested and reachable timesteps may differ
slightly, both values are recorded. Use `--exploration` and `--delta-t` for the
paper's undisclosed `a` and step-size sensitivity sweep.

If the Hub connection drops while CLIP is loading, rerun the same command: all
completed model files are cached. You can also point to a fully downloaded
local snapshot and prohibit network access:

```bash
python scripts/probe_igrpo_instant_reward.py \
  --prompt "three red cubes on a wooden table" \
  --clip-model /path/to/openai--clip-vit-large-patch14/snapshot \
  --clip-local-files-only
```

By default, a CLIP loading failure no longer discards the experiment: all
Collapse images and the grid are saved with `CLIPScore unavailable`, and the
full error is recorded in `report.json`. Add `--require-clipscore` when a
missing score should be fatal.

## Low-GPU GenEval comparison

Four isolated launchers compare FlowGRPO and iGRPO with the same base model,
512 resolution, GenEval prompts/reward, global group size 8, and eight collected
images per outer iteration:

```bash
# One GPU
bash scripts/single_node/compare_geneval_flowgrpo_1gpu.sh
bash scripts/single_node/compare_geneval_igrpo_1gpu.sh

# Two GPUs
bash scripts/single_node/compare_geneval_flowgrpo_2gpu.sh
bash scripts/single_node/compare_geneval_igrpo_2gpu.sh
```

| Run | Per-GPU batch | GPUs | Global group |
|---|---:|---:|---:|
| FlowGRPO/iGRPO 1-GPU | 8 | 1 | 8 |
| FlowGRPO/iGRPO 2-GPU | 4 | 2 | 8 |

Logs and checkpoints are separated under
`logs/compare_geneval/{flowgrpo,igrpo}_{1gpu,2gpu}`. These are controlled
low-GPU comparisons, not direct reproductions of the papers' group-24,
32-H100 results. Run the instant-reward feasibility probe before launching
iGRPO training; a meaningful CLIPScore spread and recognizable collapses are
prerequisites for interpreting the comparison.

## Run

Use the existing FlowGRPO environment and reward-server preparation. The paper
used 32 H100 GPUs, per-GPU batch size 3, 24 SD3 images per prompt, AdamW with
learning rate `1e-5`, weight decay `1e-4`, and at most 2500 iterations.

```bash
# GenEval reward (paper configuration 2)
NUM_PROCESSES=32 bash scripts/single_node/igrpo_sd3.sh

# CLIPScore reward on Pick-a-Pic prompts (paper configuration 1)
accelerate launch --config_file scripts/accelerate_configs/multi_gpu.yaml \
  --num_processes=32 scripts/train_sd3_igrpo.py \
  --config config/igrpo.py:clipscore_reward
```

For an 8-GPU functional run, the default `3 * 8 = 24` samples still form one
complete prompt group. This is useful for validation but is not the paper's
32-GPU throughput setup.

Run the CPU unit tests in an environment with PyTorch installed:

```bash
python -m pytest tests/test_igrpo_core.py -q
```

## Reproduction boundaries

The paper does not disclose the exploration coefficient `a`, the distributions
of `t` and `delta_t`, PPO clip range, LoRA/full-parameter choice, or several
systems details. It also describes natural-marginal re-noising in Eq. (6),
while the likelihood-bearing covariance in Eq. (7)-(9) has a different scale.
This implementation follows Eq. (5), (7), and (9) for the policy being
optimized and exposes the undisclosed values under `config.igrpo`. Defaults are
clearly marked as reproduction assumptions; they are not claimed author
hyperparameters.

The released paper reports results but no official source-code link. Exact
table reproduction therefore additionally depends on the authors' undisclosed
settings, access to their 32-H100 setup, reward server versions, and model
checkpoints. The implementation here is intended to make every known equation
and every assumption auditable.

## Expected paper targets

- CLIPScore optimization: GenEval overall `0.82` at 1k iterations.
- GenEval optimization: GenEval overall `0.96` at 1.5k iterations.
- Reported instant-reward training cost: 205 H100 GPU-hours, versus 2.1k for
  FlowGRPO in the paper's comparison.

These are reference targets, not assertions made by the local smoke tests.
