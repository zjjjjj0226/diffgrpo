# FLUX training timing

## SD3 entry point

To retain all hyperparameters of the original `general_ocr_sd3_1gpu` preset
while launching two GPUs:

```bash
export CUDA_VISIBLE_DEVICES=0,1
export SD3_MODEL_PATH=/your/models/sd35-medium
bash scripts/single_node/grpo_sd3_timed_2gpu_original.sh
```

Upload `config/grpo_sd3_timed_2gpu_original.py` and the matching launcher.
This preset retains optimization/sampling hyperparameters but enables
activation checkpointing in the timed SD3 entry point to reduce activation
memory. Upload the regenerated `train_sd3_timed.py` as well as this preset.
Backward recomputes forward activations, so phase times are not directly
comparable to runs without checkpointing. Memory fit remains unverified.
Per-GPU
sample/train batch stays 8, group size 8, evaluation batch 16, resolution
512, KL beta 0.04, sampling batches 8 and gradient accumulation 4.
With two GPUs this yields 128 images and 16 prompt groups per epoch,
compared with 64 images and 8 groups in the original single-GPU run.
Global update batch also doubles. It is not equivalent in global batch
or epoch sample count. GPU memory fit is unverified; retaining the original
batch sizes can OOM on a 24GB GPU, particularly during evaluation. The
existing small-batch two-GPU preset remains unchanged.

For a single RTX 4090 diagnostic run:

```bash
export CUDA_VISIBLE_DEVICES=0
export SD3_MODEL_PATH=/your/models/sd35-medium
export WANDB_MODE=offline
bash scripts/single_node/grpo_sd3_timed_1gpu.sh
```

Upload `config/grpo_sd3_timed_1gpu.py` as well as the new launcher. The new
configuration imports `grpo_sd3_timed_2gpu.py`, so keep that file too.
Single-GPU sampling uses batch 2 and group size 2 to satisfy the sampler;
training batch is also 2 for consistency. Resolution is reduced to 384.
KL beta is zero because the existing reference-forward path assumes a DDP
`.module` wrapper. This changes the objective and is a diagnostic setting,
not a reproduction of the original KL-regularized experiment. The launcher
uses Python directly rather than the original multi-GPU Accelerate YAML.
Memory fit has not been verified on a GPU.

For two RTX 4090 GPUs, a small-batch diagnostic configuration is available:

```bash
export CUDA_VISIBLE_DEVICES=0,1
# Optional: export SD3_MODEL_PATH=/data/models/stable-diffusion-3.5-medium
bash scripts/single_node/grpo_sd3_timed_2gpu.sh
```

This uses per-rank sample/train batch 1, group size 2, evaluation batch 1,
512px resolution and FP16 LoRA. It retains OCR reward and KL from the original
configuration. Group size is reduced from 8 to 2, so this is a diagnostic
configuration, not equivalent to the original experiment. DDP replicates
models on both GPUs; two 24GB GPUs do not form one 48GB memory pool. This
configuration is not GPU-validated and does not guarantee fitting in 24GB.
If it runs out of memory, try 384px resolution first, then consider explicit
gradient checkpointing or text-embedding caching/offload implementation.
Read timing logs for both ranks to investigate imbalance. Default output
directories include a timestamp; checkpoints use the configured save_dir.

`train_sd3_timed.py` provides the same timing format for `train_sd3.py`.
Neither original training script is modified. Default output is
`logs/sd3_timing`; use a new `FLOW_TIMING_DIR` for every run.

```bash
bash scripts/single_node/grpo_sd3_timed.sh
```

This launcher retains the original SD3 single-GPU OCR configuration and its
OCR dependencies. For another existing SD3 configuration, run directly:

```bash
export FLOW_TIMING_DIR=logs/sd3_timing_run_01
accelerate launch --config_file scripts/accelerate_configs/multi_gpu.yaml --num_processes=1 --main_process_port 29501 scripts/train_sd3_timed.py --config config/grpo.py:general_ocr_sd3_1gpu
```

Regenerate the SD3 copy with `python scripts/build_timed_sd3.py`.
It reuses the AST instrumentation builder alongside `phase_timing.py`.
The timing scope, synchronization overhead, asynchronous reward limitations,
and interruption behavior described below apply to SD3 as well. This entry
point instruments standard SD3 GRPO, not SD3 Fast or GRPO-Guard.

## FLUX entry point and shared output format

`train_flux_timed.py` is a standalone instrumented copy of `train_flux.py`.
The original entry point is unchanged. Run from the repository root in the
same environment as the original training script:

```bash
export FLOW_TIMING_DIR=logs/timing_run_01
export FLOW_TIMING_SYNC=1
accelerate launch --config_file scripts/accelerate_configs/deepspeed_zero2.yaml --num_processes=8 --main_process_port 29501 scripts/train_flux_timed.py --config config/grpo.py:pickscore_flux_8gpu
```

Choose a NEW directory for each run. GPU count and configuration constraints
are the same as the original entry point.

Files are written separately for each distributed rank:

* `phases_rankN.csv`: each phase invocation, epoch, training timestep index,
  duration, elapsed run time, and completion status. Flushed after every row.
* `epochs_rankN.jsonl`: phase totals and percentages of the complete epoch
  wall time, including evaluation/checkpointing when they occur. `other`
  includes loss construction, communication outside measured calls, logging,
  sample processing and instrumentation overhead.
* `rewards_rankN.jsonl`: numeric reward metrics intercepted from training
  W&B calls, with elapsed run time. Usually only rank 0 has these records.
  W&B logging is retained. Evaluation logs remain in the original eval
  function and are not intercepted by this local reward recorder.

Sampling includes VAE decode. `reward_wait_and_transfer` measures only the
remaining blocking wait and result transfer, NOT total reward execution.
Asynchronous rewards may overlap and contend with sampling; their execution
is not counted a second time in percentages. CPU data and text encoding are
separate phases. Forward includes the optional reference-model forward.

Synchronization is on by default for diagnostic wall-clock timing. It adds
overhead and can change CPU/GPU overlap: measure a few representative warm
epochs, not just epoch 0. Set `FLOW_TIMING_SYNC=0` for lower overhead, but
then phase times reflect CPU submission/wait boundaries rather than complete
GPU execution. Use a GPU profiler for kernel-level compute/communication
attribution. DeepSpeed may execute optimizer work within backward, so
`backward_call` and `optimizer_calls` name Python call boundaries rather than
guaranteeing separate mathematical backward and update costs.

Normal Ctrl+C records the partial phase (`completed=False`) and partial epoch
summary. Forced termination/power loss can lose the active phase; already
flushed records remain. Epoch summaries can include interrupted epochs, so
check CSV completion flags before comparing full epochs. Percentages are
per rank; do not add independent rank percentages or phase maxima together.

The helper `build_timed_flux.py` regenerates the new copy from the original
using Python AST (comments are not preserved). Regenerate only when needed:

```bash
python scripts/build_timed_flux.py
```
