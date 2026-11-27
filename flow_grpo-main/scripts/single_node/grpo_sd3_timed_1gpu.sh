#!/usr/bin/env bash
set -euo pipefail

# Direct Python launch avoids the multi-GPU setting in the original YAML.
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export FLOW_TIMING_DIR="${FLOW_TIMING_DIR:-logs/sd3_timing_1gpu_$(date +%Y%m%d_%H%M%S)}"
export FLOW_TIMING_SYNC="${FLOW_TIMING_SYNC:-1}"
python scripts/train_sd3_timed.py --config config/grpo_sd3_timed_1gpu.py
