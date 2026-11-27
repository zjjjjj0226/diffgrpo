#!/usr/bin/env bash
set -euo pipefail

# Run from the repository root. Override FLOW_TIMING_DIR for each new run.
export FLOW_TIMING_DIR="${FLOW_TIMING_DIR:-logs/sd3_timing_$(date +%Y%m%d_%H%M%S)}"
export FLOW_TIMING_SYNC="${FLOW_TIMING_SYNC:-1}"
accelerate launch --config_file scripts/accelerate_configs/multi_gpu.yaml \
    --num_processes=1 --main_process_port 29501 \
    scripts/train_sd3_timed.py --config config/grpo.py:general_ocr_sd3_1gpu
