#!/usr/bin/env bash
set -euo pipefail

export FLOW_TIMING_DIR="${FLOW_TIMING_DIR:-logs/sd3_timing_2gpu_original_$(date +%Y%m%d_%H%M%S)}"
export FLOW_TIMING_SYNC="${FLOW_TIMING_SYNC:-1}"
# Populate the shared PaddleOCR cache before distributed workers start.
# Concurrent first-time downloads use the same .tar.tmp filename.
python -c "from paddleocr import PaddleOCR; PaddleOCR(use_angle_cls=False, lang='en', use_gpu=False, show_log=False); print('OCR cache ready')"
accelerate launch --config_file scripts/accelerate_configs/multi_gpu.yaml \
    --num_processes=2 --main_process_port 29501 \
    scripts/train_sd3_timed.py --config config/grpo_sd3_timed_2gpu_original.py
