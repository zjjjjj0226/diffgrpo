#!/usr/bin/env bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
python scripts/check_geneval_server.py
accelerate launch --config_file scripts/accelerate_configs/multi_gpu.yaml \
  --num_processes=2 --main_process_port="${MAIN_PROCESS_PORT:-29514}" \
  scripts/train_sd3_igrpo.py --config config/igrpo.py:geneval_2gpu
