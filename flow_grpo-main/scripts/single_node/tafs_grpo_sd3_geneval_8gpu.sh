#!/usr/bin/env bash
set -euo pipefail

# Paper setup: 8 H100s, 3 trajectories/GPU/micro-batch, two micro-batches,
# group/effective batch 48, four clean predictions and three reward stages.
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
python scripts/check_geneval_server.py
accelerate launch --config_file scripts/accelerate_configs/multi_gpu.yaml \
  --num_processes=8 --main_process_port="${MAIN_PROCESS_PORT:-29521}" \
  scripts/train_sd3_tafs_grpo.py \
  --config config/tafs_grpo.py:sd35m_geneval_available_rewards
