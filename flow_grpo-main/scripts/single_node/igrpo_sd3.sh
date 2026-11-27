#!/bin/bash
set -euo pipefail

# iGRPO has its own entry point/config and does not alter any FlowGRPO launch.
accelerate launch --config_file scripts/accelerate_configs/multi_gpu.yaml \
  --num_processes="${NUM_PROCESSES:-8}" \
  --main_process_port="${MAIN_PROCESS_PORT:-29503}" \
  scripts/train_sd3_igrpo.py --config config/igrpo.py:geneval_reward
