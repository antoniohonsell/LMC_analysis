#!/bin/bash
set -e

cd "$(dirname "$0")"

source /path/to/your/venv/bin/activate  # adjust to your venv path

python SGDvsAdam/run_sgd_vs_adam.py \
  --which resnet50 \
  --resnet-dataset CIFAR100 \
  --out-dir ./SGDvsAdam_out \
  --data-root ./data \
  --seeds 0,1,2 \
  --tuning-seed 99 \
  --save-every 200 \
  --num-workers 4 \
  --batch-size-resnet50 128 \
  --tune-mode quick \
  --wandb-project "bachelor-thesis"
