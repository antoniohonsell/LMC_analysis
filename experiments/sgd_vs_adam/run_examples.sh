#!/usr/bin/env bash
set -euo pipefail

export PYTHONPATH="$(pwd)"

python SGDvsAdam/run_sgd_vs_adam.py \
  --which both \
  --out-dir ./SGDvsAdam_out

# Faster smoke test:
# python SGDvsAdam/run_sgd_vs_adam.py \
#   --which mlp \
#   --tune-mode off \
#   --epochs-mlp 10 \
#   --out-dir ./SGDvsAdam_smoke
