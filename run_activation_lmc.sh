#!/bin/bash
#SBATCH --job-name=act_lmc
#SBATCH --output=/home/3199937/slurm_logs/act_lmc_%j.out
#SBATCH --error=/home/3199937/slurm_logs/act_lmc_%j.err
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --account=3199937
#SBATCH --partition=stud
#SBATCH --qos=stud
#SBATCH --gres=gpu:1

mkdir -p /home/3199937/slurm_logs

cd /home/3199937/LMC_analysis || exit 1

source /home/3199937/envs/lmc_analysis/bin/activate

set -e

python - <<'EOF'
import torch
print("PyTorch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    props = torch.cuda.get_device_properties(0)
    print("GPU name:", props.name)
    print("GPU total memory:", props.total_memory // 1024**2, "MB")
    torch.zeros(1).cuda()
    free, total = torch.cuda.mem_get_info(0)
    print("GPU free memory:", free // 1024**2, "MB")
    print("GPU used memory:", (total - free) // 1024**2, "MB")
EOF

MANIFEST="./SGDvsAdam_out/resnet20_CIFAR10/manifest.json"
OUT_DIR="./SGDvsAdam_out/resnet20_CIFAR10/lmc_activation"

echo "=== Activation-based LMC for all pairs ==="
python SGDvsAdam/run_activation_lmc.py \
    --manifest    "$MANIFEST" \
    --out-dir     "$OUT_DIR" \
    --data-root   ./data \
    --batch-size  256 \
    --num-workers 4 \
    --match-samples 5000 \
    --num-lambdas 25

echo ""
echo "Done. Results in $OUT_DIR"
