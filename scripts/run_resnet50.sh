#!/bin/bash
#SBATCH --job-name=resnet50
#SBATCH --output=/home/3199937/slurm_logs/resnet50_%j.out
#SBATCH --error=/home/3199937/slurm_logs/resnet50_%j.err
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --account=3199937
#SBATCH --partition=stud
#SBATCH --qos=stud
#SBATCH --gres=gpu:1

mkdir -p /home/3199937/slurm_logs

cd /home/3199937/LMC_analysis || exit 1

source /home/3199937/envs/lmc_analysis/bin/activate

export WANDB_API_KEY=your_key_here  # set via: wandb login (run once on the cluster)

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
