#!/bin/bash
#SBATCH --job-name=resnet
#SBATCH --output=/home/3199937/slurm_logs/resnet_%j.out
#SBATCH --error=/home/3199937/slurm_logs/resnet_%j.err
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

export WANDB_API_KEY=your_key_here

python SGDvsAdam/run_sgd_vs_adam.py \
  --which resnet20 \
  --resnet-dataset CIFAR10 \
  --out-dir ./SGDvsAdam_out \
  --data-root ./data \
  --seeds 0,1,2 \
  --tuning-seed 99 \
  --save-every 200 \
  --num-workers 4 \
  --resnet-width 4 \
  --batch-size-resnet 128 \
  --tune-mode quick \
  --wandb-project "bachelor-thesis"
