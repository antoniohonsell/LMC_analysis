#!/bin/bash
#SBATCH --job-name=crh_resnet18
#SBATCH --output=/home/3199937/slurm_logs/crh_resnet18_%j.out
#SBATCH --error=/home/3199937/slurm_logs/crh_resnet18_%j.err
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

set -e

DATASET="CIFAR10"
OUT_DIR="./CRH_resnet18_out"
SEEDS="0,1,2"
EPOCHS=200

echo "=== Phase 1+2: Train ResNet18 on $DATASET ==="
python CRH/train_resnet_crh.py \
    --dataset    "$DATASET" \
    --out-dir    "$OUT_DIR" \
    --epochs     "$EPOCHS" \
    --seeds      "$SEEDS" \
    --batch-size 32 \
    --tune-mode  quick

echo ""
echo "=== Phase 3: Compute CRH metrics ==="
for OPT in sgd; do
    for SEED in 0 1 2; do
        RUN_DIR="$OUT_DIR/$DATASET/final/$OPT/seed_$SEED"
        echo "  -> $OPT / seed $SEED"
        python CRH/compute_crh_resnet.py \
            --run_dir  "$RUN_DIR" \
            --dataset  "$DATASET"
    done
done

echo ""
echo "Done. Results in $OUT_DIR/$DATASET/final/<opt>/seed_*/crh_metrics_resnet.csv"
