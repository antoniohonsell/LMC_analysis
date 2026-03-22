#!/usr/bin/env bash
set -e

DATASET=${1:-CIFAR10}
OUT_DIR="./CRH_resnet18_out"
SEEDS="0,1,2"
EPOCHS=200

echo "=== Phase 1+2: Train ResNet18 on $DATASET ==="
python CRH/train_resnet_crh.py \
    --dataset "$DATASET" \
    --out-dir  "$OUT_DIR" \
    --epochs   "$EPOCHS" \
    --seeds    "$SEEDS" \
    --tune-mode quick

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
