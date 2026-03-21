# SGDvsAdam

Drop this folder into the root of the repo and run it from there.

It creates two optimizer-comparison experiments:

- a **3-hidden-layer MLP** using the `mlp` entry in `architectures.py`
- a **ResNet20 with width 16**

For each architecture it runs three phases:

1. **Tune** `lr` and `weight_decay` for **SGD** and **AdamW** independently, using a fixed tuning seed. Selects the combination with best validation accuracy (tie-break: min validation loss).
2. **Train** 3 seeds per optimizer (6 models total) on the full train split, using the best hparams from phase 1.
3. **Compute LMC** (weight-matching interpolation) for:
   - **Same-optimizer pairs**: C(3,2) = 3 pairs × 2 optimizers = **6 pairs**
   - **Cross-optimizer pairs**: 3 SGD seeds × 3 AdamW seeds = **9 pairs**
   - **15 LMC curves total** per architecture.

**Adam vs AdamW**: Adam applies weight decay as L2 regularization on the adapted gradient (effective decay is scaled by `1/sqrt(v̂)` per parameter). AdamW decouples weight decay from the gradient update — true weight decay, equivalent to what SGD's `weight_decay` does. AdamW is the modern standard and gives a fairer comparison against SGD.

## Tuning grids (quick mode)

- **MLP / AdamW**: `lr in {3e-4, 1e-3, 3e-3}`, `wd in {1e-4, 1e-3, 1e-2}`
- **MLP / SGD**: `lr in {1e-2, 3e-2, 1e-1}`, `wd in {0, 1e-4, 1e-3}`, momentum `0.9`
- **ResNet20 / AdamW**: `lr in {1e-4, 3e-4, 1e-3}`, `wd in {1e-3, 5e-3, 1e-2}`, cosine schedule
- **ResNet20 / SGD**: `lr in {3e-2, 1e-1, 2e-1}`, `wd in {5e-4, 1e-3, 5e-3}`, momentum `0.9`, warmup+cosine

Tuning uses `--tuning-seed` (default: 99), which is intentionally separate from the main seeds so the tuning run is independent of the evaluation runs.

## Recommended defaults

- **MLP** defaults to `MNIST`
- **ResNet20 width 16** defaults to `CIFAR10`
- **ResNet20 norm** defaults to `flax_ln` (matches the repo's existing weight-matching setup)

## Main script

Run both experiments end-to-end:

```bash
python SGDvsAdam/run_sgd_vs_adam.py \
  --which both \
  --out-dir ./SGDvsAdam_out
```

Custom seeds:

```bash
python SGDvsAdam/run_sgd_vs_adam.py \
  --which both \
  --seeds 0,1,2 \
  --tuning-seed 99 \
  --out-dir ./SGDvsAdam_out
```

MLP only:

```bash
python SGDvsAdam/run_sgd_vs_adam.py \
  --which mlp \
  --mlp-dataset MNIST \
  --out-dir ./SGDvsAdam_out
```

ResNet only:

```bash
python SGDvsAdam/run_sgd_vs_adam.py \
  --which resnet20 \
  --resnet-dataset CIFAR10 \
  --resnet-width 16 \
  --resnet-norm flax_ln \
  --out-dir ./SGDvsAdam_out
```

Skip tuning (use baked-in defaults):

```bash
python SGDvsAdam/run_sgd_vs_adam.py \
  --which both \
  --tune-mode off \
  --out-dir ./SGDvsAdam_out
```

## Output structure

```text
SGDvsAdam_out/
  mlp_mnist/
    tuning/
      sgd/
        tuning_summary.json
        ...trial checkpoints...
      adamw/
        tuning_summary.json
        ...
    final/
      sgd/
        seed_0/
        seed_1/
        seed_2/
      adamw/
        seed_0/
        seed_1/
        seed_2/
    lmc/
      same_optimizer/
        sgd/
          seed0_vs_seed1/
          seed0_vs_seed2/
          seed1_vs_seed2/
        adamw/
          seed0_vs_seed1/
          ...
      cross_optimizer/
        sgd_seed0_vs_adamw_seed0/
        sgd_seed0_vs_adamw_seed1/
        ...  (9 pairs total)
    figs/
      ...all interpolation plots...
    manifest.json
  resnet20_cifar10/
    ...same layout...
  manifest_all.json
```

## Cluster resume

Re-running the exact same command resumes automatically:

- Tuning is skipped if `tuning_summary.json` already exists for that optimizer.
- Each training run is skipped if its `_final.pth` already exists (loads saved summary).
- Interrupted training resumes from the latest `_epoch{N}.pth` checkpoint.
- Each LMC pair is skipped if its `interp_results.pt` already exists.

Set `--save-every 5` (or smaller) to limit lost work if the job is killed mid-epoch.

## Individual scripts

### Recompute LMC from two specific checkpoints

```bash
python SGDvsAdam/lmc_weight_matching_interp.py \
  --arch resnet20 \
  --dataset CIFAR10 \
  --ckpt-a ./SGDvsAdam_out/resnet20_cifar10/final/sgd/seed_0/resnet20_CIFAR10_sgd_seed0_final.pth \
  --ckpt-b ./SGDvsAdam_out/resnet20_cifar10/final/adamw/seed_0/resnet20_CIFAR10_adamw_seed0_final.pth \
  --width-multiplier 16 \
  --norm flax_ln \
  --shortcut-option C \
  --out-dir ./tmp_lmc
```

### Re-plot an existing result

```bash
python SGDvsAdam/plot_interp_local.py \
  --results ./tmp_lmc/interp_results.pt \
  --dataset CIFAR10 \
  --arch resnet20 \
  --width 16 \
  --tag sgd_seed0_vs_adamw_seed0 \
  --out-dir ./tmp_figs
```
