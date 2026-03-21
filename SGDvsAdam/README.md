# SGDvsAdam

Drop this folder into the root of `LMC_analysis` and run it from there.

It creates two optimizer-comparison experiments:

- a **3-hidden-layer MLP** using the `mlp` entry in `architectures.py`
- a **ResNet20 with width 16**

For each architecture it:

1. tunes **SGD** and **Adam** on the **same fixed train/val split**
2. selects hyperparameters by **best validation accuracy**, with **validation loss** as tie-break
3. retrains one final model per optimizer on the full train split (`train + val`)
4. computes **weight-matching linear mode connectivity** between the final SGD and Adam checkpoints
5. re-renders the loss/accuracy interpolation plots with your repo plotting style

## Why this tuning rule

For this comparison, the cleanest thing is to keep the data split, model seed, and architecture fixed, and let the optimizers differ only through:

- optimizer family
- learning rate
- weight decay
- schedule (only where it is already standard in your repo)

That makes the comparison much easier to interpret than tuning extra knobs independently.

The defaults are intentionally small:

- **MLP / Adam**: `lr in {3e-4, 1e-3, 3e-3}`, `wd in {0, 1e-4, 1e-3}`
- **MLP / SGD**: `lr in {1e-2, 3e-2, 1e-1}`, `wd in {0, 1e-4, 1e-3}`, momentum `0.9`
- **ResNet20 / Adam**: `lr in {1e-4, 3e-4, 1e-3}`, `wd in {1e-4, 5e-4, 1e-3}`, cosine schedule
- **ResNet20 / SGD**: `lr in {3e-2, 1e-1, 2e-1}`, `wd in {5e-4, 1e-3, 5e-3}`, momentum `0.9`, warmup+cosine

## Recommended defaults

I chose these defaults because they match the current structure of your repo reasonably well:

- **MLP** defaults to `MNIST`
- **ResNet20 width 16** defaults to `CIFAR10`
- **ResNet20 norm** defaults to `flax_ln`

That last choice is deliberate: your current ResNet20 interpolation / weight-matching path is already built around the LayerNorm-style setup, so it is the lowest-friction way to get the SGD-vs-Adam experiment running. If you want classical BN instead, pass `--resnet-norm bn`.

## Main script

Run both experiments end-to-end:

```bash
python SGDvsAdam/run_sgd_vs_adam.py \
  --which both \
  --out-dir ./SGDvsAdam_out
```

Run only the MLP side:

```bash
python SGDvsAdam/run_sgd_vs_adam.py \
  --which mlp \
  --mlp-dataset MNIST \
  --out-dir ./SGDvsAdam_out
```

Run only the ResNet side:

```bash
python SGDvsAdam/run_sgd_vs_adam.py \
  --which resnet20 \
  --resnet-dataset CIFAR10 \
  --resnet-width 16 \
  --resnet-norm flax_ln \
  --out-dir ./SGDvsAdam_out
```

Skip tuning and use the baked-in defaults:

```bash
python SGDvsAdam/run_sgd_vs_adam.py \
  --which both \
  --tune-mode off \
  --out-dir ./SGDvsAdam_out
```

## Output structure

For each experiment you get something like:

```text
SGDvsAdam_out/
  mlp_mnist/
    tuning/
    final/
      sgd/
      adam/
    lmc/
      interp_results.pt
      interp_results.json
      interp_loss.png
      interp_acc.png
    figs/
      mnist_mlp_sgd_vs_adam_loss_interp.png
      mnist_mlp_sgd_vs_adam_acc_interp.png
    manifest.json
```

The same layout is used for `resnet20_cifar10/`.

## Individual scripts

### 1) Train/tune everything

```bash
python SGDvsAdam/run_sgd_vs_adam.py --which both --out-dir ./SGDvsAdam_out
```

### 2) Recompute LMC later from two checkpoints

```bash
python SGDvsAdam/lmc_weight_matching_interp.py \
  --arch resnet20 \
  --dataset CIFAR10 \
  --ckpt-a ./path/to/sgd_final.pth \
  --ckpt-b ./path/to/adam_final.pth \
  --width-multiplier 16 \
  --norm flax_ln \
  --shortcut-option C \
  --out-dir ./tmp_lmc
```

For the MLP:

```bash
python SGDvsAdam/lmc_weight_matching_interp.py \
  --arch mlp \
  --dataset MNIST \
  --ckpt-a ./path/to/sgd_final.pth \
  --ckpt-b ./path/to/adam_final.pth \
  --out-dir ./tmp_lmc_mlp
```

### 3) Re-plot an existing `interp_results.pt`

```bash
python SGDvsAdam/plot_interp_local.py \
  --results ./tmp_lmc/interp_results.pt \
  --dataset CIFAR10 \
  --arch resnet20 \
  --width 16 \
  --tag sgd_vs_adam \
  --out-dir ./tmp_figs
```

## Notes

- The final LMC comparison is done between the **final SGD model** and the **final Adam model** after tuning.
- The tuning trials save their own `history.json`, `summary.json`, and best checkpoints so you can inspect them later.
- The script uses your repo modules directly: `architectures`, `datasets`, `train_loop`, `utils`, and `linear_mode_connectivity.weight_matching_torch`.
- The MLP-side LMC script is generic over `fc1 ... fcN` checkpoints, so it should keep working if you later change hidden width but keep the same naming convention.
