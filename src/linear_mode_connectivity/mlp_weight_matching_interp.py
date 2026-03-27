# mnist_mlp_weight_matching_interp.py
from __future__ import annotations

import argparse
import os
import pickle
import re
import sys
from pathlib import Path
from typing import Dict, Optional, Tuple, List

_HERE = Path(__file__).resolve().parent
_SRC  = _HERE.parent
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms

import utils
import architectures
import datasets as ds_utils  # repo's datasets.py (for MNIST stats + optional splitting)

"""
HOW TO RUN:

MNIST :

python -m linear_mode_connectivity.mnist_mlp_weight_matching_interp \
  --ckpt-a /Users/antonio2/Bachelor_Thesis/runs_mlp/MNIST/disjoint/seed_0/subset_A/MLP_MNIST_subsetA_seed0_best.pth \
  --ckpt-b /Users/antonio2/Bachelor_Thesis/runs_mlp/MNIST/disjoint/seed_0/subset_B/MLP_MNIST_subsetB_seed0_best.pth \
  --seed 0 \
  --out-dir ./wm_mnist_mlp_disjoint_seed0

FASHION MNIST:
  
python -m linear_mode_connectivity.mnist_mlp_weight_matching_interp \
  --dataset FASHIONMNIST \
  --ckpt-a /Users/antonio2/Bachelor_Thesis/runs_mlp/FASHIONMNIST/disjoint/seed_0/subset_A/mlp_FASHIONMNIST_subsetA_seed0_best.pth \
  --ckpt-b /Users/antonio2/Bachelor_Thesis/runs_mlp/FASHIONMNIST/disjoint/seed_0/subset_B/mlp_FASHIONMNIST_subsetB_seed0_best.pth

"""

from linear_mode_connectivity.weight_matching_torch import (
    PermutationSpec,
    apply_permutation,
    permutation_spec_from_axes_to_perm,
    weight_matching,
)


def load_ckpt_state_dict(path: str) -> Dict[str, torch.Tensor]:
    obj = torch.load(path, map_location="cpu")
    if isinstance(obj, dict) and "state_dict" in obj:
        return obj["state_dict"]
    if isinstance(obj, dict):
        # allow raw state_dict saved directly
        return obj
    raise ValueError(f"Unrecognized checkpoint format at: {path}")


def normalize_state_dict_keys(state: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """
    Makes checkpoints robust to common wrappers (DataParallel/Lightning/etc.)
    by stripping leading prefixes like 'module.' or 'model.' if present.
    """
    prefixes = ("module.", "model.", "net.")
    out = dict(state)
    changed = True
    while changed:
        changed = False
        keys = list(out.keys())
        for p in prefixes:
            if all(k.startswith(p) for k in keys):
                out = {k[len(p):]: v for k, v in out.items()}
                changed = True
                break
    return out


def filter_to_spec_keys(state: Dict[str, torch.Tensor], spec_keys: set[str]) -> Dict[str, torch.Tensor]:
    return {k: v for k, v in state.items() if k in spec_keys}


def interpolate_state_dict(
    a: Dict[str, torch.Tensor],
    b: Dict[str, torch.Tensor],
    lam: float,
) -> Dict[str, torch.Tensor]:
    out: Dict[str, torch.Tensor] = {}
    for k in a.keys():
        va, vb = a[k], b[k]
        if va.dtype.is_floating_point:
            out[k] = (1.0 - lam) * va + lam * vb
        else:
            out[k] = va
    return out


@torch.no_grad()
def eval_loss_acc(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    max_batches: Optional[int] = None,
) -> Tuple[float, float]:
    model.eval()
    crit = nn.CrossEntropyLoss(reduction="sum")

    total_loss = 0.0
    total_correct = 0
    total_seen = 0

    for b_ix, (x, y) in enumerate(loader):
        if max_batches is not None and b_ix >= max_batches:
            break
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss = crit(logits, y).item()
        total_loss += loss
        preds = logits.argmax(dim=1)
        total_correct += (preds == y).sum().item()
        total_seen += y.numel()

    denom = max(1, total_seen)
    return total_loss / denom, total_correct / denom


def plot_interp_loss(title: str, lambdas: List[float],
                     train_loss_naive, test_loss_naive, train_loss_perm, test_loss_perm):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(lambdas, train_loss_naive, linestyle="dashed", alpha=0.5, linewidth=2, label="Train, naïve interp.")
    ax.plot(lambdas, test_loss_naive,  linestyle="dashed", alpha=0.5, linewidth=2, label="Test, naïve interp.")
    ax.plot(lambdas, train_loss_perm,  linestyle="solid", linewidth=2, label="Train, permuted interp.")
    ax.plot(lambdas, test_loss_perm,   linestyle="solid", linewidth=2, label="Test, permuted interp.")
    ax.set_xlabel(r"$\lambda$")
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Model A", "Model B"])
    ax.set_ylabel("Loss")
    ax.set_title(title)
    ax.legend(loc="upper right", framealpha=0.5)
    fig.tight_layout()
    return fig


def plot_interp_acc(title: str, lambdas: List[float],
                    train_acc_naive, test_acc_naive, train_acc_perm, test_acc_perm):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(lambdas, train_acc_naive, linestyle="dashed", alpha=0.5, linewidth=2, label="Train, naïve interp.")
    ax.plot(lambdas, test_acc_naive,  linestyle="dashed", alpha=0.5, linewidth=2, label="Test, naïve interp.")
    ax.plot(lambdas, train_acc_perm,  linestyle="solid", linewidth=2, label="Train, permuted interp.")
    ax.plot(lambdas, test_acc_perm,   linestyle="solid", linewidth=2, label="Test, permuted interp.")
    ax.set_xlabel(r"$\lambda$")
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Model A", "Model B"])
    ax.set_ylabel("Accuracy")
    ax.set_title(title)
    ax.legend(loc="lower right", framealpha=0.5)
    fig.tight_layout()
    return fig


def infer_mlp_fc_layer_numbers(state: Dict[str, torch.Tensor]) -> List[int]:
    """
    Looks for keys like fc1.weight, fc2.weight, ... and returns sorted layer indices.
    """
    pat = re.compile(r"^fc(\d+)\.weight$")
    layers = []
    for k in state.keys():
        m = pat.match(k)
        if m:
            layers.append(int(m.group(1)))
    layers = sorted(set(layers))
    if not layers:
        raise KeyError("No fc{n}.weight keys found in checkpoint state_dict.")
    return layers


def mlp_permutation_spec_from_state(state: Dict[str, torch.Tensor]) -> PermutationSpec:
    """
    Builds a permutation spec for an MLP with layers fc1..fcN:
      - permute hidden units (output dimension) of every layer except the last,
      - and match corresponding input dimensions of subsequent layers.

    For your repo MLP (fc1..fc4) this aligns the 3 hidden layers of size 512 and
    permutes the input axis of fc2/fc3/fc4 accordingly.
    """
    layer_ids = infer_mlp_fc_layer_numbers(state)
    n_layers = max(layer_ids)

    # Require contiguous fc1..fcN
    expected = list(range(1, n_layers + 1))
    if layer_ids != expected:
        raise ValueError(f"Expected contiguous fc layers {expected}, found {layer_ids}")

    axes: Dict[str, Tuple[Optional[str], ...]] = {}

    # Hidden perms P1..P{n_layers-1}
    prev_p: Optional[str] = None
    for i in range(1, n_layers):
        p_out = f"P{i}"
        w_key = f"fc{i}.weight"
        b_key = f"fc{i}.bias"
        axes[w_key] = (p_out, prev_p)  # (out, in)
        axes[b_key] = (p_out,)         # (out,)
        prev_p = p_out

    # Last (classifier) layer: output is class dimension => not permuted
    last_w = f"fc{n_layers}.weight"
    last_b = f"fc{n_layers}.bias"
    axes[last_w] = (None, prev_p)  # (out_classes, in_hidden)
    axes[last_b] = (None,)

    return permutation_spec_from_axes_to_perm(axes)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt-a", type=str, required=True, help="Path to checkpoint for model A (.pth)")
    parser.add_argument("--ckpt-b", type=str, required=True, help="Path to checkpoint for model B (.pth)")

    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-iter", type=int, default=100)
    parser.add_argument("--num-lambdas", type=int, default=25)

    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--eval-samples", type=int, default=1000, help="<=0 means full dataset")

    parser.add_argument("--dataset",
                        type=lambda s: s.strip().upper(),
                        default="MNIST",
                        choices=["MNIST", "FASHIONMNIST"],
                        help="Must match the dataset used to train ckpt-a/ckpt-b.")
    parser.add_argument("--data-root", type=str, default="./data")
    parser.add_argument("--no-normalize", action="store_true")
    parser.add_argument("--train-eval-split", type=str, default="full", choices=["full", "part0", "part1"],
                        help="If you trained on subset A/B via a 2-way stratified split, use part0/part1 to evaluate "
                             "on that train subset (split uses --seed).")

    parser.add_argument("--out-dir", type=str, default=None,
                        help="If unset, defaults to ./weight_matching_out_{dataset}_mlp")
    parser.add_argument("--silent", action="store_true")
    args = parser.parse_args()

    dataset = args.dataset  # already upper-cased by arg type
    if args.out_dir is None:
        args.out_dir = f"./weight_matching_out_{dataset.lower()}_mlp"
    os.makedirs(args.out_dir, exist_ok=True)
    device = utils.get_device()

    # -------------------
    # Load checkpoints
    # -------------------
    state_a_full = normalize_state_dict_keys(load_ckpt_state_dict(args.ckpt_a))
    state_b_full = normalize_state_dict_keys(load_ckpt_state_dict(args.ckpt_b))

    # Sanity: ensure MLP-like keys exist
    for k in ("fc1.weight", "fc1.bias"):
        if k not in state_a_full or k not in state_b_full:
            raise KeyError(
                f"Expected '{k}' in both checkpoints. "
                f"Keys present (sample): {list(state_a_full.keys())[:10]}"
            )

    # Infer model shapes from checkpoint (robust if you change hidden size)
    hidden = int(state_a_full["fc1.weight"].shape[0])
    flat = int(state_a_full["fc1.weight"].shape[1])
    n_classes = int(state_a_full[f"fc{max(infer_mlp_fc_layer_numbers(state_a_full))}.weight"].shape[0])

    if flat != 28 * 28:
        raise ValueError(f"Expected 28x28 flat dim 784, got {flat}. If this is intentional, adapt input_shape in MLP.")
    # Build model
    model = architectures.MLP(num_classes=n_classes, input_shape=(1, 28, 28), hidden=hidden).to(device)

    # -------------------
        # Dataset / loaders
    # -------------------
    # Use repo utility (supports MNIST + FASHIONMNIST).
    train_full, eval_full, test_set = ds_utils.build_datasets(
        dataset,
        root=args.data_root,
        download=True,
        augment_train=False,
        normalize=(not args.no_normalize),
    )
    train_base = eval_full  # evaluation transform (no augmentation)

    if args.train_eval_split in ("part0", "part1"):
        parts, _, _ = ds_utils.split_dataset_stratified(train_base, num_parts=2, seed=args.seed, exact=False)
        train_eval_set = parts[0 if args.train_eval_split == "part0" else 1]
    else:
        train_eval_set = train_base

    train_loader = torch.utils.data.DataLoader(
        train_eval_set, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers,
        pin_memory=(device.type == "cuda")
    )
    test_loader = torch.utils.data.DataLoader(
        test_set, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers,
        pin_memory=(device.type == "cuda")
    )

    # -------------------
    # Build permutation spec from the checkpoint structure
    # -------------------
    ps = mlp_permutation_spec_from_state(state_a_full)
    spec_keys = set(ps.axes_to_perm.keys())

    state_a = filter_to_spec_keys(state_a_full, spec_keys)
    state_b = filter_to_spec_keys(state_b_full, spec_keys)

    # -------------------
    # Weight matching on CPU tensors
    # -------------------
    final_perm = weight_matching(
        seed=args.seed,
        ps=ps,
        params_a=state_a,
        params_b=state_b,
        max_iter=args.max_iter,
        silent=args.silent,
    )

    perm_path = os.path.join(args.out_dir, f"permutation_seed{args.seed}.pkl")
    with open(perm_path, "wb") as f:
        pickle.dump({k: v.cpu().numpy() for k, v in final_perm.items()}, f)

    # Apply permutation to B (spec keys only)
    state_b_perm = apply_permutation(ps, final_perm, state_b)

    # Move to device for interpolation/eval
    state_a_dev = {k: v.to(device) for k, v in state_a.items()}
    state_b_dev = {k: v.to(device) for k, v in state_b.items()}
    state_b_perm_dev = {k: v.to(device) for k, v in state_b_perm.items()}

    def compute_max_batches(ds_len: int) -> Optional[int]:
        if args.eval_samples is None or args.eval_samples <= 0:
            return None
        return int((args.eval_samples + args.batch_size - 1) // args.batch_size)

    max_batches_train = compute_max_batches(len(train_eval_set))
    max_batches_test = compute_max_batches(len(test_set))

    lambdas = torch.linspace(0.0, 1.0, steps=args.num_lambdas).tolist()

    train_loss_naive, test_loss_naive, train_acc_naive, test_acc_naive = [], [], [], []
    train_loss_perm,  test_loss_perm,  train_acc_perm,  test_acc_perm  = [], [], [], []

    # Naïve interpolation
    for lam in lambdas:
        interp_sd = interpolate_state_dict(state_a_dev, state_b_dev, lam)
        model.load_state_dict(interp_sd, strict=False)

        tl, ta = eval_loss_acc(model, train_loader, device, max_batches=max_batches_train)
        vl, va = eval_loss_acc(model, test_loader, device, max_batches=max_batches_test)

        train_loss_naive.append(tl)
        train_acc_naive.append(ta)
        test_loss_naive.append(vl)
        test_acc_naive.append(va)

    # Permuted interpolation
    for lam in lambdas:
        interp_sd = interpolate_state_dict(state_a_dev, state_b_perm_dev, lam)
        model.load_state_dict(interp_sd, strict=False)

        tl, ta = eval_loss_acc(model, train_loader, device, max_batches=max_batches_train)
        vl, va = eval_loss_acc(model, test_loader, device, max_batches=max_batches_test)

        train_loss_perm.append(tl)
        train_acc_perm.append(ta)
        test_loss_perm.append(vl)
        test_acc_perm.append(va)

    title = f"{dataset} MLP: {Path(args.ckpt_a).name} vs {Path(args.ckpt_b).name} (train_eval={args.train_eval_split})"

    fig = plot_interp_loss(title, lambdas, train_loss_naive, test_loss_naive, train_loss_perm, test_loss_perm)
    loss_path = os.path.join(args.out_dir, "interp_loss.png")
    fig.savefig(loss_path, dpi=300)
    plt.close(fig)

    fig = plot_interp_acc(title, lambdas, train_acc_naive, test_acc_naive, train_acc_perm, test_acc_perm)
    acc_path = os.path.join(args.out_dir, "interp_acc.png")
    fig.savefig(acc_path, dpi=300)
    plt.close(fig)

    results = {
        "dataset": dataset,
        "lambdas": lambdas,
        "train_eval_split": args.train_eval_split,
        "train_loss_naive": train_loss_naive,
        "test_loss_naive": test_loss_naive,
        "train_acc_naive": train_acc_naive,
        "test_acc_naive": test_acc_naive,
        "train_loss_perm": train_loss_perm,
        "test_loss_perm": test_loss_perm,
        "train_acc_perm": train_acc_perm,
        "test_acc_perm": test_acc_perm,
        "permutation_path": perm_path,
        "loss_plot": loss_path,
        "acc_plot": acc_path,
        "hidden": hidden,
        "flat": flat,
        "num_classes": n_classes,
    }

    torch.save(results, os.path.join(args.out_dir, "interp_results.pt"))
    print(results)


if __name__ == "__main__":
    main()
