# cifar_resnet20_ln_weight_matching_interp.py
from __future__ import annotations

import argparse
import os
import pickle
from pathlib import Path
from typing import Dict, Tuple, Optional

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms

import utils
import architectures

from linear_mode_connectivity.weight_matching_torch import (
    apply_permutation,
    resnet20_layernorm_permutation_spec,
    weight_matching,
)


"""
HOW TO RUN:
Full Regime: seed 0 vs seed 1
python cifar_resnet20_ln_weight_matching_interp.py --dataset CIFAR10 --ckpt-a ./runs_resnet20_ln_warmcos/CIFAR10/full/seed_0/resnet20_CIFAR10_full_seed0_best.pth --ckpt-b ./runs_resnet20_ln_warmcos/CIFAR10/full/seed_1/resnet20_CIFAR10_full_seed1_best.pth --width-multiplier 1 --shortcut-option C --norm flax_ln --out-dir ./weight_matching_out/CIFAR10_full_seed0_seed1

Disjoint Regime: subset A vs subset B
python linear_mode_connectivity/cifar_resnet20_ln_weight_matching_interp.py --dataset CIFAR10 --ckpt-a ../runs_resnet20_ln_warmcos_1/CIFAR10/disjoint/seed_0/subset_A/resnet20_CIFAR10_subsetA_seed0_best.pth --ckpt-b ../runs_resnet20_ln_warmcos_1/CIFAR10/disjoint/seed_0/subset_B/resnet20_CIFAR10_subsetB_seed0_best.pth --width-multiplier 1 --shortcut-option C --norm flax_ln --out-dir ./weight_matching_out/CIFAR10_disjoint_seed0_subA_subB


==== new command to:

export PYTHONPATH="$(pwd)" 
for x in 32 ; do 
    python linear_mode_connectivity/cifar_resnet20_ln_weight_matching_interp.py \
    --dataset CIFAR10 \
    --ckpt-a ./runs_resnet20_${x}/CIFAR10/disjoint/seed_0/subset_A/resnet20_CIFAR10_seed0_subsetA_best.pth \
    --ckpt-b ./runs_resnet20_${x}/CIFAR10/disjoint/seed_0/subset_B/resnet20_CIFAR10_seed0_subsetB_best.pth \
    --width-multiplier "$x" \
    --shortcut-option C \
    --norm flax_ln \
    --out-dir ./weight_matching_out/resnet20_${x}/CIFAR10/disjoint
done

"""

DATASET_STATS = {
    "CIFAR10": {
        "mean": (0.49139968, 0.48215841, 0.44653091),
        "std":  (0.24703223, 0.24348513, 0.26158784),
        "num_classes": 10,
    },
    "CIFAR100": {
        "mean": (0.50707516, 0.48654887, 0.44091784),
        "std":  (0.26733429, 0.25643846, 0.27615047),
        "num_classes": 100,
    },
}


# add near the top (after imports are fine)
def _setup_plotting_style() -> None:
    # apply repo-wide style (rcParams) if your utils provides it
    try:
        utils.apply_stitching_trend_style()  # type: ignore[attr-defined]

        # optional: also force the color cycle from your palette
        if hasattr(utils, "get_deep_palette"):
            palette = list(utils.get_deep_palette())  # type: ignore[attr-defined]
            if palette:
                from cycler import cycler
                plt.rcParams["axes.prop_cycle"] = cycler(color=palette)
    except Exception as e:
        print(f"[WARN] Could not apply plotting style from utils: {e}")

def _strip_module_prefix(state: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    # Handles DDP checkpoints that prefix keys with "module."
    if not state:
        return state
    if any(k.startswith("module.") for k in state.keys()):
        return {k[len("module."):]: v for k, v in state.items()}
    return state


def load_ckpt_state_dict(path: str) -> Dict[str, torch.Tensor]:
    obj = torch.load(path, map_location="cpu")
    if isinstance(obj, dict) and "state_dict" in obj:
        sd = obj["state_dict"]
    elif isinstance(obj, dict):
        # allow raw state_dict saved directly
        sd = obj
    else:
        raise ValueError(f"Unrecognized checkpoint format at: {path}")
    return _strip_module_prefix(sd)


def filter_to_spec_keys(state: Dict[str, torch.Tensor], spec_keys: set[str]) -> Dict[str, torch.Tensor]:
    return {k: v for k, v in state.items() if k in spec_keys}


def interpolate_state_dict(
    a: Dict[str, torch.Tensor],
    b: Dict[str, torch.Tensor],
    lam: float,
) -> Dict[str, torch.Tensor]:
    # Robust: require identical keysets for interpolation
    if a.keys() != b.keys():
        missing_in_b = sorted(set(a.keys()) - set(b.keys()))
        missing_in_a = sorted(set(b.keys()) - set(a.keys()))
        raise KeyError(
            "State dict keysets differ; cannot interpolate safely.\n"
            f"Missing in B: {missing_in_b[:20]}{' ...' if len(missing_in_b) > 20 else ''}\n"
            f"Missing in A: {missing_in_a[:20]}{' ...' if len(missing_in_a) > 20 else ''}\n"
        )

    out: Dict[str, torch.Tensor] = {}
    for k in a.keys():
        va, vb = a[k], b[k]
        if va.dtype.is_floating_point:
            out[k] = (1.0 - lam) * va + lam * vb
        else:
            # e.g. BN num_batches_tracked (int) => keep from A
            out[k] = va
    return out


@torch.no_grad()
def eval_loss_acc(
    model: torch.nn.Module,
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

    return total_loss / max(1, total_seen), total_correct / max(1, total_seen)


def plot_interp_loss(epoch_label, lambdas, train_loss_naive, test_loss_naive, train_loss_perm, test_loss_perm, width_multiplier):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(lambdas, train_loss_naive, linestyle="dashed", alpha=0.5, linewidth=2, label="Train, naïve interp.")
    ax.plot(lambdas, test_loss_naive, linestyle="dashed", alpha=0.5, linewidth=2, label="Test, naïve interp.")
    ax.plot(lambdas, train_loss_perm, linestyle="solid", linewidth=2, label="Train, permuted interp.")
    ax.plot(lambdas, test_loss_perm, linestyle="solid", linewidth=2, label="Test, permuted interp.")
    ax.set_xlabel(r"$\lambda$")
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Model A", "Model B"])
    ax.set_ylabel("Loss")
    #ax.set_title(f"Loss Barrier : width {width_multiplier}")
    ax.legend(loc="upper right", framealpha=0.5)
    fig.tight_layout()
    return fig


def plot_interp_acc(epoch_label, lambdas, train_acc_naive, test_acc_naive, train_acc_perm, test_acc_perm):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(lambdas, train_acc_naive, linestyle="dashed", alpha=0.5, linewidth=2, label="Train, naïve interp.")
    ax.plot(lambdas, test_acc_naive, linestyle="dashed", alpha=0.5, linewidth=2, label="Test, naïve interp.")
    ax.plot(lambdas, train_acc_perm, linestyle="solid", linewidth=2, label="Train, permuted interp.")
    ax.plot(lambdas, test_acc_perm, linestyle="solid", linewidth=2, label="Test, permuted interp.")
    ax.set_xlabel(r"$\lambda$")
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Model A", "Model B"])
    ax.set_ylabel("Accuracy")
    ax.set_title(f"Accuracy between the two models ({epoch_label})")
    ax.legend(loc="lower right", framealpha=0.5)
    fig.tight_layout()
    return fig


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="CIFAR10", choices=["CIFAR10", "CIFAR100"])
    parser.add_argument("--data-root", type=str, default="./data")

    parser.add_argument("--ckpt-a", type=str, required=True, help="Path to checkpoint for model A (.pth)")
    parser.add_argument("--ckpt-b", type=str, required=True, help="Path to checkpoint for model B (.pth)")

    parser.add_argument("--width-multiplier", type=int, default=1)
    parser.add_argument("--shortcut-option", type=str, default="C", choices=["A", "B", "C"])
    parser.add_argument("--norm", type=str, default="flax_ln")  # should match train_resnet.py

    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-iter", type=int, default=100)
    parser.add_argument("--num-lambdas", type=int, default=25)

    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--eval-samples", type=int, default=1000, help="<=0 means full dataset")

    parser.add_argument("--out-dir", type=str, default="./weight_matching_out")
    parser.add_argument("--silent", action="store_true")
    args = parser.parse_args()
    _setup_plotting_style()

    os.makedirs(args.out_dir, exist_ok=True)
    device = utils.get_device()

    stats = DATASET_STATS[args.dataset]
    normalize = transforms.Normalize(stats["mean"], stats["std"])
    eval_transform = transforms.Compose([transforms.ToTensor(), normalize])

    if args.dataset == "CIFAR10":
        train_set = torchvision.datasets.CIFAR10(root=args.data_root, train=True, download=True, transform=eval_transform)
        test_set = torchvision.datasets.CIFAR10(root=args.data_root, train=False, download=True, transform=eval_transform)
    else:
        train_set = torchvision.datasets.CIFAR100(root=args.data_root, train=True, download=True, transform=eval_transform)
        test_set = torchvision.datasets.CIFAR100(root=args.data_root, train=False, download=True, transform=eval_transform)

    train_loader = torch.utils.data.DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )
    test_loader = torch.utils.data.DataLoader(
        test_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )

    num_classes = int(stats["num_classes"])
    model = architectures.build_model(
        "resnet20",
        num_classes=num_classes,
        norm=args.norm,
        width_multiplier=int(args.width_multiplier),
        shortcut_option=str(args.shortcut_option),
    ).to(device)

    # Load A/B state dicts
    state_a_full = load_ckpt_state_dict(args.ckpt_a)
    state_b_full = load_ckpt_state_dict(args.ckpt_b)

    # Build permutation spec based on your actual trained checkpoint naming/layout
    ps = resnet20_layernorm_permutation_spec(
        shortcut_option=str(args.shortcut_option),
        state_dict=state_a_full,
    )
    spec_keys = set(ps.axes_to_perm.keys())

    # Only the subset involved in matching goes into Hungarian; keep full dicts for eval/interp.
    state_a_match = filter_to_spec_keys(state_a_full, spec_keys)
    state_b_match = filter_to_spec_keys(state_b_full, spec_keys)

    # Weight matching (CPU)
    final_perm = weight_matching(
        seed=args.seed,
        ps=ps,
        params_a=state_a_match,
        params_b=state_b_match,
        max_iter=args.max_iter,
        silent=args.silent,
    )

    # Save permutation
    perm_path = os.path.join(args.out_dir, f"permutation_seed{args.seed}.pkl")
    with open(perm_path, "wb") as f:
        pickle.dump({k: v.cpu().numpy() for k, v in final_perm.items()}, f)

    # Apply permutation to the FULL B state dict (so interpolation loads are complete)
    state_b_perm_full = apply_permutation(ps, final_perm, state_b_full)

    # Move full dicts to device for fast interpolation
    state_a_dev = {k: v.to(device) for k, v in state_a_full.items()}
    state_b_dev = {k: v.to(device) for k, v in state_b_full.items()}
    state_b_perm_dev = {k: v.to(device) for k, v in state_b_perm_full.items()}

    def compute_max_batches(ds_len: int) -> Optional[int]:
        if args.eval_samples is None or args.eval_samples <= 0:
            return None
        return int((args.eval_samples + args.batch_size - 1) // args.batch_size)

    max_batches_train = compute_max_batches(len(train_set))
    max_batches_test = compute_max_batches(len(test_set))

    lambdas = torch.linspace(0.0, 1.0, steps=args.num_lambdas).tolist()

    train_loss_naive, test_loss_naive, train_acc_naive, test_acc_naive = [], [], [], []
    train_loss_perm,  test_loss_perm,  train_acc_perm,  test_acc_perm  = [], [], [], []

    # Naïve interpolation A <-> B
    for lam in lambdas:
        interp_sd = interpolate_state_dict(state_a_dev, state_b_dev, float(lam))
        model.load_state_dict(interp_sd, strict=True)

        tl, ta = eval_loss_acc(model, train_loader, device, max_batches=max_batches_train)
        vl, va = eval_loss_acc(model, test_loader, device, max_batches=max_batches_test)

        train_loss_naive.append(tl)
        train_acc_naive.append(ta)
        test_loss_naive.append(vl)
        test_acc_naive.append(va)

    # Permuted interpolation A <-> P(B)
    for lam in lambdas:
        interp_sd = interpolate_state_dict(state_a_dev, state_b_perm_dev, float(lam))
        model.load_state_dict(interp_sd, strict=True)

        tl, ta = eval_loss_acc(model, train_loader, device, max_batches=max_batches_train)
        vl, va = eval_loss_acc(model, test_loader, device, max_batches=max_batches_test)

        train_loss_perm.append(tl)
        train_acc_perm.append(ta)
        test_loss_perm.append(vl)
        test_acc_perm.append(va)

    epoch_label = f"{Path(args.ckpt_a).name} vs {Path(args.ckpt_b).name}"

    fig = plot_interp_loss(epoch_label, lambdas, train_loss_naive, test_loss_naive, train_loss_perm, test_loss_perm, int(args.width_multiplier))
    loss_path = os.path.join(args.out_dir, "interp_loss.png")
    fig.savefig(loss_path, dpi=300)
    plt.close(fig)

    fig = plot_interp_acc(epoch_label, lambdas, train_acc_naive, test_acc_naive, train_acc_perm, test_acc_perm)
    acc_path = os.path.join(args.out_dir, "interp_acc.png")
    fig.savefig(acc_path, dpi=300)
    plt.close(fig)

    results = {
        "lambdas": lambdas,
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
        "dataset": args.dataset,
        "width_multiplier": int(args.width_multiplier),
        "shortcut_option": str(args.shortcut_option),
        "norm": str(args.norm),
        "ckpt_a": str(args.ckpt_a),
        "ckpt_b": str(args.ckpt_b),
    }

    torch.save(results, os.path.join(args.out_dir, "interp_results.pt"))
    print(results)


if __name__ == "__main__":
    main()
