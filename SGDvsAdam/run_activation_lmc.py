#!/usr/bin/env python3
"""
SGDvsAdam/run_activation_lmc.py

Activation-based permutation matching + LMC interpolation for all checkpoint
pairs from a completed SGDvsAdam experiment.

Mirrors Phase 3 of run_sgd_vs_adam.py but replaces weight_matching with
activation-correlation permutation finding (Hungarian on Pearson correlations
of hidden-unit activations on the training set).

Reads the manifest.json produced by run_sgd_vs_adam.py to discover checkpoints.
Runs all pairs:
  - same-optimizer:   C(n_seeds, 2) per optimizer
  - cross-optimizer:  all SGD seeds × all AdamW seeds

For each pair saves (mirroring weight-matching output):
  interp_results.pt / interp_results.json
  interp_loss.png, interp_acc.png
  permutations.pt

Usage:
  python SGDvsAdam/run_activation_lmc.py \\
    --manifest ./SGDvsAdam_out/resnet20_CIFAR10/manifest.json \\
    --out-dir  ./SGDvsAdam_out/resnet20_CIFAR10/lmc_activation \\
    --data-root ./data
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import re
import sys
from itertools import combinations
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parent
for _p in (str(THIS_DIR), str(REPO_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import architectures   # type: ignore
import datasets        # type: ignore
import utils           # type: ignore
from linear_mode_connectivity.weight_matching_torch import (  # type: ignore
    apply_permutation,
    resnet20_layernorm_permutation_spec,
    resnet50_permutation_spec,
)
from lmc_weight_matching_interp import mlp_permutation_spec_from_state  # type: ignore
from model_stitching.activation_permutation_stitching import (  # type: ignore
    LayerPermutation,
    compute_layer_permutation_from_activations,
    interpolate_state_dict,
    load_ckpt_state_dict,
    normalize_state_dict_keys,
    to_device,
)


# --------------------------------------------------------------------------- #
# Permutation name → hook layer mapping                                        #
# --------------------------------------------------------------------------- #

_P_INNER = re.compile(r"^P_layer(\d+)_(\d+)_inner$")
_P_MLP   = re.compile(r"^P(\d+)$")


def _perm_name_to_hook_resnet20(perm_name: str) -> Tuple[str, Optional[Any], int]:
    """
    Maps a permutation name from resnet20_layernorm_permutation_spec to:
      (layer_name_to_hook, preprocess_fn_or_None, unit_dim)
    """
    if perm_name == "P_bg0":
        return "n1", torch.relu, 1
    if perm_name == "P_bg1":
        return "layer2.0", None, 1
    if perm_name == "P_bg2":
        return "layer3.0", None, 1
    m = _P_INNER.match(perm_name)
    if m:
        layer = int(m.group(1))
        block = int(m.group(2))
        return f"layer{layer}.{block}.n1", torch.relu, 1
    raise KeyError(f"No hook mapping for ResNet20 permutation '{perm_name}'")


def _perm_name_to_hook_mlp(perm_name: str) -> Tuple[str, Optional[Any], int]:
    """
    Maps a permutation name from mlp_permutation_spec_from_state to:
      (layer_name_to_hook, preprocess_fn_or_None, unit_dim)

    P{i} → hook fc{i} output and apply relu (MLP has no named relu modules).
    unit_dim=1 → hidden-unit axis for [N, d] tensors.
    """
    m = _P_MLP.match(perm_name)
    if m:
        i = int(m.group(1))
        return f"fc{i}", torch.relu, 1
    raise KeyError(f"No hook mapping for MLP permutation '{perm_name}'")


_P_INNER2 = re.compile(r"^P_layer(\d+)_(\d+)_inner([12])$")
_P_BG     = re.compile(r"^P_bg(\d+)$")
_BG_TO_LAYER = {"P_bg1": "layer1", "P_bg2": "layer2", "P_bg3": "layer3", "P_bg4": "layer4"}


def _perm_name_to_hook_resnet50(perm_name: str) -> Tuple[str, Optional[Any], int]:
    """
    Maps a ResNet50 permutation name to (layer_name, preprocess, unit_dim).

    P_bg0       → stem bn1 output (hook bn1, apply relu)
    P_bg{1-4}   → whole-stage output (hook layer{1-4}, relu already applied)
    P_layer{l}_{b}_inner1 → after conv1+bn1 (hook layer{l}.{b}.bn1, apply relu)
    P_layer{l}_{b}_inner2 → after conv2+bn2 (hook layer{l}.{b}.bn2, apply relu)
    """
    if perm_name == "P_bg0":
        return "bn1", torch.relu, 1
    if perm_name in _BG_TO_LAYER:
        return _BG_TO_LAYER[perm_name], None, 1  # relu already applied inside block
    m = _P_INNER2.match(perm_name)
    if m:
        layer = int(m.group(1))
        block = int(m.group(2))
        inner = int(m.group(3))
        bn_name = f"layer{layer}.{block}.bn1" if inner == 1 else f"layer{layer}.{block}.bn2"
        return bn_name, torch.relu, 1
    raise KeyError(f"No hook mapping for ResNet50 permutation '{perm_name}'")


# --------------------------------------------------------------------------- #
# Evaluation                                                                   #
# --------------------------------------------------------------------------- #

def has_batch_norm(model: nn.Module) -> bool:
    return any(isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d))
               for m in model.modules())


@torch.no_grad()
def reset_bn_stats(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    max_batches: int = 50,
) -> None:
    """Recalculate BatchNorm running statistics for an interpolated model."""
    bn_mods = [m for m in model.modules()
               if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d))]
    if not bn_mods:
        return
    for m in bn_mods:
        m.reset_running_stats()
        m.momentum = None  # cumulative moving average: 1/n each step
    model.train()
    for bi, (x, _) in enumerate(loader):
        if bi >= max_batches:
            break
        model(x.to(device))
    model.eval()


@torch.no_grad()
def _eval_loss_acc(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    max_batches: Optional[int] = None,
) -> Tuple[float, float]:
    model.eval()
    crit = nn.CrossEntropyLoss(reduction="sum")
    loss_sum = 0.0
    correct = 0
    seen = 0
    for bi, (x, y) in enumerate(loader):
        if max_batches is not None and bi >= max_batches:
            break
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss_sum += float(crit(logits, y).item())
        correct += int((logits.argmax(1) == y).sum().item())
        seen += int(y.numel())
    denom = max(1, seen)
    return loss_sum / denom, correct / denom


# --------------------------------------------------------------------------- #
# Core: one pair                                                               #
# --------------------------------------------------------------------------- #

def run_activation_lmc_pair(
    *,
    arch: str,
    dataset_name: str,
    ckpt_a: str,
    ckpt_b: str,
    out_dir: str,
    data_root: str = "./data",
    batch_size: int = 256,
    num_workers: int = 4,
    match_samples: int = 5000,
    num_lambdas: int = 25,
    eval_samples: int = 0,
    width_multiplier: Optional[int] = None,
    shortcut_option: Optional[str] = None,
    norm: Optional[str] = None,
    silent: bool = False,
    bn_reset_batches: int = 50,
) -> Dict[str, Any]:
    """
    Activation-based permutation matching + interpolation for one model pair.
    Returns a results dict in the same format as run_weight_matching_interp.
    """
    os.makedirs(out_dir, exist_ok=True)
    device = utils.get_device()

    # ---- load states ----
    state_a_raw = normalize_state_dict_keys(load_ckpt_state_dict(ckpt_a))
    state_b_raw = normalize_state_dict_keys(load_ckpt_state_dict(ckpt_b))

    # ---- infer arch params from checkpoint (ResNet only) ----
    arch_l = arch.strip().lower()
    if arch_l == "resnet20":
        if width_multiplier is None:
            width_multiplier = max(1, int(state_a_raw["conv1.weight"].shape[0] // 16))
        if shortcut_option is None:
            shortcut_option = "C" if any(k.endswith("shortcut.0.weight") for k in state_a_raw) else "A"
        if norm is None:
            keys = state_a_raw.keys()
            if any(k.endswith("running_mean") for k in keys):
                norm = "bn"
            elif any(".ln.weight" in k for k in keys):
                norm = "flax_ln"
            else:
                norm = "ln"

    stats = datasets.DATASET_STATS[dataset_name]
    num_classes = int(stats["num_classes"])
    in_channels = int(stats["in_channels"])

    # ---- build models ----
    def _build() -> nn.Module:
        if arch_l == "mlp":
            return architectures.build_model(
                arch, num_classes=num_classes, in_channels=in_channels,
            ).to(device)
        if arch_l == "resnet50":
            return architectures.build_model(
                arch, num_classes=num_classes, in_channels=in_channels, norm=norm or "bn",
            ).to(device)
        return architectures.build_model(
            arch,
            num_classes=num_classes,
            in_channels=in_channels,
            norm=norm,
            width_multiplier=width_multiplier,
            shortcut_option=shortcut_option,
        ).to(device)

    model_a = _build()
    model_b = _build()
    model_eval = _build()

    model_keys = set(model_a.state_dict().keys())
    state_a = {k: v for k, v in state_a_raw.items() if k in model_keys}
    state_b = {k: v for k, v in state_b_raw.items() if k in model_keys}

    model_a.load_state_dict(state_a, strict=True)
    model_b.load_state_dict(state_b, strict=True)

    # ---- data loaders ----
    _train_full, eval_full, test_ds = datasets.build_datasets(
        dataset_name, root=data_root, download=True,
        augment_train=False, normalize=True,
    )
    pin = device.type == "cuda"

    # Matching loader: first match_samples of train (no augmentation).
    match_ds = eval_full
    if match_samples > 0 and match_samples < len(eval_full):
        from torch.utils.data import Subset
        match_ds = Subset(eval_full, list(range(match_samples)))

    match_loader = DataLoader(
        match_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=pin,
    )
    train_loader = DataLoader(
        eval_full, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=pin,
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=pin,
    )

    def _max_batches(n: int) -> Optional[int]:
        if eval_samples <= 0:
            return None
        return (eval_samples + batch_size - 1) // batch_size

    max_b_train = _max_batches(len(eval_full))
    max_b_test = _max_batches(len(test_ds))

    # ---- permutation spec ----
    if arch_l == "resnet20":
        ps = resnet20_layernorm_permutation_spec(
            shortcut_option=shortcut_option, state_dict=state_a,
        )
        _hook_fn = _perm_name_to_hook_resnet20
    elif arch_l == "resnet50":
        ps = resnet50_permutation_spec(state_dict=state_a)
        _hook_fn = _perm_name_to_hook_resnet50
    elif arch_l == "mlp":
        ps = mlp_permutation_spec_from_state(state_a)
        _hook_fn = _perm_name_to_hook_mlp
    else:
        raise ValueError(f"Unsupported arch for activation LMC: {arch}")

    # ---- activation-based matching ----
    perm: Dict[str, torch.Tensor] = {}
    for p_name in sorted(ps.perm_to_axes.keys()):
        layer_name, preprocess, unit_dim = _hook_fn(p_name)
        if not silent:
            print(f"  [act-match] {p_name} → hook '{layer_name}'")
        lp: LayerPermutation = compute_layer_permutation_from_activations(
            model_a=model_a,
            model_b=model_b,
            loader=match_loader,
            layer_name=layer_name,
            device=device,
            unit_dim=unit_dim,
            preprocess=preprocess,
        )
        perm[p_name] = lp.perm_a_to_b

    # Save permutations
    perm_path = os.path.join(out_dir, "permutations_act.pkl")
    with open(perm_path, "wb") as f:
        pickle.dump({k: v.cpu().numpy() for k, v in perm.items()}, f)

    # ---- apply permutation to B ----
    state_b_perm = apply_permutation(ps, perm, state_b)

    a_dev = to_device(state_a, device)
    b_dev = to_device(state_b, device)
    b_perm_dev = to_device(state_b_perm, device)

    # ---- interpolation evaluation ----
    lambdas = torch.linspace(0.0, 1.0, steps=num_lambdas).tolist()

    train_loss_naive, test_loss_naive = [], []
    train_acc_naive,  test_acc_naive  = [], []
    train_loss_perm,  test_loss_perm  = [], []
    train_acc_perm,   test_acc_perm   = [], []

    for lam in lambdas:
        sd = interpolate_state_dict(a_dev, b_dev, float(lam))
        model_eval.load_state_dict(sd, strict=True)
        if bn_reset_batches > 0 and has_batch_norm(model_eval):
            reset_bn_stats(model_eval, train_loader, device, max_batches=bn_reset_batches)
        tl, ta = _eval_loss_acc(model_eval, train_loader, device, max_b_train)
        vl, va = _eval_loss_acc(model_eval, test_loader,  device, max_b_test)
        train_loss_naive.append(tl); train_acc_naive.append(ta)
        test_loss_naive.append(vl);  test_acc_naive.append(va)

    for lam in lambdas:
        sd = interpolate_state_dict(a_dev, b_perm_dev, float(lam))
        model_eval.load_state_dict(sd, strict=True)
        if bn_reset_batches > 0 and has_batch_norm(model_eval):
            reset_bn_stats(model_eval, train_loader, device, max_batches=bn_reset_batches)
        tl, ta = _eval_loss_acc(model_eval, train_loader, device, max_b_train)
        vl, va = _eval_loss_acc(model_eval, test_loader,  device, max_b_test)
        train_loss_perm.append(tl); train_acc_perm.append(ta)
        test_loss_perm.append(vl);  test_acc_perm.append(va)

    # ---- plots ----
    title = f"{dataset_name} {arch}: {Path(ckpt_a).parent.name} vs {Path(ckpt_b).parent.name} [act]"

    fig, ax = plt.subplots()
    ax.plot(lambdas, train_loss_naive, color="grey",  linewidth=2, alpha=0.85)
    ax.plot(lambdas, test_loss_naive,  color="grey",  linewidth=2, alpha=0.85, linestyle="dashed")
    ax.plot(lambdas, train_loss_perm,  linewidth=2, marker="^")
    ax.plot(lambdas, test_loss_perm,   linewidth=2, marker="^", linestyle="dashed")
    ax.set_xlabel(r"$\lambda$")
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Model $A$", "Model $B$"])
    ax.set_ylabel("Loss")
    ax.set_title(title)
    ax.legend(["Train, naive", "Test, naive", "Train, permuted", "Test, permuted"], loc="best")
    fig.tight_layout()
    loss_plot = os.path.join(out_dir, "interp_loss.png")
    loss_plot_pdf = os.path.join(out_dir, "interp_loss.pdf")
    fig.savefig(loss_plot, dpi=300)
    fig.savefig(loss_plot_pdf)
    plt.close(fig)

    fig, ax = plt.subplots()
    ax.plot(lambdas, [x * 100 for x in train_acc_naive], color="grey",  linewidth=2, alpha=0.85)
    ax.plot(lambdas, [x * 100 for x in test_acc_naive],  color="grey",  linewidth=2, alpha=0.85, linestyle="dashed")
    ax.plot(lambdas, [x * 100 for x in train_acc_perm],  linewidth=2, marker="^")
    ax.plot(lambdas, [x * 100 for x in test_acc_perm],   linewidth=2, marker="^", linestyle="dashed")
    ax.set_xlabel(r"$\lambda$")
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Model $A$", "Model $B$"])
    ax.set_ylabel("Accuracy (%)")
    ax.set_title(title)
    ax.legend(["Train, naive", "Test, naive", "Train, permuted", "Test, permuted"], loc="best")
    fig.tight_layout()
    acc_plot = os.path.join(out_dir, "interp_acc.png")
    acc_plot_pdf = os.path.join(out_dir, "interp_acc.pdf")
    fig.savefig(acc_plot, dpi=300)
    fig.savefig(acc_plot_pdf)
    plt.close(fig)

    # ---- save results ----
    mid = len(lambdas) // 2
    results: Dict[str, Any] = {
        "arch": arch,
        "dataset": dataset_name,
        "matching": "activation",
        "match_samples": match_samples,
        "ckpt_a": ckpt_a,
        "ckpt_b": ckpt_b,
        "lambdas": lambdas,
        "train_loss_naive": train_loss_naive,
        "test_loss_naive":  test_loss_naive,
        "train_acc_naive":  train_acc_naive,
        "test_acc_naive":   test_acc_naive,
        "train_loss_perm":  train_loss_perm,
        "test_loss_perm":   test_loss_perm,
        "train_acc_perm":   train_acc_perm,
        "test_acc_perm":    test_acc_perm,
        "permutation_path": perm_path,
        "loss_plot":     loss_plot,
        "loss_plot_pdf": loss_plot_pdf,
        "acc_plot":      acc_plot,
        "acc_plot_pdf":  acc_plot_pdf,
        "summary": {
            "mid_test_loss_naive": float(test_loss_naive[mid]),
            "mid_test_loss_perm":  float(test_loss_perm[mid]),
        },
        "width_multiplier": width_multiplier,
        "shortcut_option":  shortcut_option,
        "norm":             norm,
    }

    torch.save(results, os.path.join(out_dir, "interp_results.pt"))
    with open(os.path.join(out_dir, "interp_results.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    if not silent:
        print(f"  mid test loss — naive: {test_loss_naive[mid]:.4f}  perm: {test_loss_perm[mid]:.4f}")

    return results


# --------------------------------------------------------------------------- #
# Orchestration: all pairs from manifest                                       #
# --------------------------------------------------------------------------- #

def run_all_pairs(
    *,
    manifest_path: str,
    out_dir: str,
    data_root: str,
    batch_size: int,
    num_workers: int,
    match_samples: int,
    num_lambdas: int,
    eval_samples: int,
    silent: bool,
    bn_reset_batches: int = 50,
) -> Dict[str, Any]:
    with open(manifest_path, encoding="utf-8") as f:
        manifest = json.load(f)

    arch         = str(manifest["arch"])
    dataset_name = str(manifest["dataset"])
    seeds: List[int] = [int(s) for s in manifest["seeds"]]

    manifest_dir = Path(manifest_path).parent

    def _resolve_ckpt(raw: str) -> str:
        # 1. Absolute path that exists.
        p = Path(raw)
        if p.is_absolute() and p.exists():
            return str(p)
        # 2. Relative to cwd.
        if p.exists():
            return str(p.resolve())
        # 3. Progressively strip leading path components and try relative to
        #    the manifest's directory. Handles out-dir name mismatches between
        #    the machine that ran training and the current machine.
        parts = p.parts
        for i in range(len(parts)):
            candidate = manifest_dir.joinpath(*parts[i:])
            if candidate.exists():
                return str(candidate.resolve())
        raise FileNotFoundError(
            f"Checkpoint not found: '{raw}'\n"
            f"  tried cwd-relative: {p.resolve()}\n"
            f"  tried manifest-relative suffixes under: {manifest_dir}"
        )

    # Collect final checkpoint paths.
    final_ckpts: Dict[str, Dict[int, str]] = {}
    for opt_name, seed_dict in manifest["final"].items():
        final_ckpts[opt_name] = {}
        for seed_key, info in seed_dict.items():
            seed = int(seed_key.replace("seed_", ""))
            final_ckpts[opt_name][seed] = _resolve_ckpt(str(info["ckpt"]))

    optimizers = list(final_ckpts.keys())
    out_root = Path(out_dir)
    all_results: Dict[str, Any] = {
        "same_optimizer": {opt: {} for opt in optimizers},
        "cross_optimizer": {},
    }

    common_kw = dict(
        arch=arch,
        dataset_name=dataset_name,
        data_root=data_root,
        batch_size=batch_size,
        num_workers=num_workers,
        match_samples=match_samples,
        num_lambdas=num_lambdas,
        eval_samples=eval_samples,
        silent=silent,
        bn_reset_batches=bn_reset_batches,
    )

    # --- same-optimizer pairs ---
    for opt_name in optimizers:
        for s1, s2 in combinations(seeds, 2):
            pair_tag = f"{opt_name}_seed{s1}_vs_seed{s2}"
            pair_dir = out_root / "same_optimizer" / opt_name / f"seed{s1}_vs_seed{s2}"
            if (pair_dir / "interp_results.pt").exists():
                print(f"[skip] {pair_tag}")
                continue
            print(f"\n[act-lmc] {pair_tag}")
            all_results["same_optimizer"][opt_name][f"seed{s1}_vs_seed{s2}"] = run_activation_lmc_pair(
                ckpt_a=final_ckpts[opt_name][s1],
                ckpt_b=final_ckpts[opt_name][s2],
                out_dir=str(pair_dir),
                **common_kw,
            )

    # --- cross-optimizer pairs ---
    if "sgd" in final_ckpts and "adamw" in final_ckpts:
        for s_sgd in seeds:
            for s_adamw in seeds:
                pair_tag = f"sgd_seed{s_sgd}_vs_adamw_seed{s_adamw}"
                pair_dir = out_root / "cross_optimizer" / pair_tag
                if (pair_dir / "interp_results.pt").exists():
                    print(f"[skip] {pair_tag}")
                    continue
                print(f"\n[act-lmc] {pair_tag}")
                all_results["cross_optimizer"][pair_tag] = run_activation_lmc_pair(
                    ckpt_a=final_ckpts["sgd"][s_sgd],
                    ckpt_b=final_ckpts["adamw"][s_adamw],
                    out_dir=str(pair_dir),
                    **common_kw,
                )

    out_root.mkdir(parents=True, exist_ok=True)
    with open(out_root / "results_all.json", "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n[done] results saved to {out_root / 'results_all.json'}")
    return all_results


# --------------------------------------------------------------------------- #
# CLI                                                                          #
# --------------------------------------------------------------------------- #

def main() -> None:
    p = argparse.ArgumentParser(
        description="Activation-based LMC for all pairs in a completed SGDvsAdam run."
    )
    p.add_argument("--manifest",   type=str, required=True,
                   help="Path to manifest.json from run_sgd_vs_adam.py")
    p.add_argument("--out-dir",    type=str, default=None,
                   help="Output directory. Defaults to <manifest_dir>/lmc_activation")
    p.add_argument("--data-root",  type=str, default="./data")
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--num-workers",type=int, default=4)
    p.add_argument("--match-samples", type=int, default=5000,
                   help="Samples used to compute activation correlations. 0=all.")
    p.add_argument("--num-lambdas",   type=int, default=25)
    p.add_argument("--eval-samples",  type=int, default=0,
                   help="Samples for interpolation eval. 0=full dataset.")
    p.add_argument("--silent", action="store_true")
    p.add_argument("--bn-reset-batches", type=int, default=50,
                   help="Batches used to recalculate BN stats after each interpolation. "
                        "0 = disabled.")
    args = p.parse_args()

    manifest_path = Path(args.manifest).resolve()
    out_dir = args.out_dir or str(manifest_path.parent / "lmc_activation")

    run_all_pairs(
        manifest_path=str(manifest_path),
        out_dir=out_dir,
        data_root=args.data_root,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        match_samples=args.match_samples,
        num_lambdas=args.num_lambdas,
        eval_samples=args.eval_samples,
        silent=args.silent,
        bn_reset_batches=args.bn_reset_batches,
    )


if __name__ == "__main__":
    main()
