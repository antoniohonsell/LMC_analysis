#!/usr/bin/env python3
"""
resnet20_activation_stitching.py

Activation-based "model stitching" between two ResNet20 (LayerNorm/Flax-style) models
trained by train_resnet.py (runs_resnet20_ln_warmcos/).

Mirrors model_stitching/mlp_activation_stitching.py outputs:
  - baselines for A, B, and B_perm
  - stitching curves (naive vs perm)
  - optional interpolation curves (naive vs perm)
  - plots + csv + results.json + SUMMARY.txt


usually I need to put this before running :
PYTHONPATH="$(pwd)"
where pwd is the root  

HOW TO RUN :
export PYTHONPATH="$(pwd)" 
for x in 16 ; do
    python model_stitching/resnet20_activation_stitching.py \
    --dataset CIFAR100 \
    --runs-root ./runs_resnet20_${x} \
    --regime disjoint \
    --base-seed 0 \
    --which best \
    --match-split train_eval \
    --match-samples 10000 \
    --do-interp \
    --out-dir ./activation_out/CIFAR100/activation_stitching_out_cifar100_resnet20_${x}
  done
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

import architectures
import train_resnet
from linear_mode_connectivity.weight_matching_torch import (
    resnet20_layernorm_permutation_spec, apply_permutation)

from model_stitching.activation_permutation_stitching import (
    LayerPermutation,
    compute_layer_permutation_from_activations,
    interpolate_state_dict,
    load_ckpt_state_dict,
    normalize_state_dict_keys,
    to_device,
)

from pathlib import Path
import sys
import matplotlib.pyplot as plt

def _setup_plotting_style() -> list[str]:
    # Ensure repo root is importable (removes the need for PYTHONPATH=$(pwd))
    repo_root = Path(__file__).resolve().parents[1]  # repo_root/model_stitching/this_file.py
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    try:
        import utils  # your repo's utils.py
        utils.apply_stitching_trend_style()          # sets rcParams / style
        palette = list(utils.get_deep_palette())     # list of colors

        # If your style function doesn't already set the prop_cycle, do it here.
        if palette:
            from cycler import cycler
            plt.rcParams["axes.prop_cycle"] = cycler(color=palette)

        return palette
    except Exception as e:
        print(f"[WARN] Could not apply utils plotting style: {e}")
        return []


# ------------------------
# eval
# ------------------------
@torch.no_grad()
def eval_loss_acc(
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
    for b_ix, (x, y) in enumerate(loader):
        if max_batches is not None and b_ix >= max_batches:
            break
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss_sum += float(crit(logits, y).item())
        correct += int((logits.argmax(dim=1) == y).sum().item())
        seen += int(y.numel())
    denom = max(1, seen)
    return loss_sum / denom, correct / denom


def _device_from_utils_fallback() -> torch.device:
    try:
        import utils  # type: ignore
        return utils.get_device()  # type: ignore[attr-defined]
    except Exception:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _find_single_checkpoint(run_dir: Path, which: str) -> Path:
    # train_resnet.py uses *_best.pth and *_final.pth :contentReference[oaicite:9]{index=9}
    pats = [f"*_{which}.pth", f"*{which}.pth"]
    matches: List[Path] = []
    for p in pats:
        matches.extend(sorted(run_dir.glob(p)))
    if not matches:
        raise FileNotFoundError(f"No checkpoint matching {pats} found in {run_dir}")
    matches = sorted(matches, key=lambda x: len(x.name))
    return matches[0]


def _find_indices_file(ds_root: Path, dataset: str) -> Path:
    matches = sorted(ds_root.glob(f"indices_{dataset}_splitseed*_subsetseed*_val*.pt"))
    if not matches:
        matches = sorted(ds_root.glob("indices_*.pt"))
    if not matches:
        raise FileNotFoundError(f"No indices_*.pt file found in {ds_root} (expected output of train_resnet.py).")
    return matches[0]


def _save_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, sort_keys=True)


def _plot_stitching(
    *,
    title: str,
    cuts: List[int],
    metrics: Dict[str, Dict[str, List[float]]],
    out_dir: Path,
) -> None:
    fig = plt.figure()
    ax = fig.add_subplot(111)
    for split, d in metrics.items():
        ax.plot(cuts, d["acc_naive"], linestyle="dashed", alpha=0.5, linewidth=2, label=f"{split} naive")
        ax.plot(cuts, d["acc_perm"], linestyle="solid", linewidth=2, label=f"{split} perm")
    ax.set_xlabel("cut k (take first k sections from A)")
    ax.set_ylabel("accuracy")
    ax.set_title(title)
    ax.set_xticks(cuts)
    ax.legend(framealpha=0.6)
    fig.tight_layout()
    fig.savefig(out_dir / "stitch_acc.png", dpi=300)
    plt.close(fig)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    for split, d in metrics.items():
        ax.plot(cuts, d["loss_naive"], linestyle="dashed", alpha=0.5, linewidth=2, label=f"{split} naive")
        ax.plot(cuts, d["loss_perm"], linestyle="solid", linewidth=2, label=f"{split} perm")
    ax.set_xlabel("cut k (take first k sections from A)")
    ax.set_ylabel("Loss")
    ax.set_title(title)
    ax.set_xticks(cuts)
    ax.legend(framealpha=0.6)
    fig.tight_layout()
    fig.savefig(out_dir / "stitch_loss.png", dpi=300)
    plt.close(fig)


def _plot_interpolation(
    *,
    title: str,
    lambdas: List[float],
    metrics: Dict[str, Dict[str, List[float]]],
    out_dir: Path,
) -> None:
    fig = plt.figure()
    ax = fig.add_subplot(111)
    for split, d in metrics.items():
        ax.plot(lambdas, d["acc_naive"], linestyle="dashed", alpha=0.5, linewidth=2, label=f"{split} naive")
        ax.plot(lambdas, d["acc_perm"], linestyle="solid", linewidth=2, label=f"{split} perm")
    ax.set_xlabel(r"$\lambda$  (0 = A, 1 = B)")
    ax.set_ylabel("accuracy")
    ax.set_title(title)
    ax.legend(framealpha=0.6)
    fig.tight_layout()
    fig.savefig(out_dir / "interp_acc.png", dpi=300)
    plt.close(fig)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    for split, d in metrics.items():
        ax.plot(lambdas, d["loss_naive"], linestyle="dashed", alpha=0.5, linewidth=2, label=f"{split} naive")
        ax.plot(lambdas, d["loss_perm"], linestyle="solid", linewidth=2, label=f"{split} perm")
    ax.set_xlabel(r"$\lambda$  (0 = A, 1 = B)")
    ax.set_ylabel("cross-entropy loss")
    ax.set_title(title)
    ax.legend(framealpha=0.6)
    fig.tight_layout()
    fig.savefig(out_dir / "interp_loss.png", dpi=300)
    plt.close(fig)


def _plot_corr_diagnostics(
    corr: torch.Tensor,
    perm_a_to_b: torch.Tensor,
    out_prefix: Path,
    max_side: int = 128,
) -> Dict[str, float]:
    corr = corr.to(dtype=torch.float64, device="cpu")
    d = corr.shape[0]
    diag_before = corr.diag()
    diag_after = corr[torch.arange(d), perm_a_to_b]
    stats = {
        "d": float(d),
        "diag_before_mean": float(diag_before.mean().item()),
        "diag_after_mean": float(diag_after.mean().item()),
        "diag_before_median": float(diag_before.median().item()),
        "diag_after_median": float(diag_after.median().item()),
        "diag_after_min": float(diag_after.min().item()),
        "diag_after_max": float(diag_after.max().item()),
    }

    idx = torch.arange(d)
    if d > max_side:
        idx = torch.linspace(0, d - 1, steps=max_side).round().to(dtype=torch.long)
    C = corr.index_select(0, idx).index_select(1, idx)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    im = ax.imshow(C.numpy(), aspect="auto")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_title(f"corr heatmap (downsampled to {C.shape[0]}x{C.shape[1]})")
    fig.tight_layout()
    fig.savefig(str(out_prefix) + "_heatmap.png", dpi=300)
    plt.close(fig)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(diag_before.numpy(), linewidth=1, alpha=0.7, label="diag before")
    ax.plot(diag_after.numpy(), linewidth=1, alpha=0.9, label="diag after (Hungarian)")
    ax.set_xlabel("unit index i (A order)")
    ax.set_ylabel("corr")
    ax.set_title("Diagonal correlation before/after permutation")
    ax.legend(framealpha=0.6)
    fig.tight_layout()
    fig.savefig(str(out_prefix) + "_diag.png", dpi=300)
    plt.close(fig)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.hist(diag_after.numpy(), bins=50)
    ax.set_xlabel("corr(A_i, B_{perm(i)})")
    ax.set_ylabel("count")
    ax.set_title("Histogram of matched diagonal correlations")
    fig.tight_layout()
    fig.savefig(str(out_prefix) + "_hist.png", dpi=300)
    plt.close(fig)

    return stats


def infer_width_multiplier_from_state(state: Dict[str, torch.Tensor]) -> int:
    # conv1 out_channels = 16 * width_multiplier
    w = int(state["conv1.weight"].shape[0] // 16)
    if w <= 0:
        raise ValueError("Could not infer width_multiplier from conv1.weight")
    return w


def infer_shortcut_option_from_state(state: Dict[str, torch.Tensor]) -> str:
    # train_resnet uses option "A" (no shortcut params) or B/C (have shortcut conv+norm);
    # resnet20_layernorm_permutation_spec also detects from keys :contentReference[oaicite:10]{index=10}
    return "C" if any(k.endswith("shortcut.0.weight") for k in state.keys()) else "A"


_P_INNER = re.compile(r"^P_layer(\d+)_(\d+)_inner$")

def perm_name_to_hook(perm_name: str) -> Tuple[str, Optional[Any], int]:
    """
    Returns: (layer_name_to_hook, preprocess_fn, unit_dim)
    unit_dim=1 means channel axis for NCHW.
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
    raise KeyError(f"Don't know how to map perm name to hook layer: {perm_name}")


def stitch_state_dict_resnet20(
    *,
    state_a: Dict[str, torch.Tensor],
    state_b: Dict[str, torch.Tensor],
    cut: int,
) -> Dict[str, torch.Tensor]:
    """
    cut = k means: take first k 'sections' from A and the rest from B.
    Sections are: stem, 9 blocks, classifier.
    """
    sections: List[List[str]] = [
        ["conv1.", "n1.", "norm1."],  # stem
        ["layer1.0."],
        ["layer1.1."],
        ["layer1.2."],
        ["layer2.0."],
        ["layer2.1."],
        ["layer2.2."],
        ["layer3.0."],
        ["layer3.1."],
        ["layer3.2."],
        ["linear."],  # head
    ]
    if not (0 <= cut <= len(sections)):
        raise ValueError(f"cut must be in [0, {len(sections)}], got {cut}")

    def section_idx(key: str) -> int:
        for i, prefs in enumerate(sections):
            if any(key.startswith(p) for p in prefs):
                return i
        # If something doesn't match (unlikely), treat as "head" from B unless cut==all
        return len(sections) - 1

    out: Dict[str, torch.Tensor] = {}
    if state_a.keys() != state_b.keys():
        raise KeyError("state_a and state_b keysets differ; cannot stitch safely.")
    for k in state_a.keys():
        i = section_idx(k)
        out[k] = state_a[k] if i < cut else state_b[k]
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", type=str, default="CIFAR10", choices=["CIFAR10", "CIFAR100"])
    p.add_argument("--data-root", type=str, default="./data")
    p.add_argument("--runs-root", type=str, default="./runs_resnet20_ln_warmcos")
    p.add_argument("--regime", type=str, default="disjoint", choices=["disjoint", "full"])
    p.add_argument("--base-seed", type=int, default=0)
    p.add_argument("--seed-a", type=int, default=0, help="Only used if regime=full")
    p.add_argument("--seed-b", type=int, default=1, help="Only used if regime=full")

    p.add_argument("--ckpt-a", type=str, default=None)
    p.add_argument("--ckpt-b", type=str, default=None)
    p.add_argument("--which", type=str, default="best", choices=["best", "final"])

    p.add_argument("--match-split", type=str, default="train_eval",
                   choices=["train_eval", "subset_A_eval", "subset_B_eval", "val", "test"])
    p.add_argument("--match-samples", type=int, default=512, help="<=0 means full split.")

    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--eval-samples", type=int, default=0, help="<=0 means full dataset.")

    p.add_argument("--do-interp", action="store_true")
    p.add_argument("--num-lambdas", type=int, default=25)

    p.add_argument("--out-dir", type=str, default=None)
    p.add_argument("--width-multiplier", type=int, default=None)
    p.add_argument("--shortcut-option", type=str, default=None, choices=["A", "B", "C"])

    args = p.parse_args()

    if args.out_dir is None:
        args.out_dir = f"./activation_stitching_out_{args.dataset.lower()}_resnet20_ln"
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = _device_from_utils_fallback()
    _setup_plotting_style()

    # ds_root structure matches train_resnet.py outputs :contentReference[oaicite:11]{index=11}
    ds_root = Path(args.runs_root) / args.dataset / args.regime
    if args.regime == "disjoint":
        run_a_dir = ds_root / f"seed_{args.base_seed}" / "subset_A"
        run_b_dir = ds_root / f"seed_{args.base_seed}" / "subset_B"
    else:
        run_a_dir = ds_root / f"seed_{args.seed_a}"
        run_b_dir = ds_root / f"seed_{args.seed_b}"

    ckpt_a = Path(args.ckpt_a) if args.ckpt_a else _find_single_checkpoint(run_a_dir, args.which)
    ckpt_b = Path(args.ckpt_b) if args.ckpt_b else _find_single_checkpoint(run_b_dir, args.which)

    # indices live at runs_root/<dataset>/indices_...pt :contentReference[oaicite:12]{index=12}
    idx_file = _find_indices_file(Path(args.runs_root) / args.dataset, args.dataset)
    idx_obj = torch.load(idx_file, map_location="cpu")

    train_indices = idx_obj["train_indices"]
    val_indices = idx_obj["val_indices"]
    subset_a_idx = idx_obj["subset_a_indices"]
    subset_b_idx = idx_obj["subset_b_indices"]

    # Build datasets exactly like training does (eval_full has no augmentation) :contentReference[oaicite:13]{index=13}
    train_full, eval_full, test_ds, _targets = train_resnet.load_cifar_datasets(args.dataset, args.data_root)

    train_eval = Subset(eval_full, train_indices)
    subset_a_eval = Subset(eval_full, subset_a_idx)
    subset_b_eval = Subset(eval_full, subset_b_idx)
    val_ds = Subset(eval_full, val_indices)

    def _take_first_n(ds, n: int):
        if n is None or n <= 0 or n >= len(ds):
            return ds
        return Subset(ds, list(range(n)))

    if args.match_split == "train_eval":
        match_ds = train_eval
    elif args.match_split == "subset_A_eval":
        match_ds = subset_a_eval
    elif args.match_split == "subset_B_eval":
        match_ds = subset_b_eval
    elif args.match_split == "val":
        match_ds = val_ds
    else:
        match_ds = test_ds
    match_ds = _take_first_n(match_ds, int(args.match_samples))

    match_loader = DataLoader(
        match_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )

    loaders: Dict[str, DataLoader] = {
        "subset_A_eval": DataLoader(subset_a_eval, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=(device.type == "cuda")),
        "subset_B_eval": DataLoader(subset_b_eval, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=(device.type == "cuda")),
        #"val": DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=(device.type == "cuda")),
        "test": DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=(device.type == "cuda")),
    }

    def max_batches_for(ds_len: int) -> Optional[int]:
        if args.eval_samples is None or args.eval_samples <= 0:
            return None
        return int((args.eval_samples + args.batch_size - 1) // args.batch_size)

    max_batches = {k: max_batches_for(len(dl.dataset)) for k, dl in loaders.items()}

    # Load states
    state_a_full = normalize_state_dict_keys(load_ckpt_state_dict(str(ckpt_a)))
    state_b_full = normalize_state_dict_keys(load_ckpt_state_dict(str(ckpt_b)))

    # infer model hyperparams if not provided
    width_multiplier = int(args.width_multiplier) if args.width_multiplier else infer_width_multiplier_from_state(state_a_full)
    shortcut_option = str(args.shortcut_option) if args.shortcut_option else infer_shortcut_option_from_state(state_a_full)

    num_classes = int(train_resnet.DATASET_STATS[args.dataset]["num_classes"])  # :contentReference[oaicite:14]{index=14}

    # Build models exactly like training :contentReference[oaicite:15]{index=15}
    model_a = architectures.build_model(
        "resnet20",
        num_classes=num_classes,
        norm="flax_ln",
        width_multiplier=width_multiplier,
        shortcut_option=shortcut_option,
    ).to(device)
    model_b = architectures.build_model(
        "resnet20",
        num_classes=num_classes,
        norm="flax_ln",
        width_multiplier=width_multiplier,
        shortcut_option=shortcut_option,
    ).to(device)

    keys = set(model_a.state_dict().keys())
    state_a = {k: v for k, v in state_a_full.items() if k in keys}
    state_b = {k: v for k, v in state_b_full.items() if k in keys}

    model_a.load_state_dict(state_a, strict=True)
    model_b.load_state_dict(state_b, strict=True)

    # Build perm spec (robust to naming + LN param layout + shortcut params) :contentReference[oaicite:16]{index=16}
    ps = resnet20_layernorm_permutation_spec(shortcut_option=shortcut_option, state_dict=state_a)

    perm: Dict[str, torch.Tensor] = {}
    corr_summary: Dict[str, Dict[str, float]] = {}

    # Compute all perms present in spec
    for p_name in sorted(ps.perm_to_axes.keys()):
        layer_name, preprocess, unit_dim = perm_name_to_hook(p_name)
        lp: LayerPermutation = compute_layer_permutation_from_activations(
            model_a=model_a,
            model_b=model_b,
            loader=match_loader,
            layer_name=layer_name,
            device=device,
            max_batches=None,
            unit_dim=unit_dim,
            preprocess=preprocess,
        )
        perm[p_name] = lp.perm_a_to_b
        if lp.corr_matrix is not None:
            corr_summary[p_name] = _plot_corr_diagnostics(lp.corr_matrix, lp.perm_a_to_b, out_dir / f"corr_{p_name}")

    # Save perms
    torch.save({k: v.cpu() for k, v in perm.items()}, out_dir / "permutations.pt")
    with (out_dir / "permutations.json").open("w", encoding="utf-8") as f:
        json.dump({k: v.cpu().tolist() for k, v in perm.items()}, f, indent=2, sort_keys=True)

    # Apply perm to B (function-preserving reparameterization) :contentReference[oaicite:17]{index=17}
    state_b_perm = apply_permutation(ps, perm, state_b)

    a_dev = to_device(state_a, device)
    b_dev = to_device(state_b, device)
    bperm_dev = to_device(state_b_perm, device)

    model_eval = architectures.build_model(
        "resnet20",
        num_classes=num_classes,
        norm="flax_ln",
        width_multiplier=width_multiplier,
        shortcut_option=shortcut_option,
    ).to(device)

    def eval_state(params: Dict[str, torch.Tensor]) -> Dict[str, Dict[str, float]]:
        model_eval.load_state_dict(params, strict=True)
        out = {}
        for split, loader in loaders.items():
            loss, acc = eval_loss_acc(model_eval, loader, device, max_batches=max_batches[split])
            out[split] = {"loss": loss, "accuracy": acc}
        return out

    baselines = {"A": eval_state(a_dev), "B": eval_state(b_dev), "B_perm": eval_state(bperm_dev)}

    # Stitching sections: stem + 9 blocks + head => 11 cuts (0..11)
    num_sections = 11
    cuts = list(range(0, num_sections + 1))
    stitch_metrics = {split: {"acc_naive": [], "acc_perm": [], "loss_naive": [], "loss_perm": []} for split in loaders.keys()}

    for k in cuts:
        stitched_naive = stitch_state_dict_resnet20(state_a=a_dev, state_b=b_dev, cut=k)
        model_eval.load_state_dict(stitched_naive, strict=True)
        for split, loader in loaders.items():
            loss, acc = eval_loss_acc(model_eval, loader, device, max_batches=max_batches[split])
            stitch_metrics[split]["loss_naive"].append(loss)
            stitch_metrics[split]["acc_naive"].append(acc)

        stitched_perm = stitch_state_dict_resnet20(state_a=a_dev, state_b=bperm_dev, cut=k)
        model_eval.load_state_dict(stitched_perm, strict=True)
        for split, loader in loaders.items():
            loss, acc = eval_loss_acc(model_eval, loader, device, max_batches=max_batches[split])
            stitch_metrics[split]["loss_perm"].append(loss)
            stitch_metrics[split]["acc_perm"].append(acc)

    title = (""
        # f"{args.dataset} ResNet20-LN activation stitching, disjoint"
        # f"(width={width_multiplier}, shortcut={shortcut_option}) "
        # f"A={ckpt_a.name} B={ckpt_b.name} match={args.match_split}"
    )
    _plot_stitching(title=title, cuts=cuts, metrics=stitch_metrics, out_dir=out_dir)

    csv_path = out_dir / "stitching_table.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["split", "cut", "acc_naive", "acc_perm", "loss_naive", "loss_perm"])
        for split, d in stitch_metrics.items():
            for i, k in enumerate(cuts):
                w.writerow([split, k, d["acc_naive"][i], d["acc_perm"][i], d["loss_naive"][i], d["loss_perm"][i]])

    interp = None
    if args.do_interp:
        lambdas = torch.linspace(0.0, 1.0, steps=int(args.num_lambdas)).tolist()
        interp_metrics = {split: {"acc_naive": [], "acc_perm": [], "loss_naive": [], "loss_perm": []} for split in loaders.keys()}
        for lam in lambdas:
            sd = interpolate_state_dict(a_dev, b_dev, float(lam))
            model_eval.load_state_dict(sd, strict=True)
            for split, loader in loaders.items():
                loss, acc = eval_loss_acc(model_eval, loader, device, max_batches=max_batches[split])
                interp_metrics[split]["loss_naive"].append(loss)
                interp_metrics[split]["acc_naive"].append(acc)

            sd = interpolate_state_dict(a_dev, bperm_dev, float(lam))
            model_eval.load_state_dict(sd, strict=True)
            for split, loader in loaders.items():
                loss, acc = eval_loss_acc(model_eval, loader, device, max_batches=max_batches[split])
                interp_metrics[split]["loss_perm"].append(loss)
                interp_metrics[split]["acc_perm"].append(acc)

        _plot_interpolation(title=title, lambdas=lambdas, metrics=interp_metrics, out_dir=out_dir)

        interp_csv = out_dir / "interp_table.csv"
        with interp_csv.open("w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["split", "lambda", "acc_naive", "acc_perm", "loss_naive", "loss_perm"])
            for split, d in interp_metrics.items():
                for i, lam in enumerate(lambdas):
                    w.writerow([split, lam, d["acc_naive"][i], d["acc_perm"][i], d["loss_naive"][i], d["loss_perm"][i]])

        interp = {
            "lambdas": lambdas,
            "metrics": interp_metrics,
            "plots": {"acc": str(out_dir / "interp_acc.png"), "loss": str(out_dir / "interp_loss.png")},
            "csv": str(interp_csv),
        }

    results = {
        "dataset": args.dataset,
        "runs_root": args.runs_root,
        "regime": args.regime,
        "ckpt_a": str(ckpt_a),
        "ckpt_b": str(ckpt_b),
        "indices_file": str(idx_file),
        "match_split": args.match_split,
        "match_samples": int(args.match_samples),
        "width_multiplier": width_multiplier,
        "shortcut_option": shortcut_option,
        "permutations_files": {"pt": str(out_dir / "permutations.pt"), "json": str(out_dir / "permutations.json")},
        "corr_diag_summary": corr_summary,
        "baselines": baselines,
        "stitching": {"cuts": cuts, "metrics": stitch_metrics, "csv": str(csv_path)},
        "interpolation": interp,
    }
    _save_json(out_dir / "results.json", results)
    torch.save(results, out_dir / "results.pt")

    summary_lines = [
        f"Title: {title}",
        f"Device: {device}",
        f"Matching split: {args.match_split} (samples={len(match_ds)})",
        "",
        "Baselines (test accuracy):",
        f"  A      : {baselines['A']['test']['accuracy']:.4f}",
        f"  B      : {baselines['B']['test']['accuracy']:.4f}",
        f"  B_perm : {baselines['B_perm']['test']['accuracy']:.4f}",
        "",
        "Stitching (test accuracy vs cut):",
        "  cut k:   " + " ".join(f"{k:>3d}" for k in cuts),
        "  naive :  " + " ".join(f"{x:.3f}" for x in stitch_metrics["test"]["acc_naive"]),
        "  perm  :  " + " ".join(f"{x:.3f}" for x in stitch_metrics["test"]["acc_perm"]),
        "",
        f"Outputs saved to: {out_dir.resolve()}",
    ]
    (out_dir / "SUMMARY.txt").write_text("\n".join(summary_lines), encoding="utf-8")
    print("\n".join(summary_lines))


if __name__ == "__main__":
    main()
