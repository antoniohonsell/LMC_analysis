#!/usr/bin/env python3
"""
mnist_mlp_activation_stitching.py

Activation-based "model stitching" between two MLPs trained on different MNIST subsets
in runs_mlp/ (train_mlp.py output).

This script avoids your weight_matching implementation entirely; matching is driven by activations.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

_HERE = Path(__file__).resolve().parent
_SRC  = _HERE.parent
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

import architectures
import datasets

from model_stitching.activation_permutation_stitching import (
    LayerPermutation,
    apply_mlp_hidden_permutations_to_state_dict,
    compute_layer_permutation_from_activations,
    infer_fc_layer_numbers_from_state,
    infer_relu_module_names,
    interpolate_state_dict,
    load_ckpt_state_dict,
    normalize_state_dict_keys,
    save_permutations_pickle,
    stitch_state_dict_mlp,
    to_device,
)


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
        raise FileNotFoundError(f"No indices_*.pt file found in {ds_root} (expected output of train_mlp.py).")
    return matches[0]


def _save_json(path: Path, obj) -> None:
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
    ax.set_xlabel("cut_layer k (take fc1..fck from A)")
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
    ax.set_xlabel("cut_layer k (take fc1..fck from A)")
    ax.set_ylabel("cross-entropy loss")
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


def main() -> None:
    p = argparse.ArgumentParser()
    # IMPORTANT: runs_mlp/ uses the dataset string as folder name (e.g. "FASHIONMNIST").
    p.add_argument("--dataset", type=str, default="MNIST",
                   help="MNIST or FASHIONMNIST (case-insensitive).")
    p.add_argument("--data-root", type=str, default="./data")
    p.add_argument("--runs-root", type=str, default="./runs_mlp")
    p.add_argument("--regime", type=str, default="disjoint")
    p.add_argument("--base-seed", type=int, default=0)

    p.add_argument("--ckpt-a", type=str, default=None)
    p.add_argument("--ckpt-b", type=str, default=None)
    p.add_argument("--which", type=str, default="best", choices=["best", "last"])

    p.add_argument("--match-split", type=str, default="train_eval",
                   choices=["train_eval", "subset_A_eval", "subset_B_eval", "test"])
    p.add_argument("--match-samples", type=int, default=10000, help="<=0 means full split.")

    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--eval-samples", type=int, default=0, help="<=0 means full dataset.")

    p.add_argument("--do-interp", action="store_true")
    p.add_argument("--num-lambdas", type=int, default=25)

    p.add_argument("--out-dir", type=str, default=None,
                   help="If not set, defaults to ./activation_stitching_out_<dataset>_mlp")
    args = p.parse_args()

    # Normalize dataset name to match train_mlp.py conventions and folder names.
    args.dataset = str(args.dataset).strip().upper()
    if args.dataset not in datasets.DATASET_STATS:
        raise ValueError(f"Unsupported dataset: {args.dataset} (supported: {sorted(datasets.DATASET_STATS)})")
    if args.out_dir is None:
        args.out_dir = f"./activation_stitching_out_{args.dataset.lower()}_mlp"

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = _device_from_utils_fallback()

    ds_root = Path(args.runs_root) / args.dataset / args.regime
    run_a_dir = ds_root / f"seed_{args.base_seed}" / "subset_A"
    run_b_dir = ds_root / f"seed_{args.base_seed}" / "subset_B"

    ckpt_a = Path(args.ckpt_a) if args.ckpt_a else _find_single_checkpoint(run_a_dir, args.which)
    ckpt_b = Path(args.ckpt_b) if args.ckpt_b else _find_single_checkpoint(run_b_dir, args.which)

    idx_file = _find_indices_file(ds_root, args.dataset)
    idx_obj = torch.load(idx_file, map_location="cpu")
    subset_a_idx = idx_obj["subset_a_indices"]
    subset_b_idx = idx_obj["subset_b_indices"]

    train_full, eval_full, test_ds = datasets.build_datasets(
        args.dataset, root=args.data_root, download=True, augment_train=False, normalize=True
    )

    subset_a_eval = Subset(eval_full, subset_a_idx)
    subset_b_eval = Subset(eval_full, subset_b_idx)

    def _take_first_n(ds, n: int):
        if n is None or n <= 0 or n >= len(ds):
            return ds
        return Subset(ds, list(range(n)))

    if args.match_split == "train_eval":
        match_ds = eval_full
    elif args.match_split == "subset_A_eval":
        match_ds = subset_a_eval
    elif args.match_split == "subset_B_eval":
        match_ds = subset_b_eval
    else:
        match_ds = test_ds
    match_ds = _take_first_n(match_ds, int(args.match_samples))

    match_loader = DataLoader(match_ds, batch_size=args.batch_size, shuffle=False,
                              num_workers=args.num_workers, pin_memory=(device.type == "cuda"))

    loaders: Dict[str, DataLoader] = {
        "subset_A_eval": DataLoader(subset_a_eval, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=(device.type == "cuda")),
        "subset_B_eval": DataLoader(subset_b_eval, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=(device.type == "cuda")),
        "test": DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=(device.type == "cuda")),
    }

    def max_batches_for(ds_len: int) -> Optional[int]:
        if args.eval_samples is None or args.eval_samples <= 0:
            return None
        return int((args.eval_samples + args.batch_size - 1) // args.batch_size)

    max_batches = {k: max_batches_for(len(dl.dataset)) for k, dl in loaders.items()}

    state_a_full = normalize_state_dict_keys(load_ckpt_state_dict(str(ckpt_a)))
    state_b_full = normalize_state_dict_keys(load_ckpt_state_dict(str(ckpt_b)))

    fc_layers = infer_fc_layer_numbers_from_state(state_a_full)
    n_layers = max(fc_layers)
    if n_layers != 4:
        raise ValueError(f"Expected 4 fc layers, found {fc_layers}")

    hidden = int(state_a_full["fc1.weight"].shape[0])
    flat = int(state_a_full["fc1.weight"].shape[1])
    num_classes = int(state_a_full["fc4.weight"].shape[0])
    stats = datasets.DATASET_STATS[args.dataset]
    in_channels = int(stats["in_channels"])
    h, w = tuple(stats["image_size"])
    expected_flat = int(in_channels * h * w)
    if flat != expected_flat:
        raise ValueError(f"Expected {args.dataset} flat dim {expected_flat}, got {flat}")

    input_shape = (in_channels, h, w)
    model_a = architectures.MLP(num_classes=num_classes, input_shape=input_shape, hidden=hidden).to(device)
    model_b = architectures.MLP(num_classes=num_classes, input_shape=input_shape, hidden=hidden).to(device)
    model_a.load_state_dict({k: v for k, v in state_a_full.items() if k in model_a.state_dict()}, strict=True)
    model_b.load_state_dict({k: v for k, v in state_b_full.items() if k in model_b.state_dict()}, strict=True)

    keys = set(model_a.state_dict().keys())
    state_a = {k: v for k, v in state_a_full.items() if k in keys}
    state_b = {k: v for k, v in state_b_full.items() if k in keys}

    relu_names = infer_relu_module_names(model_a, n_fc_layers=n_layers)

    perm_dict: Dict[int, torch.Tensor] = {}
    corr_summary: Dict[str, Dict[str, float]] = {}

    for k, layer_name in enumerate(relu_names, start=1):
        lp: LayerPermutation = compute_layer_permutation_from_activations(
            model_a=model_a,
            model_b=model_b,
            loader=match_loader,
            layer_name=layer_name,
            device=device,
            max_batches=None,
        )
        perm_dict[k] = lp.perm_a_to_b
        if lp.corr_matrix is not None:
            corr_summary[layer_name] = _plot_corr_diagnostics(lp.corr_matrix, lp.perm_a_to_b, out_dir / f"corr_{layer_name}")

    perms_pkl = out_dir / "permutations.pkl"
    perms_pt = out_dir / "permutations.pt"
    save_permutations_pickle(perm_dict, str(perms_pkl))
    torch.save({int(k): v.cpu() for k, v in perm_dict.items()}, perms_pt)

    state_b_perm = apply_mlp_hidden_permutations_to_state_dict(state_b=state_b, perms=perm_dict, n_layers=n_layers)

    a_dev = to_device(state_a, device)
    b_dev = to_device(state_b, device)
    bperm_dev = to_device(state_b_perm, device)

    model_eval = architectures.MLP(num_classes=num_classes, input_shape=input_shape, hidden=hidden).to(device)


    def eval_state(params: Dict[str, torch.Tensor]) -> Dict[str, Dict[str, float]]:
        model_eval.load_state_dict(params, strict=True)
        out = {}
        for split, loader in loaders.items():
            loss, acc = eval_loss_acc(model_eval, loader, device, max_batches=max_batches[split])
            out[split] = {"loss": loss, "accuracy": acc}
        return out

    baselines = {"A": eval_state(a_dev), "B": eval_state(b_dev), "B_perm": eval_state(bperm_dev)}

    cuts = list(range(0, n_layers + 1))
    stitch_metrics = {split: {"acc_naive": [], "acc_perm": [], "loss_naive": [], "loss_perm": []} for split in loaders.keys()}

    for k in cuts:
        stitched_naive = stitch_state_dict_mlp(state_a=a_dev, state_b=b_dev, cut_layer=k, n_layers=n_layers)
        model_eval.load_state_dict(stitched_naive, strict=True)
        for split, loader in loaders.items():
            loss, acc = eval_loss_acc(model_eval, loader, device, max_batches=max_batches[split])
            stitch_metrics[split]["loss_naive"].append(loss)
            stitch_metrics[split]["acc_naive"].append(acc)

        stitched_perm = stitch_state_dict_mlp(state_a=a_dev, state_b=bperm_dev, cut_layer=k, n_layers=n_layers)
        model_eval.load_state_dict(stitched_perm, strict=True)
        for split, loader in loaders.items():
            loss, acc = eval_loss_acc(model_eval, loader, device, max_batches=max_batches[split])
            stitch_metrics[split]["loss_perm"].append(loss)
            stitch_metrics[split]["acc_perm"].append(acc)

    title = f"{args.dataset} activation stitching (seed={args.base_seed}) A={ckpt_a.name} B={ckpt_b.name} match={args.match_split}"
    _plot_stitching(title=title, cuts=cuts, metrics=stitch_metrics, out_dir=out_dir)

    csv_path = out_dir / "stitching_table.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["split", "cut_layer", "acc_naive", "acc_perm", "loss_naive", "loss_perm"])
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
        "base_seed": args.base_seed,
        "ckpt_a": str(ckpt_a),
        "ckpt_b": str(ckpt_b),
        "indices_file": str(idx_file),
        "match_split": args.match_split,
        "match_samples": int(args.match_samples),
        "permutations_files": {"pickle": str(perms_pkl), "pt": str(perms_pt)},
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
        "Stitching (test accuracy vs cut_layer):",
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
