"""
resnet20_interp_plot_local.py

Re-plot interpolation loss/accuracy from local `interp_results.pt` produced by:
  cifar_resnet20_ln_weight_matching_interp.py

Example:
  python resnet20_interp_plot_local.py \
    --results ./weight_matching_out/interp_results.pt \
    --dataset CIFAR10 \
    --arch ResNet20 \
    --width 1 \
    --out-dir ./figs

Aggregate multiple runs (mean curve):
  python resnet20_interp_plot_local.py \
    --results run1/interp_results.pt run2/interp_results.pt run3/interp_results.pt \
    --dataset CIFAR10 \
    --arch ResNet20 \
    --width 1 \
    --aggregate mean \
    --out-dir ./figs
"""

import os
import argparse
from typing import Any, Dict, List, Tuple, Optional

import numpy as np
import torch
import matplotlib.pyplot as plt

# Keep coherence with your other plotting scripts (optional)
# try:
#     import matplotlib_style as _  # noqa: F401
# except Exception:
#     pass


def _to_np(x: Any) -> np.ndarray:
    if isinstance(x, np.ndarray):
        return x
    if torch.is_tensor(x):
        return x.detach().cpu().numpy()
    if isinstance(x, (list, tuple)):
        return np.asarray(x)
    return np.asarray(x)


def load_interp_results(path: str) -> Dict[str, Any]:
    obj = torch.load(path, map_location="cpu")
    if not isinstance(obj, dict):
        raise ValueError(f"{path} did not contain a dict. Got: {type(obj)}")
    return obj


def _require_keys(d: Dict[str, Any], keys: List[str], path: str):
    missing = [k for k in keys if k not in d]
    if missing:
        raise KeyError(
            f"Missing keys in {path}: {missing}\n"
            f"Available keys: {sorted(list(d.keys()))}"
        )


def aggregate_results(dicts: List[Dict[str, Any]], mode: str) -> Dict[str, Any]:
    """
    mode:
      - "none": expects exactly one dict, returns it
      - "mean": averages curves across multiple dicts (requires identical lambdas)
    """
    if mode == "none":
        if len(dicts) != 1:
            raise ValueError("aggregate=none requires exactly one --results file.")
        return dicts[0]

    if mode != "mean":
        raise ValueError(f"Unsupported aggregate mode: {mode}")

    # Required fields (from cifar_resnet20_ln_weight_matching_interp.py)
    req = [
        "lambdas",
        "train_loss_naive", "test_loss_naive",
        "train_loss_perm",  "test_loss_perm",
        "train_acc_naive",  "test_acc_naive",
        "train_acc_perm",   "test_acc_perm",
    ]
    for i, d in enumerate(dicts):
        _require_keys(d, req, f"results[{i}]")

    lambdas0 = _to_np(dicts[0]["lambdas"]).astype(float)
    for i, d in enumerate(dicts[1:], start=1):
        li = _to_np(d["lambdas"]).astype(float)
        if li.shape != lambdas0.shape or not np.allclose(li, lambdas0, atol=1e-12, rtol=1e-8):
            raise ValueError(
                "Cannot aggregate: lambdas grids differ.\n"
                f"- first: shape={lambdas0.shape}\n"
                f"- {i}: shape={li.shape}"
            )

    out: Dict[str, Any] = {"lambdas": lambdas0}

    def stack_mean(key: str) -> np.ndarray:
        arrs = [_to_np(d[key]).astype(float) for d in dicts]
        return np.mean(np.stack(arrs, axis=0), axis=0)

    for k in req:
        if k == "lambdas":
            continue
        out[k] = stack_mean(k)

    return out


def plot_loss(
    lambdas: np.ndarray,
    train_naive: np.ndarray,
    test_naive: np.ndarray,
    train_perm: np.ndarray,
    test_perm: np.ndarray,
    title: str,
    out_png: str,
    out_pdf: Optional[str] = None,
):
    fig = plt.figure()
    ax = fig.add_subplot(111)

    # Naive (grey)
    ax.plot(lambdas, train_naive, color="grey", linewidth=2)
    ax.plot(lambdas, test_naive,  color="grey", linewidth=2, linestyle="dashed")

    # Weight matching / permuted (green)
    ax.plot(lambdas, train_perm, color="tab:green", marker="^", linewidth=2, label="Weight matching")
    ax.plot(lambdas, test_perm,  color="tab:green", marker="^", linewidth=2, linestyle="dashed")

    ax.set_xlabel(r"$\lambda$")
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Model $A$", "Model $B$"])
    ax.set_ylabel("Loss")
    ax.set_title(title)

    fig.tight_layout()
    fig.savefig(out_png, dpi=300)
    if out_pdf is not None:
        fig.savefig(out_pdf)
    plt.close(fig)


def plot_acc(
    lambdas: np.ndarray,
    train_naive: np.ndarray,
    test_naive: np.ndarray,
    train_perm: np.ndarray,
    test_perm: np.ndarray,
    title: str,
    out_png: str,
    out_pdf: Optional[str] = None,
):
    fig = plt.figure()
    ax = fig.add_subplot(111)

    # Naive (grey)
    ax.plot(lambdas, 100.0 * train_naive, color="grey", linewidth=2, label="Train")
    ax.plot(lambdas, 100.0 * test_naive,  color="grey", linewidth=2, linestyle="dashed", label="Test")

    # Weight matching / permuted (green)
    ax.plot(lambdas, 100.0 * train_perm, color="tab:green", marker="^", linewidth=2)
    ax.plot(lambdas, 100.0 * test_perm,  color="tab:green", marker="^", linewidth=2, linestyle="dashed")

    ax.set_xlabel(r"$\lambda$")
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Model $A$", "Model $B$"])
    ax.set_ylabel("Accuracy")
    ax.set_title(title)

    fig.tight_layout()
    fig.savefig(out_png, dpi=300)
    if out_pdf is not None:
        fig.savefig(out_pdf)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--results", type=str, nargs="+", required=True,
                   help="One or more paths to interp_results.pt")
    p.add_argument("--aggregate", type=str, default="none", choices=["none", "mean"],
                   help="If multiple results are provided, you can average the curves.")
    p.add_argument("--dataset", type=str, default="CIFAR10", choices=["CIFAR10", "CIFAR100"])
    p.add_argument("--arch", type=str, default="ResNet20")
    p.add_argument("--width", type=int, default=1, help="Width multiplier (for figure title/filename only).")
    p.add_argument("--tag", type=str, default="", help="Optional extra tag appended to title/filename.")
    p.add_argument("--out-dir", type=str, default="./figs")
    p.add_argument("--prefix", type=str, default="", help="Optional output filename prefix.")
    p.add_argument("--no-pdf", action="store_true", help="If set, do not save PDF copies.")
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    dicts = [load_interp_results(rp) for rp in args.results]
    res = aggregate_results(dicts, args.aggregate)

    req = [
        "lambdas",
        "train_loss_naive", "test_loss_naive",
        "train_loss_perm",  "test_loss_perm",
        "train_acc_naive",  "test_acc_naive",
        "train_acc_perm",   "test_acc_perm",
    ]
    _require_keys(res, req, "aggregated results")

    lambdas = _to_np(res["lambdas"]).astype(float)

    title = f"{args.dataset}, {args.arch} ({args.width}× width)"
    if args.tag.strip():
        title = f"{title} — {args.tag.strip()}"
    if args.aggregate == "mean" and len(args.results) > 1:
        title = f"{title} (mean over {len(args.results)} runs)"

    base = f"{args.dataset.lower()}_{args.arch.lower()}_w{args.width}"
    if args.tag.strip():
        safe_tag = args.tag.strip().replace(" ", "_")
        base = f"{base}_{safe_tag}"
    if args.prefix.strip():
        base = f"{args.prefix.strip()}_{base}"

    loss_png = os.path.join(args.out_dir, f"{base}_loss_interp.png")
    acc_png  = os.path.join(args.out_dir, f"{base}_acc_interp.png")

    loss_pdf = None if args.no_pdf else os.path.join(args.out_dir, f"{base}_loss_interp.pdf")
    acc_pdf  = None if args.no_pdf else os.path.join(args.out_dir, f"{base}_acc_interp.pdf")

    plot_loss(
        lambdas,
        _to_np(res["train_loss_naive"]),
        _to_np(res["test_loss_naive"]),
        _to_np(res["train_loss_perm"]),
        _to_np(res["test_loss_perm"]),
        title=title,
        out_png=loss_png,
        out_pdf=loss_pdf,
    )

    plot_acc(
        lambdas,
        _to_np(res["train_acc_naive"]),
        _to_np(res["test_acc_naive"]),
        _to_np(res["train_acc_perm"]),
        _to_np(res["test_acc_perm"]),
        title=title,
        out_png=acc_png,
        out_pdf=acc_pdf,
    )

    print("Saved:")
    print(" -", loss_png)
    if loss_pdf:
        print(" -", loss_pdf)
    print(" -", acc_png)
    if acc_pdf:
        print(" -", acc_pdf)


if __name__ == "__main__":
    main()
