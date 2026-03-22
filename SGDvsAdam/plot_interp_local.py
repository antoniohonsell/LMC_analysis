from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import torch

THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import utils  # type: ignore


def _setup_plotting_style() -> None:
    try:
        utils.apply_stitching_trend_style()  # type: ignore[attr-defined]
    except Exception:
        pass


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
        raise ValueError(f"{path} did not contain a dict. Got {type(obj)}")
    return obj


def _require_keys(d: Dict[str, Any], keys: List[str], path: str) -> None:
    missing = [k for k in keys if k not in d]
    if missing:
        raise KeyError(f"Missing keys in {path}: {missing}")


def aggregate_results(dicts: List[Dict[str, Any]], mode: str) -> Dict[str, Any]:
    if mode == "none":
        if len(dicts) != 1:
            raise ValueError("aggregate=none requires exactly one results file")
        return dicts[0]
    if mode != "mean":
        raise ValueError(f"Unsupported aggregate mode: {mode}")

    req = [
        "lambdas",
        "train_loss_naive", "test_loss_naive",
        "train_loss_perm", "test_loss_perm",
        "train_acc_naive", "test_acc_naive",
        "train_acc_perm", "test_acc_perm",
    ]
    for i, d in enumerate(dicts):
        _require_keys(d, req, f"results[{i}]")

    lambdas0 = _to_np(dicts[0]["lambdas"]).astype(float)
    for i, d in enumerate(dicts[1:], start=1):
        li = _to_np(d["lambdas"]).astype(float)
        if li.shape != lambdas0.shape or not np.allclose(li, lambdas0, atol=1e-12, rtol=1e-8):
            raise ValueError(f"Cannot aggregate because lambdas differ at result {i}")

    out: Dict[str, Any] = {"lambdas": lambdas0}
    for k in req:
        if k == "lambdas":
            continue
        out[k] = np.mean(np.stack([_to_np(d[k]).astype(float) for d in dicts], axis=0), axis=0)
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
) -> None:
    lambdas    = _to_np(lambdas).astype(float)
    train_naive = _to_np(train_naive).astype(float)
    test_naive  = _to_np(test_naive).astype(float)
    train_perm  = _to_np(train_perm).astype(float)
    test_perm   = _to_np(test_perm).astype(float)
    _setup_plotting_style()
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(lambdas, train_naive, color="grey", linewidth=2)
    ax.plot(lambdas, test_naive, color="grey", linewidth=2, linestyle="dashed")
    ax.plot(lambdas, train_perm, linewidth=2, marker="^")
    ax.plot(lambdas, test_perm, linewidth=2, linestyle="dashed", marker="^")
    ax.set_xlabel(r"$\lambda$")
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Model $A$", "Model $B$"])
    ax.set_ylabel("Loss")
    ax.set_title(title)
    ax.legend(["Train, naive", "Test, naive", "Train, permuted", "Test, permuted"], loc="best")
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
) -> None:
    lambdas     = _to_np(lambdas).astype(float)
    train_naive = _to_np(train_naive).astype(float)
    test_naive  = _to_np(test_naive).astype(float)
    train_perm  = _to_np(train_perm).astype(float)
    test_perm   = _to_np(test_perm).astype(float)
    _setup_plotting_style()
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(lambdas, 100.0 * train_naive, color="grey", linewidth=2)
    ax.plot(lambdas, 100.0 * test_naive, color="grey", linewidth=2, linestyle="dashed")
    ax.plot(lambdas, 100.0 * train_perm, linewidth=2, marker="^")
    ax.plot(lambdas, 100.0 * test_perm, linewidth=2, linestyle="dashed", marker="^")
    ax.set_xlabel(r"$\lambda$")
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Model $A$", "Model $B$"])
    ax.set_ylabel("Accuracy (%)")
    ax.set_title(title)
    ax.legend(["Train, naive", "Test, naive", "Train, permuted", "Test, permuted"], loc="best")
    fig.tight_layout()
    fig.savefig(out_png, dpi=300)
    if out_pdf is not None:
        fig.savefig(out_pdf)
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--results", type=str, nargs="+", required=True)
    p.add_argument("--aggregate", type=str, default="none", choices=["none", "mean"])
    p.add_argument("--dataset", type=str, default="")
    p.add_argument("--arch", type=str, default="")
    p.add_argument("--tag", type=str, default="")
    p.add_argument("--width", type=int, default=None)
    p.add_argument("--out-dir", type=str, default="./figs")
    p.add_argument("--prefix", type=str, default="")
    p.add_argument("--no-pdf", action="store_true")
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    dicts = [load_interp_results(x) for x in args.results]
    res = aggregate_results(dicts, args.aggregate)

    req = [
        "lambdas",
        "train_loss_naive", "test_loss_naive",
        "train_loss_perm", "test_loss_perm",
        "train_acc_naive", "test_acc_naive",
        "train_acc_perm", "test_acc_perm",
    ]
    _require_keys(res, req, "aggregated results")

    lambdas = _to_np(res["lambdas"]).astype(float)
    parts: List[str] = []
    if args.dataset:
        parts.append(args.dataset)
    if args.arch:
        parts.append(args.arch)
    if args.width is not None:
        parts.append(f"w={args.width}")
    if args.tag:
        parts.append(args.tag)
    title = " — ".join(parts) if parts else "Interpolation"
    if args.aggregate == "mean" and len(args.results) > 1:
        title = f"{title} (mean over {len(args.results)} runs)"

    stem_parts: List[str] = []
    if args.prefix:
        stem_parts.append(args.prefix)
    if args.dataset:
        stem_parts.append(args.dataset.lower())
    if args.arch:
        stem_parts.append(args.arch.lower())
    if args.width is not None:
        stem_parts.append(f"w{args.width}")
    if args.tag:
        stem_parts.append(args.tag.strip().replace(" ", "_"))
    base = "_".join(stem_parts) if stem_parts else "interp"

    loss_png = os.path.join(args.out_dir, f"{base}_loss_interp.png")
    acc_png = os.path.join(args.out_dir, f"{base}_acc_interp.png")
    loss_pdf = None if args.no_pdf else os.path.join(args.out_dir, f"{base}_loss_interp.pdf")
    acc_pdf = None if args.no_pdf else os.path.join(args.out_dir, f"{base}_acc_interp.pdf")

    plot_loss(
        lambdas,
        _to_np(res["train_loss_naive"]).astype(float),
        _to_np(res["test_loss_naive"]).astype(float),
        _to_np(res["train_loss_perm"]).astype(float),
        _to_np(res["test_loss_perm"]).astype(float),
        title,
        loss_png,
        loss_pdf,
    )
    plot_acc(
        lambdas,
        _to_np(res["train_acc_naive"]).astype(float),
        _to_np(res["test_acc_naive"]).astype(float),
        _to_np(res["train_acc_perm"]).astype(float),
        _to_np(res["test_acc_perm"]).astype(float),
        title,
        acc_png,
        acc_pdf,
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
