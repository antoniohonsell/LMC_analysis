#!/usr/bin/env python3
# scripts/plot_hz_metrics_sweep.py
from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


"""
Example of usage:
export PYTHONPATH="$(pwd)"

python scripts/plot_hz_metrics_sweep.py \
  --summary_csv runs_sweep_full/mnist_mlp_reg/hz_metrics_summary.csv
"""


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def plot_metric_vs_lr(df: pd.DataFrame, metric: str, outdir: Path) -> None:
    fig = plt.figure()
    ax = fig.add_subplot(111)

    for wd, g in df.groupby("weight_decay"):
        g2 = g.sort_values("lr")
        ax.plot(g2["lr"].values, g2[metric].values, marker="o", linestyle="-", label=f"wd={wd:g}")

    ax.set_xscale("log")
    ax.set_xlabel("learning rate")
    ax.set_ylabel(metric)
    ax.set_title(f"{metric} vs lr (one line per weight_decay)")
    ax.legend(framealpha=0.5)
    fig.tight_layout()
    fig.savefig(outdir / f"{metric}_vs_lr.png", dpi=250)
    plt.close(fig)


def plot_metric_vs_wd(df: pd.DataFrame, metric: str, outdir: Path) -> None:
    fig = plt.figure()
    ax = fig.add_subplot(111)

    for lr, g in df.groupby("lr"):
        g2 = g.sort_values("weight_decay")
        ax.plot(g2["weight_decay"].values, g2[metric].values, marker="o", linestyle="-", label=f"lr={lr:g}")

    ax.set_xscale("log")
    ax.set_xlabel("weight_decay")
    ax.set_ylabel(metric)
    ax.set_title(f"{metric} vs weight_decay (one line per lr)")
    ax.legend(framealpha=0.5)
    fig.tight_layout()
    fig.savefig(outdir / f"{metric}_vs_wd.png", dpi=250)
    plt.close(fig)


def plot_heatmap(df: pd.DataFrame, metric: str, outdir: Path) -> None:
    """
    Heatmap only if grid is rectangular enough; otherwise this will still plot but NaNs appear.
    Rows: weight_decay, Cols: lr
    """
    piv = df.pivot_table(index="weight_decay", columns="lr", values=metric, aggfunc="mean")
    wds = piv.index.to_numpy()
    lrs = piv.columns.to_numpy()
    vals = piv.to_numpy()

    fig = plt.figure()
    ax = fig.add_subplot(111)
    im = ax.imshow(vals, aspect="auto", interpolation="nearest")

    ax.set_title(f"{metric} heatmap (wd x lr)")
    ax.set_ylabel("weight_decay (sorted)")
    ax.set_xlabel("lr (sorted)")

    ax.set_yticks(np.arange(len(wds)))
    ax.set_yticklabels([f"{x:g}" for x in wds])

    ax.set_xticks(np.arange(len(lrs)))
    ax.set_xticklabels([f"{x:g}" for x in lrs], rotation=45, ha="right")

    fig.colorbar(im, ax=ax, shrink=0.9)
    fig.tight_layout()
    fig.savefig(outdir / f"{metric}_heatmap.png", dpi=250)
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--summary_csv", type=str, required=True, help="Path to hz_metrics_summary.csv")
    p.add_argument("--outdir", type=str, default=None, help="Where to save plots (default: alongside CSV)")
    p.add_argument("--print_top", type=int, default=20, help="Print top-N rows sorted by pow_eps")
    args = p.parse_args()

    csv_path = Path(args.summary_csv).expanduser().resolve()
    df = pd.read_csv(csv_path)

    if args.outdir is None:
        outdir = csv_path.parent / "hz_plots"
    else:
        outdir = Path(args.outdir).expanduser().resolve()
    _ensure_dir(outdir)

    # Basic cleanup
    df = df.dropna(subset=["lr", "weight_decay"]).copy()
    df["lr"] = df["lr"].astype(float)
    df["weight_decay"] = df["weight_decay"].astype(float)

    # Print
    print("\n=== Summary (sorted by pow_eps ascending) ===")
    print(df.sort_values("pow_eps").head(int(args.print_top)).to_string(index=False))

    metrics: List[str] = [
        "eps_lin",
        "rho_fro",
        "eps_comm",
        "pa_mean_cos",
        "pa_max_angle_deg",
        "powerlaw_alpha",
        "powerlaw_r2",
        "pow_eps",
        "pow_alpha",
    ]

    for m in metrics:
        plot_metric_vs_lr(df, m, outdir)
        plot_metric_vs_wd(df, m, outdir)
        plot_heatmap(df, m, outdir)

    print(f"\nSaved plots to: {outdir}")


if __name__ == "__main__":
    main()