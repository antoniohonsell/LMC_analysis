# plot_llfc_avg_by_width.py
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_SRC  = _HERE.parent.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import numpy as np
import torch
import matplotlib.pyplot as plt
from utils import apply_stitching_trend_style


RUN_DIR_RE = re.compile(r"llfc_resnet20_ln_cifar100_w(\d+)$")


def pick_llfc_file(run_dir: Path) -> Path:
    # compute_llfc.py saves a file named llfc_cos_{tag}.pt into the run folder :contentReference[oaicite:1]{index=1}
    cands = sorted(run_dir.glob("llfc_cos_*.pt"))
    if not cands:
        raise FileNotFoundError(f"No llfc_cos_*.pt found in: {run_dir}")
    # if multiple exist, pick the most recently modified
    return max(cands, key=lambda p: p.stat().st_mtime)


def load_curve(pt_path: Path) -> tuple[np.ndarray, np.ndarray]:
    data = torch.load(pt_path, map_location="cpu")
    lambdas = np.asarray(data["lambdas"], dtype=float)
    curve = np.asarray(data["cos_mean_layeravg"], dtype=float)  # layer-averaged LLFC vs lambda :contentReference[oaicite:2]{index=2}
    if lambdas.shape != curve.shape:
        raise ValueError(f"Shape mismatch in {pt_path}: lambdas {lambdas.shape} vs curve {curve.shape}")
    return lambdas, curve


def main() -> None:
    apply_stitching_trend_style()
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs_root", type=str, default="runs", help="Path to the runs/ folder")
    ap.add_argument("--widths", type=int, nargs="*", default=[1, 2, 8, 16, 32],
                    help="Width multipliers to include (default: 1 2 8 16 32)")
    ap.add_argument("--out_png", type=str, default="LLFC_results/llfc_avg_by_width.png")
    ap.add_argument("--plot_curves", action="store_true", help="Also save LLFC-vs-lambda curves per width")
    ap.add_argument("--out_curves_png", type=str, default="LLFC_results/llfc_curves_by_width.png")
    args = ap.parse_args()

    runs_root = Path(args.runs_root)
    if not runs_root.exists():
        raise FileNotFoundError(f"runs_root does not exist: {runs_root}")

    widths = list(args.widths)

    avg_by_w = {}
    curve_by_w = {}
    std_by_w = {}

    for w in widths:
        run_dir = runs_root / f"llfc_resnet20_ln_cifar100_w{w}"
        if not run_dir.exists():
            print(f"[skip] missing folder: {run_dir}")
            continue

        pt_path = pick_llfc_file(run_dir)
        lambdas, curve = load_curve(pt_path)

        avg_llfc = float(curve.mean())               # average over lambdas -> single scalar per width
        auc_llfc = float(np.trapz(curve, lambdas))   # optional: area under curve (normalized by lambda-range if you want)
        avg_llfc = float(curve.mean())
        std_llfc = float(curve.std(ddof=1))  # std over lambdas (sample std)

        avg_by_w[w] = avg_llfc
        std_by_w[w] = std_llfc

        avg_by_w[w] = avg_llfc
        curve_by_w[w] = (lambdas, curve)

        print(f"w={w:>2}  file={pt_path.name}  mean={avg_llfc:.6f}  auc={auc_llfc:.6f}")

    if not avg_by_w:
        raise RuntimeError("No widths were found / loaded. Check folder names and runs_root.")

    # --- Plot: width vs average LLFC ---
    xs = np.array(sorted(avg_by_w.keys()), dtype=float)
    ys = np.array([avg_by_w[int(x)] for x in xs], dtype=float)
    yerr = np.array([std_by_w[int(x)] for x in xs], dtype=float)

    plt.figure(figsize=(8, 6))
    # Plot line with markers
    plt.plot(xs, ys, "o-", linewidth=2.5, markersize=8, label="Mean LLFC", color="C0")
    # Add shaded area for ±1 std (variance visualization)
    plt.fill_between(xs, ys - yerr, ys + yerr, alpha=0.3, color="C0", label="±1 std")
    plt.xlabel("width multiplier", fontsize=12)
    plt.ylabel("average LLFC (mean over λ of cos_mean_layeravg)", fontsize=12)
    plt.title("Average LLFC by width", fontsize=13, fontweight="bold")
    plt.xscale("log")  # Add log scale to x-axis
    plt.xticks(xs, xs.astype(int), fontsize=11)  # Show actual width values as labels
    plt.ylim(0, None)  # Start y-axis from 0
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(args.out_png, dpi=200)
    plt.close()
    print(f"Saved: {args.out_png}")

    # --- Optional: plot curves LLFC(λ) for each width ---
    if args.plot_curves:
        plt.figure(figsize=(9, 6))
        for w in sorted(curve_by_w.keys()):
            lambdas, curve = curve_by_w[w]
            plt.plot(lambdas, curve, label=f"w={w}", linewidth=2, marker="o", markersize=4, alpha=0.8)
        plt.xlabel(r"$\lambda$", fontsize=12)
        plt.ylabel("LLFC (layer-avg cosine similarity)", fontsize=12)
        plt.title("LLFC vs λ (layer-averaged) for each width", fontsize=13, fontweight="bold")
        plt.legend(fontsize=11)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(args.out_curves_png, dpi=200)
        plt.close()
        print(f"Saved: {args.out_curves_png}")


if __name__ == "__main__":
    main()