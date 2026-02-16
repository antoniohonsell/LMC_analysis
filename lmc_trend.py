#!/usr/bin/env python3
"""
lmc_trend.py

Make a "stitching_trend-style" plot (mean ± std vs width on log2 x-axis),
but for Linear Mode Connectivity (interpolation) permuted loss.

Inputs supported:
- Weight method:  **interp_results.pt** from linear_mode_connectivity/cifar_resnet20_ln_weight_matching_interp.py
  (expects keys: lambdas, test_loss_perm, width_multiplier) and saved to interp_results.pt.
- Activation method: **results.json** containing:
    interpolation:
      metrics:
        <split>:
          loss_perm: [...]

Output:
- A single plot with two curves (weight vs activation) if both roots are provided.

Example usage:
  python lmc_trend.py \
    --weight-root ./weight_matching_out \
    --activation-root . \
    --out overall_plots/lmc_mean_std_CIFAR100.png \
    --split test \
    --weight-dataset CIFAR100 \
    --weight-regime disjoint


Fianl usage :
python lmc_trend.py \
  --weight-dataset CIFAR100 \
  --weight-root ./weight_matching_out \
  --activation-root . \
  --activation-weight-dataset CIFAR100 \
  --activation-weight-regime disjoint \
  --split test \
  --out overall_plots/lmc_mean_std_CIFAR100.png

If you only have one method, just omit the other root.
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, DefaultDict, Dict, Iterable, List, Optional, Tuple

# Defaults mirror stitching_trend.py
DEFAULT_TARGET_WS = [1, 2, 8, 16]

# For activation-method results.json
ACTIVATION_LOSS_PATH = ("interpolation", "metrics")  # then [split]["loss_perm"]
ACTIVATION_LAMBDAS_PATH = ("interpolation", "lambdas")

# For weight-method interp_results.pt
WEIGHT_KEYS_REQUIRED = ("lambdas", "test_loss_perm")

# match folder suffix "..._16" (or "-16")
SUFFIX_INT_RE = re.compile(r"(?:_|-)(\d+)$")


def _get_nested(d: Any, path: Iterable[str]) -> Any:
    cur = d
    for k in path:
        if not isinstance(cur, dict) or k not in cur:
            return None
        cur = cur[k]
    return cur


def _to_int(x: Any) -> Optional[int]:
    try:
        if isinstance(x, bool):
            return None
        if isinstance(x, int):
            return x
        if isinstance(x, float) and x.is_integer():
            return int(x)
        if isinstance(x, str):
            s = x.strip().lower()
            for prefix in ("w=", "window=", "width_multiplier="):
                if s.startswith(prefix):
                    s = s[len(prefix):]
            return int(s)
    except Exception:
        return None
    return None


def _extract_w_from_path(file_path: Path, target_ws: List[int]) -> Optional[int]:
    """
    Infer w from the right-most directory name that ends with _<int> or -<int>.
    Example: activation_stitching_out_cifar10_resnet20_16 -> 16
    """
    for part in reversed(file_path.parts[:-1]):  # exclude filename
        m = SUFFIX_INT_RE.search(part)
        if not m:
            continue
        w = int(m.group(1))
        if w in target_ws:
            return w
    return None


def _extract_w_from_json(obj: Dict[str, Any], file_path: Path, target_ws: List[int]) -> Optional[int]:
    # Common candidates
    for k in ("width_multiplier", "w", "window"):
        if k in obj:
            w = _to_int(obj[k])
            if w in target_ws:
                return w
    for container in ("params", "param", "config", "cfg", "args", "hparams"):
        if container in obj and isinstance(obj[container], dict):
            for k in ("width_multiplier", "w", "window"):
                if k in obj[container]:
                    w = _to_int(obj[container][k])
                    if w in target_ws:
                        return w
    return _extract_w_from_path(file_path, target_ws)


def _is_endpoint_lambda(lam: float, eps: float = 1e-12) -> bool:
    return abs(lam - 0.0) <= eps or abs(lam - 1.0) <= eps


def _lambda_key(lam: float, ndp: int = 6) -> str:
    # stable bucket key across runs
    return f"{lam:.{ndp}f}"

def _norm_dataset_name(x: str) -> str:
    return str(x).strip().upper()


def _norm_regime_name(x: str) -> str:
    return str(x).strip().lower()


def _resolve_activation_search_root(
    root: Path,
    *,
    weight_dataset: str,
    weight_regime: str,
    verbose: bool,
) -> Path:
    """
    Try to route the search into the folder structure you showed:

        <repo_root>/activation_out/<DATASET>/.../results.json

    Also supports:
      - passing activation_out directly
      - passing the dataset folder directly (e.g. activation_out/CIFAR100)
      - (optional) regime subfolder if you ever add it
    Falls back to the original root if nothing matches.
    """
    if root.is_file():
        return root

    ds_norm = _norm_dataset_name(weight_dataset)
    reg_norm = _norm_regime_name(weight_regime)

    ds_dir_names = {
        str(weight_dataset).strip(),
        str(weight_dataset).strip().upper(),
        str(weight_dataset).strip().lower(),
        str(weight_dataset).strip().capitalize(),
    }
    ds_dir_names = {s for s in ds_dir_names if s}

    candidates: List[Path] = []

    # If user passes repo root known to contain activation_out/<DATASET>/...
    for d in ds_dir_names:
        candidates.append(root / "activation_out" / d / reg_norm)
        candidates.append(root / "activation_out" / d)

    # If user passes activation_out directly
    if root.name == "activation_out":
        for d in ds_dir_names:
            candidates.append(root / d / reg_norm)
            candidates.append(root / d)

    # If user passes dataset folder directly
    if _norm_dataset_name(root.name) == ds_norm:
        candidates.append(root / reg_norm)
        candidates.append(root)

    for c in candidates:
        if c.exists():
            if verbose:
                print(f"[activation] resolved search root: {c}", file=sys.stderr)
            return c

    if verbose:
        print(
            f"[activation][WARN] could not resolve activation_out/<dataset> under {root}; using {root}",
            file=sys.stderr,
        )
    return root


def _iter_results_json_files(root: Path) -> List[Path]:
    if root.is_file():
        return [root]
    return sorted(root.rglob("results.json"))


def _iter_interp_pt_files(root: Path, *, dataset: str = "CIFAR10", regime: str = "disjoint") -> List[Path]:
    if root.is_file():
        return [root]
    pattern = f"resnet20_*/{dataset}/{regime}/interp_results.pt"
    return sorted(root.glob(pattern))


def _collect_activation(
    *,
    root: Path,
    split: str,
    weight_dataset: str = "CIFAR10",
    weight_regime: str = "disjoint",
    include_endpoints: bool,
    target_ws: List[int],
    acc: DefaultDict[Tuple[str, int, str], List[float]],
    verbose: bool,
) -> int:
    search_root = _resolve_activation_search_root(
        root,
        weight_dataset=weight_dataset,
        weight_regime=weight_regime,
        verbose=verbose,
    )
    files = _iter_results_json_files(search_root)
    n_used = 0

    for fp in files:
        try:
            obj = json.loads(fp.read_text(encoding="utf-8"))
        except Exception as e:
            if verbose:
                print(f"[activation][WARN] failed to read {fp}: {e}", file=sys.stderr)
            continue
        if not isinstance(obj, dict):
            continue
        ds = obj.get("dataset", None)
        rg = obj.get("regime", None)

        # If these fields exist in results.json, enforce them.
        if isinstance(ds, str) and _norm_dataset_name(ds) != _norm_dataset_name(weight_dataset):
            continue
        if isinstance(rg, str) and _norm_regime_name(rg) != _norm_regime_name(weight_regime):
            continue

        # Must have interpolation
        metrics = _get_nested(obj, ACTIVATION_LOSS_PATH)
        lambdas = _get_nested(obj, ACTIVATION_LAMBDAS_PATH)
        if not isinstance(metrics, dict) or not isinstance(lambdas, list):
            continue
        if split not in metrics or not isinstance(metrics[split], dict):
            continue
        loss_perm = metrics[split].get("loss_perm", None)
        if not (isinstance(loss_perm, list) and all(isinstance(v, (int, float)) for v in loss_perm)):
            continue
        if not (isinstance(lambdas, list) and len(lambdas) == len(loss_perm)):
            continue

        w = _extract_w_from_json(obj, fp, target_ws)
        if w not in target_ws:
            continue

        for lam, loss in zip(lambdas, loss_perm):
            if not isinstance(lam, (int, float)):
                continue
            lamf = float(lam)
            if (not include_endpoints) and _is_endpoint_lambda(lamf):
                continue
            acc[("activation", int(w), _lambda_key(lamf))].append(float(loss))

        n_used += 1

    return n_used


def _collect_weight(
    *,
    root: Path,
    include_endpoints: bool,
    target_ws: List[int],
    acc: DefaultDict[Tuple[str, int, str], List[float]],
    verbose: bool,
    weight_dataset: str = "CIFAR10",
    weight_regime: str = "disjoint",
) -> int:
    try:
        import torch
    except Exception as e:
        raise RuntimeError("Weight-method collection requires torch to load interp_results.pt") from e

    files = _iter_interp_pt_files(root, dataset=weight_dataset, regime=weight_regime)
    n_used = 0

    for fp in files:
        try:
            d = torch.load(fp, map_location="cpu")
        except Exception as e:
            if verbose:
                print(f"[weight][WARN] failed to torch.load {fp}: {e}", file=sys.stderr)
            continue
        if not isinstance(d, dict):
            continue

        if not all(k in d for k in WEIGHT_KEYS_REQUIRED):
            continue

        lambdas = d.get("lambdas", None)
        loss_perm = d.get("test_loss_perm", None)  # weight script stores test_loss_perm :contentReference[oaicite:4]{index=4}
        w = d.get("width_multiplier", None)

        if not (isinstance(w, (int, float)) and int(w) in target_ws):
            # fallback to folder suffix if needed
            w2 = _extract_w_from_path(fp, target_ws)
            if w2 is None:
                continue
            w = w2

        if not (isinstance(lambdas, list) and isinstance(loss_perm, list) and len(lambdas) == len(loss_perm)):
            continue
        if not all(isinstance(v, (int, float)) for v in lambdas):
            continue
        if not all(isinstance(v, (int, float)) for v in loss_perm):
            continue

        for lam, loss in zip(lambdas, loss_perm):
            lamf = float(lam)
            if (not include_endpoints) and _is_endpoint_lambda(lamf):
                continue
            acc[("weight", int(w), _lambda_key(lamf))].append(float(loss))

        n_used += 1

    return n_used


def _reduce_over_lambdas(
    *,
    acc: DefaultDict[Tuple[str, int, str], List[float]],
    method: str,
    target_ws: List[int],
) -> Tuple[List[int], List[float], List[float]]:
    """
    For each width w:
      1) average replicates per lambda bucket
      2) compute mean ± std across lambdas
    """
    import numpy as np

    xs: List[int] = []
    ys: List[float] = []
    yerrs: List[float] = []

    # mean per (method, w, lam_key)
    per_cell_means: Dict[Tuple[int, str], float] = {}
    for (m, w, lk), vals in acc.items():
        if m != method or not vals:
            continue
        per_cell_means[(w, lk)] = statistics.fmean(vals)

    for w in target_ws:
        lam_vals = [v for (ww, _lk), v in per_cell_means.items() if ww == w]
        if not lam_vals:
            continue
        arr = np.asarray(lam_vals, dtype=float)
        mean = float(arr.mean())
        std = float(arr.std(ddof=1)) if arr.size > 1 else 0.0

        xs.append(w)
        ys.append(mean)
        yerrs.append(std)

    return xs, ys, yerrs


def _plot_two_methods(
    *,
    out_path: Path,
    title: Optional[str],
    target_ws: List[int],
    weight_series: Optional[Tuple[List[int], List[float], List[float]]],
    activation_series: Optional[Tuple[List[int], List[float], List[float]]],
    ylabel: str,
) -> None:
    import matplotlib as mpl
    import matplotlib.pyplot as plt

    # Copy the styling philosophy from stitching_trend.py
    try:
        plt.style.use("seaborn-v0_8-whitegrid")
    except OSError:
        plt.style.use("seaborn-whitegrid")

    deep = ["#4c72b0", "#dd8452", "#55a868", "#c44e52", "#8172b3", "#937860"]
    mpl.rcParams["axes.prop_cycle"] = mpl.cycler(color=deep)

    mpl.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "#f6efe7",
            "grid.color": "#a7a29f",
            "grid.linewidth": 1.2,
            "grid.alpha": 1.0,
            "axes.edgecolor": "black",
            "axes.linewidth": 2.6,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.labelweight": "bold",
            "axes.labelsize": 22,
            "xtick.labelsize": 14,
            "ytick.labelsize": 14,
            "legend.framealpha": 1.0,
            "legend.facecolor": "white",
            "legend.edgecolor": "#808080",
            "legend.fontsize": 14,
        }
    )

    fig, ax = plt.subplots(figsize=(8.5, 5.5))

    any_plotted = False

    if weight_series is not None:
        xs, ys, yerrs = weight_series
        if xs:
            ax.errorbar(
                xs, ys, yerr=yerrs,
                fmt="o-", linewidth=3, markersize=9,
                capsize=7, capthick=2.2,
                label="Weight matching (mean ± std over λ)",
            )
            any_plotted = True

    if activation_series is not None:
        xs, ys, yerrs = activation_series
        if xs:
            ax.errorbar(
                xs, ys, yerr=yerrs,
                fmt="s-", linewidth=3, markersize=8,
                capsize=7, capthick=2.2,
                label="Activation stitching (mean ± std over λ)",
            )
            any_plotted = True

    if not any_plotted:
        raise RuntimeError("No data available to plot. Check your roots/split and that interpolation results exist.")

    ax.set_xlabel("Width (w)")
    ax.set_ylabel(ylabel)
    ax.set_xscale("log", base=2)

    # Keep ticks aligned to widths we care about
    ax.set_xticks(target_ws)
    ax.set_xticklabels([str(w) for w in target_ws])

    if title:
        ax.set_title(title, fontsize=18, fontweight="bold", pad=10)

    ax.legend(loc="best")
    fig.tight_layout()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--weight-root", type=str, default=None,
                    help="Root containing weight-method interp_results.pt files (rglob).")
    ap.add_argument("--activation-root", type=str, default=None,
                    help="Root containing activation-method results.json files (rglob).")
    ap.add_argument("--split", type=str, default="test",
                    help="Split name for activation interpolation metrics (default: test).")
    ap.add_argument("--include-endpoints", action="store_true",
                    help="If set, include λ=0 and λ=1 in the mean/std over λ.")
    ap.add_argument("--ws", type=str, default="1,2,8,16",
                    help="Comma-separated widths to consider (default: 1,2,8,16).")
    ap.add_argument("--out", type=str, default="lmc_mean_std.png",
                    help="Output plot path (default: lmc_mean_std.png).")
    ap.add_argument("--title", type=str, default=None, help="Optional plot title.")
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--weight-dataset", type=str, default="CIFAR10",
                help="Dataset subdir for weight-matching results (default: CIFAR10).")
    ap.add_argument("--weight-regime", type=str, default="disjoint",
                help="Regime subdir for weight-matching results (default: disjoint).")
    ap.add_argument("--activation-weight-dataset", type=str, default="CIFAR10",
                    help="Dataset to select activation_out/<dataset>/... and filter results.json (default: CIFAR10).")
    ap.add_argument("--activation-weight-regime", type=str, default="disjoint",
                    help="Regime to filter activation results.json (default: disjoint).")
    args = ap.parse_args()

    target_ws = [int(s.strip()) for s in args.ws.split(",") if s.strip()]

    acc: DefaultDict[Tuple[str, int, str], List[float]] = defaultdict(list)

    n_w = 0
    n_a = 0

    if args.weight_root:
        n_w = _collect_weight(
        root=Path(args.weight_root).expanduser().resolve(),
        include_endpoints=bool(args.include_endpoints),
        target_ws=target_ws,
        acc=acc,
        verbose=bool(args.verbose),
        weight_dataset=str(args.weight_dataset),
        weight_regime=str(args.weight_regime),
    )

    if args.activation_root:
        n_a = _collect_activation(
            root=Path(args.activation_root).expanduser().resolve(),
            split=str(args.split),
            weight_dataset=str(args.activation_weight_dataset),
            weight_regime=str(args.activation_weight_regime),
            include_endpoints=bool(args.include_endpoints),
            target_ws=target_ws,
            acc=acc,
            verbose=bool(args.verbose),
        )

    if (not args.weight_root) and (not args.activation_root):
        print("Provide at least one of --weight-root or --activation-root.", file=sys.stderr)
        return 2

    weight_series = None
    activation_series = None

    if args.weight_root:
        weight_series = _reduce_over_lambdas(acc=acc, method="weight", target_ws=target_ws)
    if args.activation_root:
        activation_series = _reduce_over_lambdas(acc=acc, method="activation", target_ws=target_ws)

    out_path = Path(args.out).expanduser()
    if not out_path.is_absolute():
        out_path = (Path.cwd() / out_path).resolve()

    ylabel = "Interpolation loss"

    _plot_two_methods(
        out_path=out_path,
        title=args.title,
        target_ws=target_ws,
        weight_series=weight_series,
        activation_series=activation_series,
        ylabel=ylabel,
    )

    print(f"Used files: weight={n_w}, activation={n_a}")
    print(f"Saved plot to: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())