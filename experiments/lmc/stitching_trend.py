#!/usr/bin/env python3
"""
stitching_trend.py

Your layout (as in VSCode sidebar):
  activation_out/
    CIFAR10/
      activation_stitching_out_cifar10_resnet20_1/  ... results.json
      activation_stitching_out_cifar10_resnet20_2/  ... results.json
      ...
    CIFAR100/
      activation_stitching_out_cifar100_resnet20_1/ ... results.json
      ...

Reads:
  stitching -> metrics -> test -> loss_perm

Outputs:
1) Table: mean loss_perm for cuts 1..9 for w in {1,2,8,16}
2) (Optional) Plot: for each w, aggregate the cut-values (cuts 1..9),
   compute mean and std across cuts, and save the plot.

Example:
  # run from repo root, pick CIFAR100 under ./activation_out/CIFAR100
  python stitching_trend.py --dataset CIFAR100 --plot

How to use:
python stitching_trend.py --root . \
    --activation-out activation_out \
    --dataset CIFAR100 \
    --plot \
    --out overall_plots/disjoint/stitching_mean_std_CIFAR100.png \
    --verbose
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

TARGET_WS = [1, 2, 8, 16]
TARGET_CUTS = list(range(1, 10))  # cuts 1..9

LOSS_PATH = ("stitching", "metrics", "test", "loss_perm")
CUTS_PATH = ("stitching", "cuts")

# Optional (if you ever store w inside json)
W_KEY_CANDIDATES_DEFAULT = ["w", "window", "width_multiplier"]

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
                    s = s[len(prefix) :]
            return int(s)
    except Exception:
        return None
    return None


def _extract_w_from_path(file_path: Path) -> Optional[int]:
    """
    Infer w from the right-most directory name that ends with _<int> or -<int>.
    Example: activation_stitching_out_cifar100_resnet20_16 -> 16
    """
    for part in reversed(file_path.parts[:-1]):  # exclude filename
        m = SUFFIX_INT_RE.search(part)
        if not m:
            continue
        w = int(m.group(1))
        if w in TARGET_WS:
            return w
    return None


def _extract_w(obj: Dict[str, Any], w_keys: List[str], file_path: Path) -> Optional[int]:
    # 1) from JSON (direct)
    for k in w_keys:
        if k in obj:
            w = _to_int(obj[k])
            if w is not None:
                return w

    # 2) from JSON (inside typical config containers)
    for container in ("params", "param", "config", "cfg", "args", "hparams"):
        if container in obj and isinstance(obj[container], dict):
            for k in w_keys:
                if k in obj[container]:
                    w = _to_int(obj[container][k])
                    if w is not None:
                        return w

    # 3) from folder suffix (your layout)
    return _extract_w_from_path(file_path)


def _iter_results_json_files(root: Path) -> List[Path]:
    if root.is_file():
        return [root]
    return sorted(root.rglob("results.json"))


def _as_float_list(x: Any) -> Optional[List[float]]:
    if isinstance(x, list) and all(isinstance(v, (int, float)) for v in x):
        return [float(v) for v in x]
    return None


def _collect_one_file(
    file_path: Path,
    w_keys: List[str],
    acc: DefaultDict[Tuple[int, int], List[float]],
    verbose: bool = False,
) -> None:
    try:
        obj = json.loads(file_path.read_text(encoding="utf-8"))
    except Exception as e:
        if verbose:
            print(f"[WARN] Failed to read {file_path}: {e}", file=sys.stderr)
        return

    if not isinstance(obj, dict):
        if verbose:
            print(f"[WARN] {file_path} is not a JSON object", file=sys.stderr)
        return

    w = _extract_w(obj, w_keys, file_path)
    if w not in TARGET_WS:
        if verbose:
            print(f"[SKIP] {file_path} (w={w})", file=sys.stderr)
        return

    loss_perm = _get_nested(obj, LOSS_PATH)
    loss_list = _as_float_list(loss_perm)
    if loss_list is None:
        if verbose:
            print(f"[WARN] {file_path} missing stitching.metrics.test.loss_perm", file=sys.stderr)
        return

    cuts = _get_nested(obj, CUTS_PATH)
    if isinstance(cuts, list) and all(isinstance(c, (int, float)) for c in cuts):
        cuts_list = [int(c) for c in cuts]
    else:
        cuts_list = list(range(len(loss_list)))  # fallback

    idx_of = {c: i for i, c in enumerate(cuts_list)}

    for cut in TARGET_CUTS:
        i = idx_of.get(cut, None)
        if i is None or i < 0 or i >= len(loss_list):
            continue
        acc[(w, cut)].append(loss_list[i])


def _print_table(means: Dict[Tuple[int, int], float], counts: Dict[Tuple[int, int], int]) -> None:
    headers = ["cut"] + [f"w={w}" for w in TARGET_WS]
    rows: List[List[str]] = []

    for cut in TARGET_CUTS:
        row = [str(cut)]
        for w in TARGET_WS:
            key = (w, cut)
            row.append(f"{means[key]:.6g}" if key in means else "-")
        rows.append(row)

    widths = [len(h) for h in headers]
    for r in rows:
        for i, cell in enumerate(r):
            widths[i] = max(widths[i], len(cell))

    def fmt_line(cells: List[str]) -> str:
        return " | ".join(cells[i].rjust(widths[i]) for i in range(len(cells)))

    sep = "-+-".join("-" * w for w in widths)

    print(fmt_line(headers))
    print(sep)
    for r in rows:
        print(fmt_line(r))

    print("\nCounts (n results.json contributing to each mean):")
    print(fmt_line(headers))
    print(sep)
    for cut in TARGET_CUTS:
        r = [str(cut)]
        for w in TARGET_WS:
            r.append(str(counts.get((w, cut), 0)))
        print(fmt_line(r))


def _plot_mean_std_across_cuts(
    per_cell_means: Dict[Tuple[int, int], float],
    out_path: Path,
    title: Optional[str] = None,
) -> None:
    import numpy as np
    import matplotlib as mpl
    import matplotlib.pyplot as plt

    # ----- style -----
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
    # ----------------------------

    xs: List[int] = []
    ys: List[float] = []
    yerrs: List[float] = []

    for w in TARGET_WS:
        cut_vals = []
        for cut in TARGET_CUTS:
            key = (w, cut)
            if key in per_cell_means:
                cut_vals.append(per_cell_means[key])

        if not cut_vals:
            continue

        arr = np.array(cut_vals, dtype=float)
        mean = float(arr.mean())
        std = float(arr.std(ddof=1)) if arr.size > 1 else 0.0

        xs.append(w)
        ys.append(mean)
        yerrs.append(std)

    if not xs:
        raise RuntimeError("No data available to plot (check that cuts 1..9 exist).")

    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    ax.errorbar(
        xs,
        ys,
        yerr=yerrs,
        fmt="o-",
        linewidth=3,
        markersize=9,
        capsize=7,
        capthick=2.2,
        label="mean ± std (across cuts 1..9)",
    )

    ax.set_xlabel("Width (w)")
    ax.set_ylabel("Stitching penalty (test loss)")
    ax.set_xscale("log", base=2)
    ax.set_xticks(xs)
    ax.set_xticklabels([str(w) for w in xs])

    if title:
        ax.set_title(title, fontsize=18, fontweight="bold", pad=10)

    ax.legend(loc="best")
    fig.tight_layout()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def _normalize_dataset_token(dataset: str) -> Tuple[str, str]:
    """
    Returns (token, canonical_name)
      token: used for path filtering, e.g. "cifar100"
      canonical_name: e.g. "CIFAR100"
    """
    s = dataset.strip()
    s_low = s.lower().replace("-", "").replace("_", "")
    if s_low in ("cifar10", "c10"):
        return "cifar10", "CIFAR10"
    if s_low in ("cifar100", "c100"):
        return "cifar100", "CIFAR100"
    # generic fallback
    token = re.sub(r"[^a-z0-9]+", "", s.lower())
    return token, s


def _resolve_dataset_root(base: Path, dataset: str, activation_out: str) -> Tuple[Path, str]:
    """
    Prefer the explicit directory (base/activation_out/<DATASET>), but be robust:
    - base might already be activation_out
    - dataset folder could have different casing
    Returns (search_root, token_for_filtering)
    """
    token, canon = _normalize_dataset_token(dataset)

    candidates: List[Path] = []
    # common "repo-root" case
    candidates += [
        base / activation_out / canon,
        base / activation_out / canon.upper(),
        base / activation_out / canon.lower(),
        base / activation_out / dataset,
    ]
    # if base already points at activation_out, or user points base somewhere else
    candidates += [
        base / canon,
        base / canon.upper(),
        base / canon.lower(),
        base / dataset,
    ]

    for p in candidates:
        if p.is_dir():
            return p, token

    # last-resort: check immediate children of base and base/activation_out for a matching folder name
    for parent in [base, base / activation_out]:
        if not parent.is_dir():
            continue
        for child in parent.iterdir():
            if not child.is_dir():
                continue
            child_token = re.sub(r"[^a-z0-9]+", "", child.name.lower())
            if child_token == re.sub(r"[^a-z0-9]+", "", canon.lower()):
                return child, token

    # not found: fall back to base, but still return token so we can filter results.json paths
    return base, token


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--root",
        default=".",
        help="Base directory (typically repo root). With --dataset, the script will look under ./activation_out/<DATASET>/ by default.",
    )
    ap.add_argument(
        "--dataset",
        default=None,
        help='Dataset selector (e.g. "CIFAR100" or "CIFAR10"). If provided, the script focuses on that dataset folder.',
    )
    ap.add_argument(
        "--activation-out",
        default="activation_out",
        help='Name of the activation output directory (default: "activation_out").',
    )
    ap.add_argument(
        "--w-keys",
        default=",".join(W_KEY_CANDIDATES_DEFAULT),
        help='Comma-separated JSON keys to try for w (default: "w,window,width_multiplier").',
    )
    ap.add_argument("--verbose", action="store_true")

    # plotting options
    ap.add_argument("--plot", action="store_true", help="Also save mean±std plot (across cuts 1..9) vs width.")
    ap.add_argument(
        "--out",
        default="stitching_mean_std.png",
        help="Output path for plot (default: stitching_mean_std.png).",
    )
    ap.add_argument("--title", default=None, help="Optional plot title.")

    args = ap.parse_args()

    base = Path(args.root).expanduser().resolve()
    w_keys = [s.strip() for s in args.w_keys.split(",") if s.strip()]

    token = None
    search_root = base
    if args.dataset:
        search_root, token = _resolve_dataset_root(base, args.dataset, args.activation_out)

    files = _iter_results_json_files(search_root)

    # If we couldn't resolve the dataset directory, still try to filter by token (path contains cifar100/cifar10, etc.)
    if args.dataset and token:
        files = [fp for fp in files if token in str(fp).lower()]

    if args.verbose:
        print(f"[INFO] Base:        {base}", file=sys.stderr)
        if args.dataset:
            print(f"[INFO] Dataset:     {args.dataset}", file=sys.stderr)
            print(f"[INFO] Search root: {search_root}", file=sys.stderr)
            if token:
                print(f"[INFO] Path token:  {token}", file=sys.stderr)
        print(f"[INFO] Found {len(files)} results.json files", file=sys.stderr)

    if not files:
        print(f"No results.json found under: {search_root}", file=sys.stderr)
        if args.dataset:
            print(
                f"(and none matched dataset token in paths: {token})",
                file=sys.stderr,
            )
        return 2

    # raw samples per (w, cut)
    acc: DefaultDict[Tuple[int, int], List[float]] = defaultdict(list)
    for fp in files:
        _collect_one_file(fp, w_keys, acc, verbose=args.verbose)

    # mean per (w, cut)
    per_cell_means: Dict[Tuple[int, int], float] = {}
    per_cell_counts: Dict[Tuple[int, int], int] = {}
    for key, vals in acc.items():
        if vals:
            per_cell_means[key] = statistics.fmean(vals)
            per_cell_counts[key] = len(vals)

    if not per_cell_means:
        print(
            "No usable data found.\n"
            "Expected to infer w from folder suffix (e.g. ..._1, ..._2, ..._8, ..._16)\n"
            "and to find stitching.metrics.test.loss_perm in each results.json.",
            file=sys.stderr,
        )
        return 3

    _print_table(per_cell_means, per_cell_counts)

    if args.plot:
        out_path = Path(args.out).expanduser()
        if not out_path.is_absolute():
            out_path = (Path.cwd() / out_path).resolve()
        _plot_mean_std_across_cuts(per_cell_means, out_path, title=args.title)
        print(f"\nSaved plot to: {out_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())