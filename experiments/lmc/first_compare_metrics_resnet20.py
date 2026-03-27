# compare_cka_resnet20.py
#
# Plot per-layer alignment for ResNet20 across:
#   - CKA
#   - mutual-kNN for topk in {100, 500, 1000}
# And across scenarios:
#   - CIFAR10: (seed0 vs seed1) and (subsetA vs subsetB)
#   - CIFAR100: (seed0 vs seed1) and (subsetA vs subsetB)
#
# USAGE:
#   python compare_cka_resnet20.py \
#     --dir ./results/alignment_resnet \
#     --save ./results/alignment_resnet/alignment_resnet20_grid.png
#
# Notes:
# - It expects .npz files produced by measure_alignment_platonic.py containing:
#     scores, layers, ckpt_a, ckpt_b, split, metric, topk (for knn), pairing (optional)
# - If "pairing" is missing from the .npz, it is inferred from the filename.

import argparse
import glob
import os
import re
import sys
from collections import defaultdict
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_SRC  = _HERE.parent.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import numpy as np
import matplotlib.pyplot as plt


def _as_py(x):
    """Convert np scalar/bytes -> python type."""
    if isinstance(x, np.ndarray) and x.shape == ():
        x = x.item()
    if isinstance(x, (bytes, bytearray)):
        x = x.decode("utf-8")
    return x


def parse_filename(npz_path: str) -> dict:
    """
    Expected filename format (from measure_alignment_platonic.py):
      base_a__vs__base_b__{split}__{metric_tag}__{pairing}.npz

    metric_tag may be e.g.:
      cka
      mutual_knn_topk100
    """
    name = os.path.basename(npz_path).replace(".npz", "")
    if "__vs__" not in name:
        return {}

    left, rest = name.split("__vs__", 1)
    parts = rest.split("__")
    if len(parts) < 4:
        return {}

    base_b = parts[0]
    split = parts[1]
    metric_tag = parts[2]
    pairing = parts[3]

    topk = None
    m = re.search(r"topk(\d+)", metric_tag)
    if m:
        topk = int(m.group(1))

    metric = re.sub(r"_topk\d+$", "", metric_tag)
    return {
        "base_a": left,
        "base_b": base_b,
        "split": split,
        "pairing": pairing,
        "metric_tag": metric_tag,
        "metric": metric,
        "topk": topk,
    }


def load_result(npz_path: str) -> dict:
    d = np.load(npz_path, allow_pickle=True)

    scores = np.array(d["scores"], dtype=float)
    layers = np.array(d["layers"]).astype(str).tolist()

    # meta from file (preferred)
    meta = {
        "ckpt_a": str(_as_py(d.get("ckpt_a", ""))),
        "ckpt_b": str(_as_py(d.get("ckpt_b", ""))),
        "split": str(_as_py(d.get("split", "unknown"))),
        "metric": str(_as_py(d.get("metric", "unknown"))),
        "pairing": str(_as_py(d.get("pairing", "unknown"))),
        "topk": _as_py(d.get("topk", None)),
    }

    # fill missing/infer from filename
    pf = parse_filename(npz_path)
    if meta["split"] in ("unknown", "", None) and "split" in pf:
        meta["split"] = pf["split"]
    if meta["pairing"] in ("unknown", "", None) and "pairing" in pf:
        meta["pairing"] = pf["pairing"]

    # metric: keep file metric if valid, else infer from filename
    if meta["metric"] in ("unknown", "", None) and "metric" in pf:
        meta["metric"] = pf["metric"]

    # topk: prefer file; else infer from filename
    if meta["topk"] is None and "topk" in pf:
        meta["topk"] = pf["topk"]

    # normalize topk to int/None
    try:
        if meta["topk"] is not None:
            meta["topk"] = int(meta["topk"])
    except Exception:
        meta["topk"] = None

    return {"path": npz_path, "scores": scores, "layers": layers, "meta": meta}


def align_scores_to_layers(ref_layers, layers, scores):
    """If layer lists differ, align by layer name."""
    if layers == ref_layers:
        return scores
    m = {ln: sc for ln, sc in zip(layers, scores)}
    out = []
    for ln in ref_layers:
        if ln not in m:
            raise ValueError(f"Layer '{ln}' missing in one file. Have: {layers}")
        out.append(m[ln])
    return np.array(out, dtype=float)


def infer_dataset_and_mode(meta: dict) -> tuple[str, str]:
    """
    Determine dataset (CIFAR10/100) and mode (non-disjoint/disjoint) from ckpt paths.
    """
    text = " ".join([meta.get("ckpt_a", ""), meta.get("ckpt_b", "")])

    if "CIFAR100" in text:
        dataset = "CIFAR100"
    elif "CIFAR10" in text:
        dataset = "CIFAR10"
    else:
        dataset = "UNKNOWN"

    # disjoint checkpoints are named ...subsetA... / ...subsetB...
    disjoint = any(tok in text for tok in ["subsetA", "subsetB", "subset_A", "subset_B"])
    mode = "disjoint" if disjoint else "non-disjoint"
    return dataset, mode


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", type=str, default="./results/alignment_resnet",
                    help="Folder containing saved .npz alignment results")
    ap.add_argument("--split", type=str, default="val", choices=["train", "val", "test"],
                    help="Which data split to plot")
    ap.add_argument("--pairing", type=str, default="diagonal",
                    help="Pairing to plot (typically 'diagonal')")
    ap.add_argument("--topk", nargs="+", type=int, default=[100, 500, 1000],
                    help="topk values to plot for mutual_knn")
    ap.add_argument("--save", type=str, default=None,
                    help="If set, save the figure to this path")
    ap.add_argument("--no_show", action="store_true",
                    help="If set, do not call plt.show()")
    args = ap.parse_args()

    # Discover candidate results
    pat = os.path.join(args.dir, "resnet20_*__vs__*__.npz")
    paths = sorted(glob.glob(pat))
    if len(paths) == 0:
        # fallback: any npz in dir
        paths = sorted(glob.glob(os.path.join(args.dir, "*.npz")))

    if len(paths) == 0:
        raise SystemExit(f"No .npz files found in: {args.dir}")

    results = [load_result(p) for p in paths]

    # Filter to requested split/pairing + desired metrics
    filtered = []
    for r in results:
        split = r["meta"]["split"]
        pairing = r["meta"]["pairing"]
        metric = r["meta"]["metric"]

        if split != args.split:
            continue
        if pairing not in ("unknown", "", None) and pairing != args.pairing:
            continue

        if metric == "cka" or metric == "unbiased_cka" or ("cka" in metric):
            filtered.append(r)
        elif metric == "mutual_knn":
            filtered.append(r)

    if len(filtered) == 0:
        raise SystemExit(
            f"No matching results for split={args.split} and pairing={args.pairing} in {args.dir}"
        )

    # Group by scenario -> metric -> topk(None for cka) -> result
    grouped = defaultdict(lambda: defaultdict(dict))
    scenario_layers = {}

    for r in filtered:
        dataset, mode = infer_dataset_and_mode(r["meta"])
        scenario = (dataset, mode)

        metric = r["meta"]["metric"]
        topk = r["meta"]["topk"] if metric == "mutual_knn" else None

        # Keep only requested topk for mutual_knn
        if metric == "mutual_knn" and topk not in set(args.topk):
            continue

        # Establish ref layer order per scenario
        if scenario not in scenario_layers:
            scenario_layers[scenario] = r["layers"]

        ref_layers = scenario_layers[scenario]
        r_scores = align_scores_to_layers(ref_layers, r["layers"], r["scores"])

        grouped[scenario][metric][topk] = {
            "scores": r_scores,
            "layers": ref_layers,
            "path": r["path"],
        }

    # Plot grid: rows = dataset (CIFAR10, CIFAR100), cols = mode (non-disjoint, disjoint)
    grid_order = [
        ("CIFAR10", "non-disjoint"),
        ("CIFAR10", "disjoint"),
        ("CIFAR100", "non-disjoint"),
        ("CIFAR100", "disjoint"),
    ]

    titles = {
        ("CIFAR10", "non-disjoint"): "CIFAR-10 | same dataset, different seeds",
        ("CIFAR10", "disjoint"):     "CIFAR-10 | disjoint subsets A vs B",
        ("CIFAR100", "non-disjoint"): "CIFAR-100 | same dataset, different seeds",
        ("CIFAR100", "disjoint"):     "CIFAR-100 | disjoint subsets A vs B",
    }

    fig, axes = plt.subplots(2, 2, figsize=(12, 7), sharey=True)
    fig.suptitle(f"ResNet20 alignment (split={args.split}, pairing={args.pairing})")

    for idx, scenario in enumerate(grid_order):
        ax = axes[idx // 2][idx % 2]
        ax.set_title(titles.get(scenario, f"{scenario[0]} | {scenario[1]}"))

        if scenario not in scenario_layers:
            ax.text(0.5, 0.5, "No data found", ha="center", va="center", transform=ax.transAxes)
            ax.set_axis_off()
            continue

        layers = scenario_layers[scenario]
        x = np.arange(len(layers))

        # Lines: CKA + mutual_knn@topk...
        # Pick a single "cka-like" metric if present (prefer exact 'cka', else any containing 'cka')
        cka_key = None
        if "cka" in grouped[scenario]:
            cka_key = "cka"
        else:
            for k in grouped[scenario].keys():
                if "cka" in k:
                    cka_key = k
                    break

        if cka_key is not None and None in grouped[scenario][cka_key]:
            y = grouped[scenario][cka_key][None]["scores"]
            ax.plot(x, y, marker="o", linewidth=2, label=f"{cka_key} (mean={y.mean():.3f})")
        else:
            ax.plot([], [], label="cka (missing)")

        if "mutual_knn" in grouped[scenario]:
            for tk in args.topk:
                if tk in grouped[scenario]["mutual_knn"]:
                    y = grouped[scenario]["mutual_knn"][tk]["scores"]
                    ax.plot(x, y, marker="o", linewidth=2, label=f"mutual_knn@{tk} (mean={y.mean():.3f})")
                else:
                    ax.plot([], [], label=f"mutual_knn@{tk} (missing)")
        else:
            for tk in args.topk:
                ax.plot([], [], label=f"mutual_knn@{tk} (missing)")

        ax.set_xticks(x)
        ax.set_xticklabels(layers)
        ax.set_ylim(0.0, 1.0)
        ax.set_xlabel("Layer")
        ax.set_ylabel("Metric value")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)

    plt.tight_layout(rect=[0, 0.02, 1, 0.95])

    if args.save is not None:
        os.makedirs(os.path.dirname(args.save) or ".", exist_ok=True)
        plt.savefig(args.save, dpi=200)
        print(f"Saved figure to: {args.save}")

    if not args.no_show:
        plt.show()


if __name__ == "__main__":
    main()
