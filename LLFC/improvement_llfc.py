#!/usr/bin/env python3
"""
improve_llfc.py

Compute how much the LLFC metric improved "after permutation", given results saved in .pt files
produced by LLFC/compute_llfc.py.

Typical workflow:
  1) Run LLFC without permutation -> saves llfc_cos_*.pt
  2) Run LLFC with --wm_perm ...   -> saves llfc_cos_*.pt
  3) Compare:
       python improve_llfc.py path/to/no_perm.pt path/to/with_perm.pt

This script compares:
  - layer-averaged LLFC curve: cos_mean_layeravg (shape [K])
  - optionally layerwise LLFC: cos_mean (shape [L, K])

Outputs:
  - delta at lambda_value (default 0.5)
  - mean delta across lambdas
  - AUC delta across lambdas
  - (optional) top-k layer deltas at the selected lambda

"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple, List

import torch


def _as_tensor_1d(x: Any, name: str) -> torch.Tensor:
    t = x if isinstance(x, torch.Tensor) else torch.as_tensor(x)
    if t.ndim != 1:
        raise ValueError(f"Expected {name} to be 1D, got shape={tuple(t.shape)}")
    return t.detach().cpu()


def _as_tensor_2d(x: Any, name: str) -> torch.Tensor:
    t = x if isinstance(x, torch.Tensor) else torch.as_tensor(x)
    if t.ndim != 2:
        raise ValueError(f"Expected {name} to be 2D, got shape={tuple(t.shape)}")
    return t.detach().cpu()


def _load_pt(path: str) -> Any:
    return torch.load(path, map_location="cpu")


def _looks_like_llfc_dict(d: Any) -> bool:
    if not isinstance(d, dict):
        return False
    needed = {"lambdas", "cos_mean_layeravg"}
    return all(k in d for k in needed)


def _extract_llfc(obj: Any, role: str) -> Dict[str, Any]:
    """
    Accepts:
      - direct dict with LLFC keys
      - dict containing nested LLFC dicts under common names
      - list/tuple of dicts containing LLFC dicts
    """
    if _looks_like_llfc_dict(obj):
        return obj

    # Common nesting patterns
    if isinstance(obj, dict):
        for k in ("before", "after", "baseline", "permuted", "no_perm", "with_perm", "unpermuted", "aligned"):
            v = obj.get(k, None)
            if _looks_like_llfc_dict(v):
                return v

        # If it's a dict of runs, try heuristics:
        # pick the first dict that looks like LLFC
        for v in obj.values():
            if _looks_like_llfc_dict(v):
                return v

    # List/tuple of dicts: pick first LLFC-like
    if isinstance(obj, (list, tuple)):
        cands = [v for v in obj if _looks_like_llfc_dict(v)]
        if len(cands) == 1:
            return cands[0]
        if len(cands) > 1:
            # try to pick baseline/permuted by wm_perm presence
            if role in ("before", "baseline"):
                for v in cands:
                    if isinstance(v, dict) and v.get("wm_perm", None) in (None, "", False):
                        return v
            if role in ("after", "permuted"):
                for v in cands:
                    if isinstance(v, dict) and v.get("wm_perm", None) not in (None, "", False):
                        return v
            return cands[0]

    raise ValueError(
        f"Could not extract LLFC dict for {role}. "
        f"Expected keys like 'lambdas' and 'cos_mean_layeravg' in the loaded object."
    )


def _nearest_index(x: torch.Tensor, val: float) -> int:
    # x: 1D
    return int(torch.argmin(torch.abs(x - float(val))).item())


def _trapz(y: torch.Tensor, x: torch.Tensor) -> float:
    """
    Trapezoidal integration of y(x).
    Assumes x is 1D increasing; y is 1D same length.
    """
    if y.numel() < 2:
        return float("nan")
    dx = x[1:] - x[:-1]
    return float(((y[1:] + y[:-1]) * 0.5 * dx).sum().item())


def plot_llfc_improvement(
    lambdas: torch.Tensor,
    base_avg: torch.Tensor,
    perm_avg: torch.Tensor,
    out_dir: str,
    title: str = "LLFC improvement",
    lambda_mark: float | None = None,
    base_cos: torch.Tensor | None = None,   # optional [L, K]
    perm_cos: torch.Tensor | None = None,   # optional [L, K]
    layers: list[str] | None = None,
) -> None:
    """
    Saves:
      - llfc_layeravg_before_after.png (before vs after curve)
      - llfc_layeravg_delta.png        (delta curve, annotated)
      - llfc_layerwise_delta_heatmap.png (optional, if base_cos/perm_cos provided)
    """
    import os
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(out_dir, exist_ok=True)

    # convert to cpu numpy
    x = lambdas.detach().cpu().numpy()
    y0 = base_avg.detach().cpu().numpy()
    y1 = perm_avg.detach().cpu().numpy()
    d = (perm_avg - base_avg).detach().cpu().numpy()

    # ---- Plot 1: before vs after (layer-avg) ----
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(x, y0, linewidth=2.5, label="before (no perm)")
    ax.plot(x, y1, linewidth=2.5, label="after (perm)")
    if lambda_mark is not None:
        ax.axvline(float(lambda_mark), linestyle="dashed", linewidth=1.5, alpha=0.8)
    ax.set_xlabel(r"$\lambda$")
    ax.set_ylabel("LLFC cosine similarity (layer-avg)")
    ax.set_title(title)
    ax.set_ylim(-1.05, 1.05)
    ax.legend(framealpha=0.7)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "llfc_layeravg_before_after.png"), dpi=250)
    plt.close(fig)

    # ---- Plot 2: delta curve (layer-avg) with simple annotation ----
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(x, d, linewidth=2.5, label=r"$\Delta$ (after - before)")
    ax.axhline(0.0, linewidth=1.2, alpha=0.7)
    if lambda_mark is not None:
        ax.axvline(float(lambda_mark), linestyle="dashed", linewidth=1.5, alpha=0.8)

        # annotate delta at nearest lambda
        li = int(torch.argmin(torch.abs(lambdas.detach().cpu() - float(lambda_mark))).item())
        ax.scatter([x[li]], [d[li]])
        ax.text(x[li], d[li], f"  Δ@λ≈{x[li]:.3f}: {d[li]:+.4f}", va="center")

    ax.set_xlabel(r"$\lambda$")
    ax.set_ylabel(r"$\Delta$ LLFC (layer-avg)")
    ax.set_title(title)
    ax.legend(framealpha=0.7)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "llfc_layeravg_delta.png"), dpi=250)
    plt.close(fig)

    # ---- Plot 3 (optional): layerwise delta heatmap ----
    if base_cos is not None and perm_cos is not None:
        A = base_cos.detach().cpu()
        B = perm_cos.detach().cpu()
        if A.ndim == 2 and B.ndim == 2 and A.shape == B.shape:
            delta_LK = (B - A).numpy()  # [L, K]
            fig = plt.figure()
            ax = fig.add_subplot(111)
            im = ax.imshow(
                delta_LK,
                aspect="auto",
                origin="lower",
                extent=[float(x[0]), float(x[-1]), 0, delta_LK.shape[0] - 1],
            )
            fig.colorbar(im, ax=ax, label=r"$\Delta$ cosine (after - before)")
            ax.set_xlabel(r"$\lambda$")
            ax.set_ylabel("layer index")
            ax.set_title(title + " (layerwise Δ)")
            if layers is not None and len(layers) == delta_LK.shape[0] and len(layers) <= 40:
                ax.set_yticks(list(range(len(layers))))
                ax.set_yticklabels(layers)
            fig.tight_layout()
            fig.savefig(os.path.join(out_dir, "llfc_layerwise_delta_heatmap.png"), dpi=250)
            plt.close(fig)


@dataclass
class Summary:
    lambda_value: float
    lambda_index: int
    lambda_actual: float

    base_at: float
    perm_at: float
    delta_at: float
    relpct_at: float

    mean_base: float
    mean_perm: float
    mean_delta: float
    mean_relpct: float

    auc_base: float
    auc_perm: float
    auc_delta: float
    auc_relpct: float


def main() -> None:
    p = argparse.ArgumentParser(description="Compute LLFC improvement after permutation from .pt results.")
    p.add_argument("before", type=str, help="Path to LLFC .pt results WITHOUT permutation (baseline).")
    p.add_argument("after", type=str, help="Path to LLFC .pt results WITH permutation (permuted/aligned).")
    p.add_argument("--lambda_value", type=float, default=0.5, help="Lambda at which to report point improvement (default: 0.5).")
    p.add_argument("--topk_layers", type=int, default=0, help="If >0, report top-k layer improvements at lambda_value (requires cos_mean).")
    p.add_argument("--out_json", type=str, default=None, help="Optional: write a JSON report to this path.")
    p.add_argument("--plot_dir", type=str, default=None, help="If set, save plots to this directory.")
    args = p.parse_args()

    obj_before = _load_pt(args.before)
    obj_after = _load_pt(args.after)

    base = _extract_llfc(obj_before, role="before")
    perm = _extract_llfc(obj_after, role="after")

    lamb_base = _as_tensor_1d(base["lambdas"], "lambdas")
    lamb_perm = _as_tensor_1d(perm["lambdas"], "lambdas")

    if lamb_base.numel() != lamb_perm.numel() or not torch.allclose(lamb_base, lamb_perm, atol=0.0, rtol=0.0):
        raise ValueError(
            "The two results files have different 'lambdas'. "
            "Re-run LLFC with the same --lambdas grid (same number of steps) before comparing."
        )

    base_avg = _as_tensor_1d(base["cos_mean_layeravg"], "cos_mean_layeravg").double()
    perm_avg = _as_tensor_1d(perm["cos_mean_layeravg"], "cos_mean_layeravg").double()
    if base_avg.numel() != lamb_base.numel() or perm_avg.numel() != lamb_base.numel():
        raise ValueError("Shape mismatch: cos_mean_layeravg must have same length as lambdas.")

    # Compute summary metrics (always needed)
    delta = perm_avg - base_avg
    eps = 1e-12
    relpct = 100.0 * delta / (base_avg.abs() + eps)

    li = _nearest_index(lamb_base.double(), float(args.lambda_value))
    lam_actual = float(lamb_base[li].item())

    base_at = float(base_avg[li].item())
    perm_at = float(perm_avg[li].item())
    delta_at = float(delta[li].item())
    relpct_at = float(relpct[li].item())

    mean_base = float(base_avg.mean().item())
    mean_perm = float(perm_avg.mean().item())
    mean_delta = float(delta.mean().item())
    mean_relpct = float((100.0 * mean_delta / (abs(mean_base) + eps)) if not math.isnan(mean_base) else float("nan"))

    auc_base = _trapz(base_avg, lamb_base.double())
    auc_perm = _trapz(perm_avg, lamb_base.double())
    auc_delta = auc_perm - auc_base
    auc_relpct = 100.0 * auc_delta / (abs(auc_base) + eps)

    # Plot if requested
    if args.plot_dir is not None:
        base_cos = None
        perm_cos = None
        layers = None
        if "cos_mean" in base and "cos_mean" in perm:
            base_cos = base["cos_mean"]
            perm_cos = perm["cos_mean"]
            layers = base.get("layers", None) or perm.get("layers", None)

        plot_llfc_improvement(
            lambdas=lamb_base.double(),
            base_avg=base_avg,
            perm_avg=perm_avg,
            out_dir=args.plot_dir,
            title="LLFC improvement",
            lambda_mark=args.lambda_value,
            base_cos=base_cos,
            perm_cos=perm_cos,
            layers=layers,
        )

    # Header metadata (best-effort)
    meta_fields = ("dataset", "arch", "norm", "width_multiplier", "shortcut_option", "wm_perm", "wm_perm_invert")
    meta = {}
    for k in meta_fields:
        if k in base or k in perm:
            meta[k] = {"before": base.get(k, None), "after": perm.get(k, None)}

    summary = Summary(
        lambda_value=float(args.lambda_value),
        lambda_index=int(li),
        lambda_actual=lam_actual,
        base_at=base_at,
        perm_at=perm_at,
        delta_at=delta_at,
        relpct_at=relpct_at,
        mean_base=mean_base,
        mean_perm=mean_perm,
        mean_delta=mean_delta,
        mean_relpct=mean_relpct,
        auc_base=auc_base,
        auc_perm=auc_perm,
        auc_delta=auc_delta,
        auc_relpct=float(auc_relpct),
    )

    # Print report
    print("=== LLFC Improvement Report ===")
    if meta:
        print("Metadata (best-effort):")
        for k, v in meta.items():
            print(f"  {k}: before={v['before']} | after={v['after']}")
    print("")
    print(f"Lambda target={summary.lambda_value:.4f} -> nearest grid λ={summary.lambda_actual:.4f} (index {summary.lambda_index})")
    print(f"Layer-avg LLFC at λ: before={summary.base_at:.6f} | after={summary.perm_at:.6f} | Δ={summary.delta_at:+.6f} | rel={summary.relpct_at:+.2f}%")
    print("")
    print(f"Mean over λ:          before={summary.mean_base:.6f} | after={summary.mean_perm:.6f} | Δ={summary.mean_delta:+.6f} | rel={summary.mean_relpct:+.2f}%")
    print(f"AUC over λ:           before={summary.auc_base:.6f} | after={summary.auc_perm:.6f} | Δ={summary.auc_delta:+.6f} | rel={summary.auc_relpct:+.2f}%")

    # Optional layerwise top-k at lambda_value
    if int(args.topk_layers) > 0:
        if "cos_mean" not in base or "cos_mean" not in perm:
            print("\n[warn] --topk_layers requested but 'cos_mean' missing in one of the files; skipping layerwise report.")
        else:
            base_cos = _as_tensor_2d(base["cos_mean"], "cos_mean").double()
            perm_cos = _as_tensor_2d(perm["cos_mean"], "cos_mean").double()
            if base_cos.shape != perm_cos.shape:
                print("\n[warn] 'cos_mean' shapes differ; skipping layerwise report.")
            else:
                dL = perm_cos[:, li] - base_cos[:, li]  # [L]
                layers = base.get("layers", None) or perm.get("layers", None)
                if not isinstance(layers, list) or len(layers) != dL.numel():
                    layers = [f"layer_{i}" for i in range(dL.numel())]
                topk = min(int(args.topk_layers), dL.numel())
                vals, idx = torch.topk(dL, k=topk, largest=True)
                print(f"\nTop-{topk} layer improvements at λ={summary.lambda_actual:.4f}:")
                for rank in range(topk):
                    i = int(idx[rank].item())
                    print(f"  {rank+1:>2d}. {layers[i]}: Δ={float(vals[rank].item()):+.6f} "
                          f"(before={float(base_cos[i, li].item()):.6f}, after={float(perm_cos[i, li].item()):.6f})")

    # Optional JSON
    if args.out_json is not None:
        report = {
            "before_path": args.before,
            "after_path": args.after,
            "meta": meta,
            "lambdas": lamb_base.tolist(),
            "base_cos_mean_layeravg": base_avg.tolist(),
            "perm_cos_mean_layeravg": perm_avg.tolist(),
            "delta_cos_mean_layeravg": delta.tolist(),
            "summary": {
                **summary.__dict__,
            },
        }
        with open(args.out_json, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
        print(f"\nWrote JSON report: {args.out_json}")


if __name__ == "__main__":
    main()