#!/usr/bin/env python3
"""
CRH/compute_crh_resnet.py

Computes CRH metrics for ALL layers (Conv2d + Linear) of a trained ResNet18.

For each layer the following matrix pairs are computed and compared via
hz_metrics.compute_all_metrics:

  Output-side (always):
    Hb  = E[h_b h_b^T]   (output representation correlation, shape C_out × C_out)
    WWT = W W^T           (filter correlation,                shape C_out × C_out)

  Input-side (only when C_in * kH * kW <= --max_input_dim, default 512):
    Ha  = E[h_a h_a^T]   (input representation correlation,  shape d_in × d_in)
    WTW = W^T W           (filter correlation,                shape d_in × d_in)

For Conv2d layers:
  h_a: input patches obtained via F.unfold  → shape (B·H_out·W_out, C_in·kH·kW)
  h_b: output feature map flattened spatially → shape (B·H_out·W_out, C_out)
  W  : weight reshaped to (C_out, C_in·kH·kW)

For Linear layers:
  h_a: inputs[0] → shape (B, d_in)
  h_b: outputs (before bias)  → shape (B, d_out)
  W  : weight → shape (d_out, d_in)

Both raw and trace-removed variants of each pair are stored.

Results are saved to a CSV with one row per (checkpoint, layer).

Usage (from repo root):
  python CRH/compute_crh_resnet.py \\
    --run_dir ./CRH_resnet_out/CIFAR10/final/sgd/seed_0 \\
    --dataset CIFAR10 \\
    --out_csv  ./CRH_resnet_out/CIFAR10/final/sgd/seed_0/crh_metrics.csv

  # All seeds of one optimizer at once:
  for seed in 0 1 2; do
    python CRH/compute_crh_resnet.py \\
      --run_dir ./CRH_resnet18_out/CIFAR10/final/sgd/seed_$seed \\
      --dataset CIFAR10
  done
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

_HERE = Path(__file__).resolve().parent
_SRC  = _HERE.parent.parent / "src"
for _p in (str(_SRC), str(_HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import architectures  # type: ignore
import datasets       # type: ignore
import utils          # type: ignore
from hz_metrics import compute_all_metrics  # type: ignore


# --------------------------------------------------------------------------- #
# Utilities                                                                    #
# --------------------------------------------------------------------------- #

def load_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def infer_epoch_from_path(path: Path) -> int:
    m = re.search(r"epoch(\d+)", path.name)
    if m:
        return int(m.group(1))
    if "final" in path.name.lower():
        return 10**9
    if "best" in path.name.lower():
        return -1
    return -2


def checkpoint_candidates(run_dir: Path) -> List[Path]:
    """Return all checkpoint .pth files sorted by epoch."""
    pats = ["*_epoch*.pth", "*_final.pth", "*_best.pth", "*.pth"]
    seen: set = set()
    out: List[Path] = []
    for pat in pats:
        for p in sorted(run_dir.glob(pat)):
            if p.is_file() and p not in seen:
                out.append(p)
                seen.add(p)
    out.sort(key=lambda p: (infer_epoch_from_path(p), p.name))
    return out


def normalize_state_dict(obj: Any) -> Dict[str, torch.Tensor]:
    if isinstance(obj, dict) and "state_dict" in obj:
        sd = obj["state_dict"]
    elif isinstance(obj, dict):
        sd = obj
    else:
        raise ValueError(f"Unrecognised checkpoint type: {type(obj)}")
    prefixes = ("module.", "model.", "net.")
    changed = True
    while changed:
        changed = False
        keys = list(sd.keys())
        for pfx in prefixes:
            if keys and all(k.startswith(pfx) for k in keys):
                sd = {k[len(pfx):]: v for k, v in sd.items()}
                changed = True
                break
    return sd


def fro_norm(A: torch.Tensor) -> float:
    return float(torch.linalg.norm(A, ord="fro").item())


def trace_remove(A: torch.Tensor) -> torch.Tensor:
    n = A.shape[0]
    tr = torch.trace(A)
    return A - (tr / float(n)) * torch.eye(n, dtype=A.dtype, device=A.device)


def flatten_metrics(prefix: str, m: Dict[str, Any]) -> Dict[str, Any]:
    """Flatten a nested metrics dict into flat key→scalar pairs."""
    out: Dict[str, Any] = {}
    for k, v in m.items():
        if k in ("shape", "settings"):
            continue
        if isinstance(v, dict):
            for kk, vv in v.items():
                if isinstance(vv, (int, float)):
                    out[f"{prefix}__{k}__{kk}"] = vv
        elif isinstance(v, (int, float)):
            out[f"{prefix}__{k}"] = v
    return out


# --------------------------------------------------------------------------- #
# Layer enumeration                                                            #
# --------------------------------------------------------------------------- #

def find_conv_and_linear_layers(
    model: nn.Module,
) -> List[Tuple[str, nn.Module]]:
    """Return (name, module) for all Conv2d and Linear layers in order."""
    layers = []
    for name, module in model.named_modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            layers.append((name, module))
    return layers


# --------------------------------------------------------------------------- #
# Activation accumulation                                                      #
# --------------------------------------------------------------------------- #

@torch.no_grad()
def compute_crh_matrices_resnet(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    *,
    max_batches: Optional[int] = None,
    max_input_dim: int = 512,
) -> Dict[str, Dict[str, Any]]:
    """
    Registers forward hooks on all Conv2d and Linear layers, runs the
    dataloader, and accumulates:
      - Ha, Hb  (representation correlations)
      - WTW, WWT (weight correlations, computed once from the final weights)

    Returns a dict: layer_name → {Ha, Hb, WTW, WWT, d_in, d_out, layer_type}
    Ha / WTW are None for layers where d_in > max_input_dim.
    """
    model = model.to(device)
    model.eval()

    layers = find_conv_and_linear_layers(model)
    if not layers:
        raise RuntimeError("No Conv2d or Linear layers found in the model.")

    # Buffers: initialised lazily on the first batch.
    buffers: Dict[str, Dict[str, Any]] = {name: {} for name, _ in layers}
    cache:   Dict[str, Dict[str, torch.Tensor]] = {name: {} for name, _ in layers}

    # Register forward hooks.
    handles = []

    def _make_hook(lname: str, lmodule: nn.Module):
        def hook(module, inputs, outputs):
            x = inputs[0].detach()
            y = outputs.detach()

            if isinstance(module, nn.Conv2d):
                B, C_out, H_out, W_out = y.shape
                # Unfold input → patches: (B, C_in*kH*kW, H_out*W_out)
                patches = F.unfold(
                    x,
                    kernel_size=module.kernel_size,
                    dilation=module.dilation,
                    padding=module.padding,
                    stride=module.stride,
                )
                # (B*H_out*W_out, C_in*kH*kW)
                h_a = patches.permute(0, 2, 1).reshape(B * H_out * W_out, -1)
                # (B*H_out*W_out, C_out)
                h_b = y.permute(0, 2, 3, 1).reshape(B * H_out * W_out, C_out)
            else:  # nn.Linear
                h_a = x                              # (B, d_in)
                h_b = y                              # (B, d_out)
                if module.bias is not None:
                    h_b = h_b - module.bias.detach().unsqueeze(0)

            cache[lname]["h_a"] = h_a
            cache[lname]["h_b"] = h_b

        return hook

    for lname, lmodule in layers:
        handles.append(lmodule.register_forward_hook(_make_hook(lname, lmodule)))

    n_samples = 0

    for bi, (x, y) in enumerate(loader):
        if max_batches is not None and bi >= max_batches:
            break
        x = x.to(device)
        _ = model(x)

        N = x.size(0)
        n_samples += N

        for lname, lmodule in layers:
            if "h_a" not in cache[lname]:
                continue

            h_a = cache[lname]["h_a"]  # (N_eff, d_in)
            h_b = cache[lname]["h_b"]  # (N_eff, d_out)
            N_eff = h_a.shape[0]
            d_in  = h_a.shape[1]
            d_out = h_b.shape[1]

            buf = buffers[lname]

            # Lazy initialisation.
            if "Hb" not in buf:
                buf["d_in"]  = d_in
                buf["d_out"] = d_out
                buf["N_eff"] = 0
                buf["layer_type"] = "conv" if isinstance(lmodule, nn.Conv2d) else "linear"
                buf["Hb"] = torch.zeros(d_out, d_out, dtype=torch.float32, device=device)
                if d_in <= max_input_dim:
                    buf["Ha"] = torch.zeros(d_in, d_in, dtype=torch.float32, device=device)
                else:
                    buf["Ha"] = None

            buf["Hb"]   += h_b.float().T @ h_b.float()
            buf["N_eff"] += N_eff
            if buf["Ha"] is not None:
                buf["Ha"] += h_a.float().T @ h_a.float()

            cache[lname].clear()

    for handle in handles:
        handle.remove()

    if n_samples == 0:
        raise RuntimeError("No samples processed.")

    # Finalise: normalise by N_eff and compute weight matrices.
    results: Dict[str, Dict[str, Any]] = {}
    for lname, lmodule in layers:
        buf = buffers[lname]
        if "Hb" not in buf:
            continue

        N_eff = buf["N_eff"]
        Hb = (buf["Hb"] / float(N_eff)).cpu()
        Ha = (buf["Ha"] / float(N_eff)).cpu() if buf["Ha"] is not None else None

        # Weight matrices.
        if isinstance(lmodule, nn.Conv2d):
            W = lmodule.weight.detach().reshape(lmodule.out_channels, -1).cpu().float()
        else:
            W = lmodule.weight.detach().cpu().float()

        WWT = W @ W.T   # (d_out, d_out)
        WTW = W.T @ W   # (d_in,  d_in)  — kept in memory but may be large

        results[lname] = {
            "Ha":         Ha,
            "Hb":         Hb,
            "WTW":        WTW if Ha is not None else None,
            "WWT":        WWT,
            "d_in":       buf["d_in"],
            "d_out":      buf["d_out"],
            "layer_type": buf["layer_type"],
            "N_eff":      N_eff,
        }

    return results


# --------------------------------------------------------------------------- #
# Metrics computation                                                           #
# --------------------------------------------------------------------------- #

def compute_metrics_for_layer(
    mats: Dict[str, Any],
    *,
    k: int,
    eig_tol: float,
    alpha_min: float,
    alpha_max: float,
    alpha_steps: int,
) -> Dict[str, Any]:
    """
    Given pre-computed matrices for one layer, compute all CRH metrics
    and return a flat dict of scalars.
    """
    row: Dict[str, Any] = {
        "d_in":       mats["d_in"],
        "d_out":      mats["d_out"],
        "layer_type": mats["layer_type"],
        "N_eff":      mats["N_eff"],
    }

    def _run(prefix: str, A: torch.Tensor, B: torch.Tensor, indefinite: bool) -> None:
        try:
            m = compute_all_metrics(
                A, B,
                sym=True,
                k=k,
                eig_tol=eig_tol,
                alpha_min=alpha_min,
                alpha_max=alpha_max,
                alpha_steps=alpha_steps,
                allow_indefinite=indefinite,
            )
            row.update(flatten_metrics(prefix, m))
            row[f"{prefix}__||A||_F"] = fro_norm(A)
            row[f"{prefix}__||B||_F"] = fro_norm(B)
        except Exception as e:
            row[f"{prefix}__error"] = str(e)

    # Output-side: Hb vs WWT (always).
    _run("Hb_WWT", mats["Hb"], mats["WWT"], indefinite=False)

    # Output-side trace-removed.
    Hb_tr  = trace_remove(mats["Hb"])
    WWT_tr = trace_remove(mats["WWT"])
    _run("Hb_WWT_tr", Hb_tr, WWT_tr, indefinite=True)
    row["Hb_tr__||F||"]  = fro_norm(Hb_tr)
    row["WWT_tr__||F||"] = fro_norm(WWT_tr)

    # Input-side: Ha vs WTW (only if available).
    if mats["Ha"] is not None and mats["WTW"] is not None:
        _run("Ha_WTW", mats["Ha"], mats["WTW"], indefinite=False)

        Ha_tr  = trace_remove(mats["Ha"])
        WTW_tr = trace_remove(mats["WTW"])
        _run("Ha_WTW_tr", Ha_tr, WTW_tr, indefinite=True)
        row["Ha_tr__||F||"]  = fro_norm(Ha_tr)
        row["WTW_tr__||F||"] = fro_norm(WTW_tr)
    else:
        row["Ha_WTW__skipped"] = f"d_in={mats['d_in']} > max_input_dim"

    return row


# --------------------------------------------------------------------------- #
# Main                                                                         #
# --------------------------------------------------------------------------- #

def main() -> None:
    p = argparse.ArgumentParser(
        description="Compute CRH metrics for all Conv2d + Linear layers of a ResNet18 run."
    )
    p.add_argument("--run_dir",   type=str, required=True,
                   help="Directory containing the model checkpoints (and config.json).")
    p.add_argument("--dataset",   type=str, default=None,
                   help="Dataset name (CIFAR10 or CIFAR100). "
                        "Inferred from config.json if omitted.")
    p.add_argument("--arch",      type=str, default="resnet18")
    p.add_argument("--norm",      type=str, default=None,
                   help="Normalisation type. Inferred from config.json if omitted (default: bn).")

    p.add_argument("--data_root",    type=str, default="./data")
    p.add_argument("--split",        type=str, default="train",
                   choices=["train", "val"],
                   help="Which split to use for computing representations. Default: train.")
    p.add_argument("--batch_size",   type=int, default=256)
    p.add_argument("--num_workers",  type=int, default=4)
    p.add_argument("--max_batches",  type=int, default=0,
                   help="Max batches per checkpoint (0 = all).")
    p.add_argument("--max_input_dim", type=int, default=512,
                   help="Skip input-side (Ha/WTW) for layers where d_in > this. "
                        "Default: 512. Set 0 to always skip, -1 to never skip.")
    p.add_argument("--seed", type=int, default=0)

    p.add_argument("--k",           type=int,   default=10)
    p.add_argument("--eig_tol",     type=float, default=1e-12)
    p.add_argument("--alpha_min",   type=float, default=0.0)
    p.add_argument("--alpha_max",   type=float, default=3.0)
    p.add_argument("--alpha_steps", type=int,   default=121)

    p.add_argument("--out_csv", type=str, default=None,
                   help="Output CSV path. Default: <run_dir>/crh_metrics_resnet.csv")
    args = p.parse_args()

    run_dir = Path(args.run_dir).expanduser().resolve()
    if not run_dir.exists():
        raise FileNotFoundError(f"run_dir not found: {run_dir}")

    # Load config and infer missing args.
    cfg = load_json(run_dir / "config.json")
    dataset_name = (args.dataset or cfg.get("dataset") or "CIFAR10").strip().upper()
    norm         = str(args.norm or cfg.get("resnet_norm", "bn"))

    max_batches  = None if args.max_batches <= 0 else int(args.max_batches)
    max_input_dim = int(args.max_input_dim) if args.max_input_dim >= 0 else 10**9

    out_csv = Path(args.out_csv) if args.out_csv else run_dir / "crh_metrics_resnet.csv"
    out_csv = out_csv if out_csv.is_absolute() else run_dir / out_csv

    device = utils.get_device()

    # Build dataset.
    stats       = datasets.DATASET_STATS[dataset_name]
    num_classes = int(stats["num_classes"])
    in_channels = int(stats["in_channels"])

    train_ds, eval_ds, _ = datasets.build_datasets(
        dataset_name, root=args.data_root, download=True,
        augment_train=False, normalize=True,
    )
    ds = train_ds if args.split == "train" else eval_ds
    loader = DataLoader(
        ds,
        batch_size=int(args.batch_size),
        shuffle=False,
        num_workers=int(args.num_workers),
        pin_memory=(device.type == "cuda"),
    )

    # Build model.
    model = architectures.build_model(
        "resnet18",
        num_classes=num_classes,
        in_channels=in_channels,
        norm=norm,
    ).to(device)

    # Discover checkpoints.
    ckpts = checkpoint_candidates(run_dir)
    if not ckpts:
        raise FileNotFoundError(f"No .pth checkpoints found in {run_dir}")
    print(f"Found {len(ckpts)} checkpoint(s) in {run_dir}")

    all_rows: List[Dict[str, Any]] = []

    for ckpt_path in ckpts:
        epoch = infer_epoch_from_path(ckpt_path)
        print(f"\n[ckpt] {ckpt_path.name}  (epoch={epoch})")

        obj   = torch.load(ckpt_path, map_location=device)
        state = normalize_state_dict(obj)
        model.load_state_dict(state, strict=True)

        # Accumulate CRH matrices for all layers.
        layer_mats = compute_crh_matrices_resnet(
            model, loader, device,
            max_batches=max_batches,
            max_input_dim=max_input_dim,
        )

        # Compute metrics per layer.
        for layer_name, mats in layer_mats.items():
            print(f"  [{layer_name}] type={mats['layer_type']}  "
                  f"d_in={mats['d_in']}  d_out={mats['d_out']}  N_eff={mats['N_eff']}")
            metrics = compute_metrics_for_layer(
                mats,
                k=int(args.k),
                eig_tol=float(args.eig_tol),
                alpha_min=float(args.alpha_min),
                alpha_max=float(args.alpha_max),
                alpha_steps=int(args.alpha_steps),
            )
            row = {
                "checkpoint":    ckpt_path.name,
                "epoch":         epoch,
                "dataset":       dataset_name,
                "arch":          "resnet18",
                "norm":          norm,
                "layer":         layer_name,
            }
            row.update(metrics)
            all_rows.append(row)

    # Sort by (epoch, layer).
    all_rows.sort(key=lambda r: (r["epoch"], r["layer"]))

    # Write CSV.
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(dict.fromkeys(k for row in all_rows for k in row))
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(all_rows)

    print(f"\nSaved {len(all_rows)} rows → {out_csv}")


if __name__ == "__main__":
    main()
