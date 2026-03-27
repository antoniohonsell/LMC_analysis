#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
from torch.utils.data import DataLoader

_HERE = Path(__file__).resolve().parent
_SRC  = _HERE.parent.parent / "src"
for _p in (str(_SRC), str(_HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import architectures
import datasets
import utils

from hz_metrics import compute_all_metrics
from train_track_crh_pah_mlp_v2 import (
    compute_crh_matrices_all_layers,
    flatten_pair,
    fro_norm,
    parse_int_list,
    seed_worker,
    set_seed,
    trace_remove,
)


def load_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def normalize_state_dict_keys(state: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    prefixes = ("module.", "model.", "net.")
    out = dict(state)
    changed = True
    while changed:
        changed = False
        keys = list(out.keys())
        for p in prefixes:
            if keys and all(k.startswith(p) for k in keys):
                out = {k[len(p):]: v for k, v in out.items()}
                changed = True
                break
    return out


def extract_state_dict(obj: Any) -> Dict[str, torch.Tensor]:
    if isinstance(obj, dict) and "state_dict" in obj and isinstance(obj["state_dict"], dict):
        return normalize_state_dict_keys(obj["state_dict"])
    if isinstance(obj, dict):
        return normalize_state_dict_keys(obj)
    raise ValueError(f"Unsupported checkpoint payload type: {type(obj)}")


def infer_epoch_from_path(path: Path) -> int:
    m = re.search(r"epoch(\d+)", path.name)
    if m:
        return int(m.group(1))
    if "final" in path.name.lower():
        return 10**9
    return -1


def checkpoint_candidates(run_dir: Path) -> List[Path]:
    pats = [
        "ckpt_epoch*.pth",
        "*_epoch*.pth",
        "*_final.pth",
        "model.pth",
        "*.pth",
    ]
    out: List[Path] = []
    seen = set()
    for pat in pats:
        for p in sorted(run_dir.glob(pat)):
            if p.is_file() and p not in seen:
                out.append(p)
                seen.add(p)
    out.sort(key=lambda p: (infer_epoch_from_path(p), p.name))
    return out


def build_loader(
    dataset_name: str,
    data_root: str,
    batch_size: int,
    num_workers: int,
    seed: int,
    device: torch.device,
    split: str,
) -> DataLoader:
    train_full, eval_full, _ = datasets.build_datasets(
        dataset_name,
        root=data_root,
        download=True,
        augment_train=False,
        normalize=True,
    )

    ds = train_full if split == "train" else eval_full

    if split == "train":
        g = torch.Generator().manual_seed(seed)
        return DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            worker_init_fn=seed_worker if num_workers > 0 else None,
            generator=g,
            pin_memory=(device.type == "cuda"),
        )

    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
    )


def M(A: torch.Tensor, B: torch.Tensor, *, allow_indefinite: bool, k: int, eig_tol: float,
      alpha_min: float, alpha_max: float, alpha_steps: int) -> Dict[str, Any]:
    return compute_all_metrics(
        A,
        B,
        sym=True,
        k=k,
        eig_tol=eig_tol,
        alpha_min=alpha_min,
        alpha_max=alpha_max,
        alpha_steps=alpha_steps,
        allow_indefinite=allow_indefinite,
    )


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--run_dir", type=str, required=True,
                   help="Path to the actual run directory containing checkpoints/config.json")
    p.add_argument("--data_root", type=str, default="./data")
    p.add_argument("--dataset", type=str, default=None)
    p.add_argument("--arch", type=str, default=None)
    p.add_argument("--hidden_dims", type=str, default=None)
    p.add_argument("--dropout", type=float, default=None)
    p.add_argument("--label_smoothing", type=float, default=None)
    p.add_argument("--seed", type=int, default=None)

    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--metrics_split", type=str, default="train", choices=["train", "eval"])
    p.add_argument("--max_batches", type=int, default=200)

    p.add_argument("--k", type=int, default=10)
    p.add_argument("--eig_tol", type=float, default=1e-12)
    p.add_argument("--alpha_min", type=float, default=0.0)
    p.add_argument("--alpha_max", type=float, default=3.0)
    p.add_argument("--alpha_steps", type=int, default=121)

    p.add_argument("--out_csv", type=str, default="crh_pah_metrics_all_layers_core.csv")
    args = p.parse_args()

    run_dir = Path(args.run_dir).expanduser().resolve()
    if not run_dir.exists():
        raise FileNotFoundError(f"run_dir does not exist: {run_dir}")

    cfg = load_json(run_dir / "config.json")

    dataset_name = (args.dataset or cfg.get("dataset") or "FASHIONMNIST").strip().upper()
    arch = (args.arch or cfg.get("arch") or "mnist_mlp_reg").strip()

    hidden_dims_raw = args.hidden_dims
    if hidden_dims_raw is None:
        hidden_cfg = cfg.get("hidden_dims", [2048, 512])
        if isinstance(hidden_cfg, str):
            hidden_dims_raw = hidden_cfg
        else:
            hidden_dims_raw = ",".join(str(x) for x in hidden_cfg)

    hidden_dims = parse_int_list(hidden_dims_raw)
    dropout = float(args.dropout if args.dropout is not None else cfg.get("dropout", 0.0))
    seed = int(args.seed if args.seed is not None else cfg.get("seed", 0))

    device = utils.get_device()
    set_seed(seed)

    stats = datasets.DATASET_STATS[dataset_name]
    num_classes = int(stats["num_classes"])
    in_channels = int(stats["in_channels"])
    input_shape = (in_channels, *tuple(stats["image_size"]))

    loader = build_loader(
        dataset_name=dataset_name,
        data_root=args.data_root,
        batch_size=int(args.batch_size),
        num_workers=int(args.num_workers),
        seed=seed,
        device=device,
        split=str(args.metrics_split),
    )

    model = architectures.build_model(
        arch,
        num_classes=num_classes,
        in_channels=in_channels,
        input_shape=input_shape,
        hidden_dims=hidden_dims,
        dropout=dropout,
    ).to(device)

    ckpts = checkpoint_candidates(run_dir)
    if not ckpts:
        raise FileNotFoundError(f"No checkpoints found in {run_dir}")

    rows: List[Dict[str, Any]] = []
    max_batches = None if int(args.max_batches) <= 0 else int(args.max_batches)

    for ckpt_path in ckpts:
        print(f"[INFO] processing {ckpt_path.name}")
        obj = torch.load(ckpt_path, map_location=str(device))
        state = extract_state_dict(obj)
        model.load_state_dict(state, strict=True)
        model.eval()

        epoch = infer_epoch_from_path(ckpt_path)
        all_layer_results = compute_crh_matrices_all_layers(
            model=model,
            loader=loader,
            device=device,
            max_batches=max_batches,
            g_mode="batch",
        )

        row: Dict[str, Any] = {
            "checkpoint": ckpt_path.name,
            "epoch": epoch,
            "dataset": dataset_name,
            "arch": arch,
            "hidden_dims": ",".join(str(x) for x in hidden_dims),
            "dropout": dropout,
        }

        for layer_name, mats in all_layer_results.items():
            if layer_name == "__last__":
                continue

            Ha, Hb = mats["Ha"], mats["Hb"]
            WTW, WWT = mats["WTW"], mats["WWT"]

            # backward core pair
            m_Ha_WTW = M(
                Ha, WTW,
                allow_indefinite=False,
                k=int(args.k),
                eig_tol=float(args.eig_tol),
                alpha_min=float(args.alpha_min),
                alpha_max=float(args.alpha_max),
                alpha_steps=int(args.alpha_steps),
            )
            row.update({f"{layer_name}_{k}": v for k, v in flatten_pair("Ha_WTW", m_Ha_WTW).items()})

            # forward core pair
            m_Hb_WWT = M(
                Hb, WWT,
                allow_indefinite=False,
                k=int(args.k),
                eig_tol=float(args.eig_tol),
                alpha_min=float(args.alpha_min),
                alpha_max=float(args.alpha_max),
                alpha_steps=int(args.alpha_steps),
            )
            row.update({f"{layer_name}_{k}": v for k, v in flatten_pair("Hb_WWT", m_Hb_WWT).items()})

            # optional trace-removed variants
            Ha_tr, WTW_tr = trace_remove(Ha), trace_remove(WTW)
            Hb_tr, WWT_tr = trace_remove(Hb), trace_remove(WWT)

            row[f"{layer_name}_||Ha_tr||F"] = fro_norm(Ha_tr)
            row[f"{layer_name}_||WTW_tr||F"] = fro_norm(WTW_tr)
            row[f"{layer_name}_||Hb_tr||F"] = fro_norm(Hb_tr)
            row[f"{layer_name}_||WWT_tr||F"] = fro_norm(WWT_tr)

            m_Ha_WTW_tr = M(
                Ha_tr, WTW_tr,
                allow_indefinite=True,
                k=int(args.k),
                eig_tol=float(args.eig_tol),
                alpha_min=float(args.alpha_min),
                alpha_max=float(args.alpha_max),
                alpha_steps=int(args.alpha_steps),
            )
            row.update({f"{layer_name}_{k}": v for k, v in flatten_pair("Ha_WTW_tr", m_Ha_WTW_tr).items()})

            m_Hb_WWT_tr = M(
                Hb_tr, WWT_tr,
                allow_indefinite=True,
                k=int(args.k),
                eig_tol=float(args.eig_tol),
                alpha_min=float(args.alpha_min),
                alpha_max=float(args.alpha_max),
                alpha_steps=int(args.alpha_steps),
            )
            row.update({f"{layer_name}_{k}": v for k, v in flatten_pair("Hb_WWT_tr", m_Hb_WWT_tr).items()})

        # keep last-layer unprefixed columns too
        if "__last__" in all_layer_results:
            mats = all_layer_results["__last__"]
            Ha, Hb = mats["Ha"], mats["Hb"]
            WTW, WWT = mats["WTW"], mats["WWT"]

            row.update(flatten_pair(
                "Ha_WTW",
                M(Ha, WTW, allow_indefinite=False,
                  k=int(args.k), eig_tol=float(args.eig_tol),
                  alpha_min=float(args.alpha_min), alpha_max=float(args.alpha_max),
                  alpha_steps=int(args.alpha_steps))
            ))
            row.update(flatten_pair(
                "Hb_WWT",
                M(Hb, WWT, allow_indefinite=False,
                  k=int(args.k), eig_tol=float(args.eig_tol),
                  alpha_min=float(args.alpha_min), alpha_max=float(args.alpha_max),
                  alpha_steps=int(args.alpha_steps))
            ))

        rows.append(row)

    rows.sort(key=lambda r: (r["epoch"], r["checkpoint"]))

    out_csv = Path(args.out_csv)
    if not out_csv.is_absolute():
        out_csv = run_dir / out_csv

    fieldnames = sorted({k for row in rows for k in row.keys()})
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nSaved: {out_csv}")


if __name__ == "__main__":
    main()