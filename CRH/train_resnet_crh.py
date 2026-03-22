#!/usr/bin/env python3
"""
CRH/train_resnet_crh.py

Trains ResNet18 (BatchNorm) on CIFAR-10 and/or CIFAR-100 for CRH analysis.

Pipeline:
  Phase 1 — Tune lr and weight_decay for SGD and AdamW independently,
             using a single fixed tuning seed. Best hparams selected by
             max validation accuracy (tie-break: min validation loss).
  Phase 2 — Train N seeds per optimizer with the best hparams.
             Periodic checkpoints are saved every --save_every epochs so
             that CRH metrics can be tracked across training.

Output structure:
  <out_dir>/<DATASET>/
    tuning/
      sgd/tuning_summary.json  ...trial checkpoints...
      adamw/tuning_summary.json  ...
    final/
      sgd/seed_0/   seed_1/   seed_2/
      adamw/seed_0/   seed_1/   seed_2/
    manifest.json

Usage (from repo root):
  python CRH/train_resnet_crh.py --dataset CIFAR10
  python CRH/train_resnet_crh.py --dataset CIFAR100
  python CRH/train_resnet_crh.py --dataset CIFAR10 --dataset CIFAR100

Note: uses ResNet18 with standard BatchNorm. No width/shortcut options.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

THIS_DIR  = Path(__file__).resolve().parent   # CRH/
REPO_ROOT = THIS_DIR.parent
SGD_DIR   = REPO_ROOT / "SGDvsAdam"

for _p in (str(REPO_ROOT), str(SGD_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import architectures  # type: ignore

# Reuse the full tuning/training infrastructure from SGDvsAdam.
import common as _common  # type: ignore
from common import (  # type: ignore
    TrainConfig,
    datasets,
    prepare_experiment_setup,
    save_json,
    train_run,
    tune_grid,
    utils,
)
import torch
from torch.utils.data import DataLoader, Subset

# Patch common.build_model to support resnet18 without modifying SGDvsAdam.
_orig_build_model = _common.build_model

def _patched_build_model(arch, dataset_name, *, resnet_norm="bn", **kwargs):
    if arch.lower() == "resnet18":
        stats = datasets.DATASET_STATS[dataset_name]
        return architectures.build_model(
            "resnet18",
            num_classes=int(stats["num_classes"]),
            in_channels=int(stats["in_channels"]),
            norm=resnet_norm,
        )
    return _orig_build_model(arch, dataset_name, resnet_norm=resnet_norm, **kwargs)

_common.build_model = _patched_build_model

# Local hparam grids for ResNet18 (not in common.py by design).
_RESNET18_GRIDS: Dict[str, Dict[str, Any]] = {
    "sgd":   {"quick": {"lr": [3e-2, 1e-1, 2e-1], "wd": [5e-4, 1e-3, 5e-3]},
              "full":  {"lr": [1e-2, 3e-2, 1e-1, 2e-1], "wd": [5e-4, 1e-3, 5e-3]}},
    "adamw": {"quick": {"lr": [1e-4, 3e-4, 1e-3], "wd": [1e-3, 5e-3, 1e-2]},
              "full":  {"lr": [1e-4, 3e-4, 1e-3, 3e-3], "wd": [1e-4, 1e-3, 5e-3, 1e-2]}},
}

def _resnet18_hparam_grid(optimizer_name: str, mode: str) -> Dict[str, Any]:
    return _RESNET18_GRIDS[optimizer_name][mode if mode in ("quick", "full") else "quick"]

OPTIMIZERS = ("sgd",)

# Schedules and cosine eta_min for ResNet18.
RESNET_SCHEDULE    = {"sgd": "warmup_cosine", "adamw": "cosine"}
RESNET_ETA_MIN     = {"sgd": 0.0,             "adamw": 1e-5}

DEFAULT_EPOCHS     = 200
DEFAULT_BATCH_SIZE = 256
DEFAULT_VAL_SIZE   = 5000


# --------------------------------------------------------------------------- #
# Helpers                                                                      #
# --------------------------------------------------------------------------- #

def _make_test_loader(test_ds, batch_size: int, num_workers: int) -> DataLoader:
    device = utils.get_device()
    return DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
    )


def _template_cfg(
    *,
    dataset_name: str,
    optimizer_name: str,
    args: argparse.Namespace,
) -> TrainConfig:
    return TrainConfig(
        dataset=dataset_name,
        arch="resnet18",
        optimizer_name=optimizer_name,
        lr=0.0,
        weight_decay=0.0,
        epochs=int(args.epochs),
        batch_size=int(args.batch_size),
        num_workers=int(args.num_workers),
        model_seed=0,           # overridden per run
        split_seed=int(args.split_seed),
        val_size=int(args.val_size),
        mlp_hidden=512,         # unused for resnet
        mlp_dropout=0.0,        # unused
        resnet_norm="bn",
        sgd_momentum=float(args.sgd_momentum),
        schedule=RESNET_SCHEDULE[optimizer_name],
        cosine_eta_min=float(RESNET_ETA_MIN[optimizer_name]),
        warmup_epochs=int(args.warmup_epochs),
        warmup_lr_start=float(args.warmup_lr_start),
        save_every=int(args.save_every),
        save_last=True,
    )


def _default_hparams_off(opt_name: str) -> Tuple[float, float]:
    if opt_name == "adamw":
        return 3e-4, 5e-3
    return 1e-1, 5e-4  # sgd


# --------------------------------------------------------------------------- #
# Per-dataset experiment                                                        #
# --------------------------------------------------------------------------- #

def run_dataset(
    *,
    dataset_name: str,
    args: argparse.Namespace,
    out_root: Path,
) -> Dict[str, Any]:
    seeds: List[int] = [int(s.strip()) for s in str(args.seeds).split(",") if s.strip()]

    setup = prepare_experiment_setup(
        dataset_name=dataset_name,
        data_root=args.data_root,
        split_seed=int(args.split_seed),
        val_size=int(args.val_size),
    )
    test_loader = _make_test_loader(
        setup.test_ds,
        batch_size=int(args.batch_size),
        num_workers=int(args.num_workers),
    )

    exp_dir    = out_root / dataset_name
    tuning_dir = exp_dir / "tuning"
    final_dir  = exp_dir / "final"
    exp_dir.mkdir(parents=True, exist_ok=True)

    manifest: Dict[str, Any] = {
        "dataset":      dataset_name,
        "arch":         "resnet18",
        "norm":         "bn",
        "seeds":        seeds,
        "tuning_seed":  int(args.tuning_seed),
        "epochs":       int(args.epochs),
        "tuning":       {},
        "final":        {opt: {} for opt in OPTIMIZERS},
    }

    full_indices  = setup.train_indices + setup.val_indices
    train_full_ds = Subset(setup.train_full, full_indices)
    monitor_ds    = Subset(setup.eval_full,  full_indices)

    # ------------------------------------------------------------------ #
    # Phase 1 — Hyperparameter tuning                                     #
    # ------------------------------------------------------------------ #
    best_hparams: Dict[str, Dict[str, float]] = {}

    for opt_name in OPTIMIZERS:
        summary_path = tuning_dir / opt_name / "tuning_summary.json"

        if args.tune_mode.lower() == "off":
            best_lr, best_wd = _default_hparams_off(opt_name)
            tuning_summary: Dict[str, Any] = {
                "selection_rule": "manual defaults (tune_mode=off)",
                "best_lr": best_lr,
                "best_weight_decay": best_wd,
                "trials": [],
            }
            save_json(summary_path, tuning_summary)

        elif summary_path.exists():
            print(f"[skip] Tuning for {opt_name} on {dataset_name} already done.")
            with summary_path.open(encoding="utf-8") as f:
                tuning_summary = json.load(f)

        else:
            template = _template_cfg(
                dataset_name=dataset_name, optimizer_name=opt_name, args=args,
            )
            template.model_seed = int(args.tuning_seed)
            grid = _resnet18_hparam_grid(opt_name, args.tune_mode)
            tuning_summary = tune_grid(
                out_dir=tuning_dir / opt_name,
                setup=setup,
                template_cfg=template,
                lr_grid=grid["lr"],
                wd_grid=grid["wd"],
                run_prefix=f"resnet18_{dataset_name}_{opt_name}_tune",
            )

        best_hparams[opt_name] = {
            "lr": float(tuning_summary["best_lr"]),
            "wd": float(tuning_summary["best_weight_decay"]),
        }
        manifest["tuning"][opt_name] = {
            "best_lr":           best_hparams[opt_name]["lr"],
            "best_weight_decay": best_hparams[opt_name]["wd"],
            "summary_path":      str(summary_path),
        }
        print(
            f"[tune] {dataset_name} / {opt_name}: "
            f"lr={best_hparams[opt_name]['lr']:.2e}, "
            f"wd={best_hparams[opt_name]['wd']:.2e}"
        )

    # ------------------------------------------------------------------ #
    # Phase 2 — Train N seeds per optimizer                               #
    # ------------------------------------------------------------------ #
    for opt_name in OPTIMIZERS:
        for seed in seeds:
            cfg = _template_cfg(
                dataset_name=dataset_name, optimizer_name=opt_name, args=args,
            )
            cfg.lr           = best_hparams[opt_name]["lr"]
            cfg.weight_decay = best_hparams[opt_name]["wd"]
            cfg.model_seed   = seed
            cfg.save_last    = True
            # save_every is already set from args

            run_name = f"resnet18_{dataset_name}_{opt_name}_seed{seed}"
            run_dir  = final_dir / opt_name / f"seed_{seed}"

            torch.cuda.empty_cache()
            print(f"\n[train] {dataset_name} / {opt_name} / seed={seed}")
            summary = train_run(
                run_dir=run_dir,
                run_name=run_name,
                train_ds=train_full_ds,
                val_ds=monitor_ds,
                test_loader=test_loader,
                cfg=cfg,
            )
            manifest["final"][opt_name][f"seed_{seed}"] = {
                "run_dir":      str(run_dir),
                "run_name":     run_name,
                "ckpt_final":   str(run_dir / f"{run_name}_final.pth"),
                "ckpt_best":    str(run_dir / f"{run_name}_best.pth"),
                "lr":           cfg.lr,
                "weight_decay": cfg.weight_decay,
                "summary":      summary,
            }

    save_json(exp_dir / "manifest.json", manifest)
    print(f"\n[done] {dataset_name} — manifest saved to {exp_dir / 'manifest.json'}")
    return manifest


# --------------------------------------------------------------------------- #
# CLI                                                                          #
# --------------------------------------------------------------------------- #

def main() -> None:
    p = argparse.ArgumentParser(
        description=(
            "Train ResNet18 (BatchNorm) on CIFAR-10/100 for CRH analysis.\n"
            "Phase 1: tune lr/wd per optimizer. "
            "Phase 2: train N seeds with best hparams."
        )
    )
    p.add_argument("--dataset",  type=str, action="append", dest="datasets",
                   choices=["CIFAR10", "CIFAR100"],
                   help="Dataset to run (can be repeated for both). Default: CIFAR10.")
    p.add_argument("--out-dir",   type=str, default="./CRH_resnet_out")
    p.add_argument("--data-root", type=str, default="./data")

    p.add_argument("--epochs",      type=int,   default=DEFAULT_EPOCHS)
    p.add_argument("--batch-size",  type=int,   default=DEFAULT_BATCH_SIZE)
    p.add_argument("--val-size",    type=int,   default=DEFAULT_VAL_SIZE)
    p.add_argument("--num-workers", type=int,   default=4)
    p.add_argument("--save-every",  type=int,   default=10,
                   help="Save a checkpoint every N epochs (for CRH tracking). Default: 10.")

    p.add_argument("--seeds",       type=str,   default="0,1,2",
                   help="Comma-separated model seeds. Default: 0,1,2.")
    p.add_argument("--tuning-seed", type=int,   default=99,
                   help="Seed used during tuning (separate from main seeds). Default: 99.")
    p.add_argument("--split-seed",  type=int,   default=50)

    p.add_argument("--sgd-momentum",    type=float, default=0.9)
    p.add_argument("--warmup-epochs",   type=int,   default=1)
    p.add_argument("--warmup-lr-start", type=float, default=1e-6)

    p.add_argument("--tune-mode", type=str, default="quick",
                   choices=["off", "quick", "full"])
    args = p.parse_args()

    # Default to CIFAR10 if nothing specified.
    if not args.datasets:
        args.datasets = ["CIFAR10"]

    out_root = Path(args.out_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    all_manifests: Dict[str, Any] = {}
    for dataset_name in args.datasets:
        all_manifests[dataset_name] = run_dataset(
            dataset_name=dataset_name,
            args=args,
            out_root=out_root,
        )

    save_json(out_root / "manifest_all.json", all_manifests)
    print(f"\nAll done. Full manifest: {out_root / 'manifest_all.json'}")


if __name__ == "__main__":
    main()
