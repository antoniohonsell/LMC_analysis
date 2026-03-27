#!/usr/bin/env python3
"""
run_muon_resnet_lmc.py

Trains ResNet-20 (LayerNorm, width=16) on CIFAR-10 with the Muon optimizer,
tunes hyperparameters, then checks Linear Mode Connectivity (LMC) against:
  - Other Muon seeds        (same-optimizer pairs)
  - SGD checkpoints         (cross-optimizer, loaded from --sgd-adam-out-dir)
  - AdamW checkpoints       (cross-optimizer, loaded from --sgd-adam-out-dir)

Usage:
  # Muon only:
  PYTHONPATH=~/LMC_analysis nohup python SGDvsAdam/run_muon_resnet_lmc.py --out-dir ./muon_resnet_out > muon_resnet.log 2>&1 &

  # With cross-optimizer comparison:
  PYTHONPATH=~/LMC_analysis nohup python SGDvsAdam/run_muon_resnet_lmc.py --out-dir ./muon_resnet_out --sgd-adam-out-dir ./SGDvsAdam_out > muon_resnet.log 2>&1 &
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from itertools import combinations
from pathlib import Path
from typing import Any, Dict, List, Optional

from torch.utils.data import DataLoader, Subset

_HERE = Path(__file__).resolve().parent
_SRC  = _HERE.parent.parent / "src"
for _p in (str(_SRC), str(_HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from common import (  # type: ignore
    TrainConfig,
    datasets,
    default_hparam_grid,
    prepare_experiment_setup,
    save_json,
    train_run,
    tune_grid,
    utils,
)
from lmc_weight_matching_interp import run_weight_matching_interp  # type: ignore
from plot_interp_local import load_interp_results, plot_acc, plot_loss  # type: ignore


ARCH      = "resnet20"
DATASET   = "CIFAR10"
OPTIMIZER = "muon"

DEFAULTS = {
    "epochs":       200,
    "batch_size":   256,
    "val_size":     5000,
    "schedule":     "cosine",
    "cosine_eta_min": 1e-5,
    "early_stopping_patience": 15,   # patience is longer for ResNet
    "resnet_width":    16,
    "resnet_norm":     "flax_ln",
    "resnet_shortcut": "C",
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_test_loader(test_ds, batch_size: int, num_workers: int) -> DataLoader:
    device = utils.get_device()
    return DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
    )


def rerender_named_plots(results_path, out_dir, tag):
    res = load_interp_results(str(results_path))
    out_dir.mkdir(parents=True, exist_ok=True)
    base  = f"cifar10_resnet20_{tag}"
    title = f"CIFAR-10 — ResNet-20: {tag}"
    plot_loss(
        res["lambdas"],
        res["train_loss_naive"], res["test_loss_naive"],
        res["train_loss_perm"],  res["test_loss_perm"],
        title,
        str(out_dir / f"{base}_loss_interp.png"),
        str(out_dir / f"{base}_loss_interp.pdf"),
    )
    plot_acc(
        res["lambdas"],
        res["train_acc_naive"], res["test_acc_naive"],
        res["train_acc_perm"],  res["test_acc_perm"],
        title,
        str(out_dir / f"{base}_acc_interp.png"),
        str(out_dir / f"{base}_acc_interp.pdf"),
    )


def _template_cfg(args: argparse.Namespace) -> TrainConfig:
    return TrainConfig(
        dataset=DATASET,
        arch=ARCH,
        optimizer_name=OPTIMIZER,
        lr=0.0,
        weight_decay=0.0,
        epochs=int(args.epochs),
        batch_size=int(args.batch_size),
        num_workers=int(args.num_workers),
        model_seed=0,
        split_seed=int(args.split_seed),
        val_size=int(args.val_size),
        resnet_width=int(args.resnet_width),
        resnet_norm=str(args.resnet_norm),
        resnet_shortcut=str(args.resnet_shortcut),
        muon_momentum=float(args.muon_momentum),
        muon_ns_steps=int(args.muon_ns_steps),
        schedule=DEFAULTS["schedule"],
        cosine_eta_min=DEFAULTS["cosine_eta_min"],
        save_every=int(args.save_every),
        save_last=True,
        early_stopping_patience=DEFAULTS["early_stopping_patience"],
    )


def _run_lmc_pair(*, ckpt_a, ckpt_b, pair_dir, figs_dir, pair_tag, args, wandb_lmc_run=None):
    if (pair_dir / "interp_results.pt").exists():
        print(f"[skip] LMC '{pair_tag}' already computed.")
        return {"results_pt": str(pair_dir / "interp_results.pt"), "skipped": True}

    clean_tag = pair_tag.replace("_vs_", " vs ").replace("_seed", " s").replace("sgd", "SGD").replace("adamw", "AdamW").replace("muon", "Muon")
    plot_title = f"CIFAR-10 — ResNet-20: {clean_tag}"

    results = run_weight_matching_interp(
        arch=ARCH,
        dataset_name=DATASET,
        ckpt_a=ckpt_a,
        ckpt_b=ckpt_b,
        out_dir=str(pair_dir),
        data_root=args.data_root,
        batch_size=int(args.batch_size),
        num_workers=int(args.num_workers),
        eval_samples=0,
        num_lambdas=int(args.num_lambdas),
        seed=int(args.split_seed),
        max_iter=int(args.max_iter),
        width_multiplier=int(args.resnet_width),
        shortcut_option=str(args.resnet_shortcut),
        norm=str(args.resnet_norm),
        silent=bool(args.silent),
        bn_reset_batches=0,   # LayerNorm — no BN recalibration needed
        plot_title=plot_title,
    )
    rerender_named_plots(pair_dir / "interp_results.pt", figs_dir, clean_tag)

    if wandb_lmc_run is not None:
        try:
            import wandb  # type: ignore
            log_dict: Dict[str, Any] = {}
            for kind, key in [("loss_interp", f"lmc/{pair_tag}/loss"), ("acc_interp", f"lmc/{pair_tag}/acc")]:
                png = figs_dir / f"cifar10_resnet20_{clean_tag}_{kind}.png"
                if png.exists():
                    log_dict[key] = wandb.Image(str(png))
            if log_dict:
                wandb_lmc_run.log(log_dict)
        except Exception:
            pass

    mid = len(results["test_loss_naive"]) // 2
    return {
        "results_pt": str(pair_dir / "interp_results.pt"),
        "summary": {
            "mid_test_loss_naive": float(results["test_loss_naive"][mid]),
            "mid_test_loss_perm":  float(results["test_loss_perm"][mid]),
        },
    }


def _load_existing_ckpts(sgd_adam_out_dir: Path, seeds: List[int]) -> Dict[str, Dict[int, str]]:
    manifest_path = sgd_adam_out_dir / f"{ARCH}_{DATASET.lower()}" / "manifest.json"
    if not manifest_path.exists():
        print(f"[warn] No manifest at {manifest_path} — skipping cross-optimizer LMC.")
        return {}
    with manifest_path.open() as f:
        manifest = json.load(f)
    result: Dict[str, Dict[int, str]] = {}
    for opt_name in ("sgd", "adamw"):
        result[opt_name] = {}
        for seed in seeds:
            try:
                ckpt = manifest["final"][opt_name][f"seed_{seed}"]["ckpt"]
                if Path(ckpt).exists():
                    result[opt_name][seed] = ckpt
                else:
                    print(f"[warn] Checkpoint not found: {ckpt}")
            except (KeyError, TypeError):
                print(f"[warn] No {opt_name} seed {seed} in manifest.")
    return result


# ---------------------------------------------------------------------------
# Main experiment
# ---------------------------------------------------------------------------

def run_muon_resnet_experiment(args: argparse.Namespace) -> Dict[str, Any]:
    seeds: List[int] = [int(s.strip()) for s in str(args.seeds).split(",") if s.strip()]

    out_root   = Path(args.out_dir)
    exp_dir    = out_root / f"{ARCH}_{DATASET.lower()}"
    tuning_dir = exp_dir / "tuning"
    final_dir  = exp_dir / "final"
    lmc_dir    = exp_dir / "lmc"
    figs_dir   = exp_dir / "figs"
    exp_dir.mkdir(parents=True, exist_ok=True)

    setup = prepare_experiment_setup(
        dataset_name=DATASET,
        data_root=args.data_root,
        split_seed=int(args.split_seed),
        val_size=int(args.val_size),
    )
    test_loader    = _make_test_loader(setup.test_ds, int(args.batch_size), int(args.num_workers))
    full_indices   = setup.train_indices + setup.val_indices
    train_full_ds  = Subset(setup.train_full, full_indices)

    manifest: Dict[str, Any] = {
        "arch": ARCH, "dataset": DATASET, "seeds": seeds,
        "optimizer": OPTIMIZER,
        "resnet_width": int(args.resnet_width),
        "resnet_norm":  str(args.resnet_norm),
        "tuning": {},
        "final": {},
        "lmc": {"muon_vs_muon": {}, "muon_vs_sgd": {}, "muon_vs_adamw": {}},
    }

    # ------------------------------------------------------------------ #
    # Phase 1 — Hyperparameter tuning                                     #
    # ------------------------------------------------------------------ #
    tuning_summary_path = tuning_dir / OPTIMIZER / "tuning_summary.json"

    if args.tune_mode.lower() == "off":
        tuning_summary: Dict[str, Any] = {
            "best_lr": float(args.muon_lr_default),
            "best_weight_decay": 0.0,
            "selection_rule": "manual default (tune_mode=off)",
            "trials": [],
        }
        save_json(tuning_summary_path, tuning_summary)

    elif tuning_summary_path.exists():
        print(f"[skip] Tuning for {OPTIMIZER} already done.")
        with tuning_summary_path.open() as f:
            tuning_summary = json.load(f)

    else:
        template = _template_cfg(args)
        template.model_seed = int(args.tuning_seed)
        grid = default_hparam_grid(arch=ARCH, optimizer_name=OPTIMIZER, mode=args.tune_mode)
        tuning_summary = tune_grid(
            out_dir=tuning_dir / OPTIMIZER,
            setup=setup,
            template_cfg=template,
            lr_grid=grid["lr"],
            wd_grid=grid["wd"],
            run_prefix=f"{ARCH}_{DATASET}_{OPTIMIZER}_tune",
        )
        for trial_dir in (tuning_dir / OPTIMIZER).iterdir():
            if trial_dir.is_dir():
                shutil.rmtree(trial_dir)

    best_lr = float(tuning_summary["best_lr"])
    best_wd = float(tuning_summary["best_weight_decay"])
    manifest["tuning"] = {"best_lr": best_lr, "best_weight_decay": best_wd}
    print(f"[tuning] Best lr={best_lr}, wd={best_wd}")

    # ------------------------------------------------------------------ #
    # Phase 2 — Train N Muon seeds                                        #
    # ------------------------------------------------------------------ #
    muon_ckpts: Dict[int, str] = {}

    for seed in seeds:
        cfg = _template_cfg(args)
        cfg.lr           = best_lr
        cfg.weight_decay = best_wd
        cfg.model_seed   = seed
        cfg.save_every   = int(max(args.save_every, cfg.epochs))
        cfg.save_last    = True

        run_name = f"{ARCH}_{DATASET}_{OPTIMIZER}_seed{seed}"
        run_dir  = final_dir / OPTIMIZER / f"seed_{seed}"

        summary = train_run(
            run_dir=run_dir,
            run_name=run_name,
            train_ds=train_full_ds,
            val_ds=setup.test_ds,
            test_loader=test_loader,
            cfg=cfg,
            wandb_project=getattr(args, "wandb_project", None),
            wandb_entity=getattr(args, "wandb_entity", None),
            wandb_tags=[ARCH, DATASET, OPTIMIZER],
        )
        ckpt_path = str(run_dir / f"{run_name}_final.pth")
        muon_ckpts[seed] = ckpt_path
        manifest["final"][f"seed_{seed}"] = {
            "ckpt": ckpt_path,
            "lr":   cfg.lr,
            "weight_decay": cfg.weight_decay,
            "summary": summary,
        }

    # ------------------------------------------------------------------ #
    # Phase 3 — LMC                                                       #
    # ------------------------------------------------------------------ #
    if args.skip_lmc:
        save_json(exp_dir / "manifest.json", manifest)
        return manifest

    _wandb_lmc_run = None
    if getattr(args, "wandb_project", None):
        try:
            import wandb  # type: ignore
            _wandb_lmc_run = wandb.init(
                project=args.wandb_project,
                entity=getattr(args, "wandb_entity", None),
                name=f"{ARCH}_{DATASET}_{OPTIMIZER}_lmc",
                tags=[ARCH, DATASET, OPTIMIZER, "lmc"],
                reinit=True,
            )
        except ImportError:
            pass

    # Muon vs Muon
    for s1, s2 in combinations(seeds, 2):
        pair_tag = f"muon_seed{s1}_vs_seed{s2}"
        pair_dir = lmc_dir / "muon_vs_muon" / f"seed{s1}_vs_seed{s2}"
        manifest["lmc"]["muon_vs_muon"][f"seed{s1}_vs_seed{s2}"] = _run_lmc_pair(
            ckpt_a=muon_ckpts[s1], ckpt_b=muon_ckpts[s2],
            pair_dir=pair_dir, figs_dir=figs_dir, pair_tag=pair_tag,
            args=args, wandb_lmc_run=_wandb_lmc_run,
        )

    # Cross-optimizer: Muon vs SGD / AdamW
    existing: Dict[str, Dict[int, str]] = {}
    if args.sgd_adam_out_dir is not None:
        existing = _load_existing_ckpts(Path(args.sgd_adam_out_dir), seeds)

    for other_opt in ("sgd", "adamw"):
        lmc_key = f"muon_vs_{other_opt}"
        if not existing.get(other_opt):
            print(f"[skip] No {other_opt} checkpoints — skipping {lmc_key}.")
            continue
        for s_muon in seeds:
            for s_other, other_ckpt in existing[other_opt].items():
                pair_tag = f"muon_seed{s_muon}_vs_{other_opt}_seed{s_other}"
                pair_dir = lmc_dir / lmc_key / f"muon_seed{s_muon}_vs_{other_opt}_seed{s_other}"
                manifest["lmc"][lmc_key][pair_tag] = _run_lmc_pair(
                    ckpt_a=muon_ckpts[s_muon], ckpt_b=other_ckpt,
                    pair_dir=pair_dir, figs_dir=figs_dir, pair_tag=pair_tag,
                    args=args, wandb_lmc_run=_wandb_lmc_run,
                )

    if _wandb_lmc_run is not None:
        _wandb_lmc_run.finish()

    save_json(exp_dir / "manifest.json", manifest)
    return manifest


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser(
        description="Muon ResNet-20/CIFAR-10 LMC experiment."
    )
    p.add_argument("--out-dir",          type=str, default="./muon_resnet_out")
    p.add_argument("--data-root",        type=str, default="./data")
    p.add_argument("--sgd-adam-out-dir", type=str, default=None)

    p.add_argument("--epochs",     type=int,   default=DEFAULTS["epochs"])
    p.add_argument("--batch-size", type=int,   default=DEFAULTS["batch_size"])
    p.add_argument("--val-size",   type=int,   default=DEFAULTS["val_size"])
    p.add_argument("--num-workers",type=int,   default=4)

    p.add_argument("--seeds",       type=str, default="0,1,2")
    p.add_argument("--tuning-seed", type=int, default=99)
    p.add_argument("--split-seed",  type=int, default=50)
    p.add_argument("--save-every",  type=int, default=10)

    p.add_argument("--resnet-width",    type=int, default=DEFAULTS["resnet_width"])
    p.add_argument("--resnet-norm",     type=str, default=DEFAULTS["resnet_norm"],
                   choices=["bn", "ln", "flax_ln", "none"])
    p.add_argument("--resnet-shortcut", type=str, default=DEFAULTS["resnet_shortcut"],
                   choices=["A", "B", "C"])

    p.add_argument("--muon-momentum",   type=float, default=0.95)
    p.add_argument("--muon-ns-steps",   type=int,   default=5)
    p.add_argument("--muon-lr-default", type=float, default=0.003)

    p.add_argument("--tune-mode", type=str, default="quick", choices=["off", "quick", "full"])
    p.add_argument("--skip-lmc",  action="store_true")
    p.add_argument("--silent",    action="store_true")

    p.add_argument("--num-lambdas", type=int, default=25)
    p.add_argument("--max-iter",    type=int, default=100)

    p.add_argument("--wandb-project", type=str, default=None)
    p.add_argument("--wandb-entity",  type=str, default=None)

    args = p.parse_args()

    manifest = run_muon_resnet_experiment(args)
    print(json.dumps(manifest, indent=2, default=str))


if __name__ == "__main__":
    main()
