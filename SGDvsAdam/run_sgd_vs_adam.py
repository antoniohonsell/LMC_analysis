from __future__ import annotations

import argparse
import json
import sys
from itertools import combinations
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from torch.utils.data import DataLoader, Subset

THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

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


OPTIMIZERS = ("sgd", "adamw")

DEFAULTS: Dict[str, Dict[str, Any]] = {
    "mlp": {
        "dataset": "FASHIONMNIST",
        "epochs": 30,
        "batch_size": 256,
        "val_size": 5000,
        "schedule": {"sgd": "none", "adamw": "none"},
        "cosine_eta_min": {"sgd": 0.0, "adamw": 0.0},
    },
    "resnet20": {
        "dataset": "CIFAR10",
        "epochs": 200,
        "batch_size": 256,
        "val_size": 5000,
        "schedule": {"sgd": "warmup_cosine", "adamw": "cosine"},
        "cosine_eta_min": {"sgd": 0.0, "adamw": 1e-5},
    },
    "resnet50": {
        "dataset": "CIFAR100",
        "epochs": 200,
        "batch_size": 128,
        "val_size": 5000,
        "schedule": {"sgd": "warmup_cosine", "adamw": "cosine"},
        "cosine_eta_min": {"sgd": 0.0, "adamw": 1e-5},
    },
}


# --------------------------------------------------------------------------- #
# Helpers                                                                      #
# --------------------------------------------------------------------------- #

def rerender_named_plots(
    results_path: Path,
    out_dir: Path,
    dataset: str,
    arch: str,
    tag: str,
    width: Optional[int],
) -> None:
    res = load_interp_results(str(results_path))
    out_dir.mkdir(parents=True, exist_ok=True)
    base = "_".join(
        [x for x in [dataset.lower(), arch.lower(), f"w{width}" if width is not None else None, tag] if x]
    )
    title_parts = [dataset, arch]
    if width is not None:
        title_parts.append(f"w={width}")
    if tag:
        title_parts.append(tag)
    title = " — ".join(title_parts)

    plot_loss(
        res["lambdas"],
        res["train_loss_naive"],
        res["test_loss_naive"],
        res["train_loss_perm"],
        res["test_loss_perm"],
        title,
        str(out_dir / f"{base}_loss_interp.png"),
        str(out_dir / f"{base}_loss_interp.pdf"),
    )
    plot_acc(
        res["lambdas"],
        res["train_acc_naive"],
        res["test_acc_naive"],
        res["train_acc_perm"],
        res["test_acc_perm"],
        title,
        str(out_dir / f"{base}_acc_interp.png"),
        str(out_dir / f"{base}_acc_interp.pdf"),
    )


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
    arch: str,
    dataset_name: str,
    optimizer_name: str,
    args: argparse.Namespace,
) -> TrainConfig:
    defaults = DEFAULTS[arch]
    epochs = int(args.epochs_mlp if arch == "mlp" else args.epochs_resnet)
    batch_size = int(args.batch_size_mlp if arch == "mlp" else args.batch_size_resnet50 if arch == "resnet50" else args.batch_size_resnet)
    return TrainConfig(
        dataset=dataset_name,
        arch=arch,
        optimizer_name=optimizer_name,
        lr=0.0,           # overridden after tuning
        weight_decay=0.0,  # overridden after tuning
        epochs=epochs,
        batch_size=batch_size,
        num_workers=int(args.num_workers),
        model_seed=0,      # overridden per seed
        split_seed=int(args.split_seed),
        val_size=int(args.val_size_mlp if arch == "mlp" else args.val_size_resnet),
        mlp_hidden=int(args.mlp_hidden),
        mlp_dropout=float(args.mlp_dropout),
        resnet_width=int(args.resnet_width) if arch != "resnet50" else 1,
        resnet_norm=str(args.resnet_norm) if arch != "resnet50" else "bn",
        resnet_shortcut=str(args.resnet_shortcut) if arch != "resnet50" else "C",
        sgd_momentum=float(args.sgd_momentum),
        schedule=defaults["schedule"][optimizer_name],
        cosine_eta_min=float(defaults["cosine_eta_min"][optimizer_name]),
        warmup_epochs=int(args.warmup_epochs),
        warmup_lr_start=float(args.warmup_lr_start),
        save_every=int(args.save_every),
        save_last=True,
    )


def _default_hparams_off(arch: str, opt_name: str) -> Tuple[float, float]:
    """Fallback (lr, wd) when --tune-mode off."""
    if arch == "mlp" and opt_name == "adamw":
        return 1e-3, 1e-2
    if arch == "mlp":   # sgd
        return 3e-2, 0.0
    if arch == "resnet50" and opt_name == "adamw":
        return 3e-4, 5e-3
    if arch == "resnet50":  # sgd
        return 1e-1, 5e-4
    if opt_name == "adamw":
        return 3e-4, 5e-3
    # sgd resnet20
    return 1e-1, 5e-4


def _run_lmc_pair(
    *,
    arch: str,
    dataset_name: str,
    ckpt_a: str,
    ckpt_b: str,
    pair_dir: Path,
    figs_dir: Path,
    pair_tag: str,
    width_for_plot: Optional[int],
    args: argparse.Namespace,
    wandb_lmc_run: Optional[Any] = None,
) -> Dict[str, Any]:
    """Run weight-matching LMC for one checkpoint pair. Skip if already done."""
    if (pair_dir / "interp_results.pt").exists():
        print(f"[skip] LMC '{pair_tag}' already computed.")
        return {"results_pt": str(pair_dir / "interp_results.pt"), "skipped": True}

    results = run_weight_matching_interp(
        arch=arch,
        dataset_name=dataset_name,
        ckpt_a=ckpt_a,
        ckpt_b=ckpt_b,
        out_dir=str(pair_dir),
        data_root=args.data_root,
        batch_size=int(args.batch_size_resnet50 if arch == "resnet50" else args.batch_size_resnet if arch == "resnet20" else args.batch_size_mlp),
        num_workers=int(args.num_workers),
        eval_samples=int(args.eval_samples),
        num_lambdas=int(args.num_lambdas),
        seed=int(args.split_seed),
        max_iter=int(args.max_iter),
        width_multiplier=(int(args.resnet_width) if arch == "resnet20" else None),
        shortcut_option=(str(args.resnet_shortcut) if arch == "resnet20" else None),
        norm=(str(args.resnet_norm) if arch == "resnet20" else None),
        silent=bool(args.silent),
        bn_reset_batches=int(args.bn_reset_batches),
    )
    rerender_named_plots(
        results_path=pair_dir / "interp_results.pt",
        out_dir=figs_dir,
        dataset=dataset_name,
        arch=arch,
        tag=pair_tag,
        width=width_for_plot,
    )

    if wandb_lmc_run is not None:
        try:
            import wandb  # type: ignore
            base = "_".join(
                [x for x in [dataset_name.lower(), arch.lower(),
                              f"w{width_for_plot}" if width_for_plot is not None else None,
                              pair_tag] if x]
            )
            log_dict: Dict[str, Any] = {}
            for kind, key in [("loss_interp", f"lmc/{pair_tag}/loss"), ("acc_interp", f"lmc/{pair_tag}/acc")]:
                png = figs_dir / f"{base}_{kind}.png"
                if png.exists():
                    log_dict[key] = wandb.Image(str(png))
            if log_dict:
                wandb_lmc_run.log(log_dict)
        except Exception:
            pass

    mid = len(results["test_loss_naive"]) // 2
    return {
        "results_pt": str(pair_dir / "interp_results.pt"),
        "results_json": str(pair_dir / "interp_results.json"),
        "summary": {
            "mid_test_loss_naive": float(results["test_loss_naive"][mid]),
            "mid_test_loss_perm": float(results["test_loss_perm"][mid]),
        },
    }


# --------------------------------------------------------------------------- #
# Main experiment                                                              #
# --------------------------------------------------------------------------- #

def run_one_experiment(
    *,
    arch: str,
    dataset_name: str,
    args: argparse.Namespace,
    out_root: Path,
) -> Dict[str, Any]:
    seeds: List[int] = [int(s.strip()) for s in str(args.seeds).split(",") if s.strip()]

    setup = prepare_experiment_setup(
        dataset_name=dataset_name,
        data_root=args.data_root,
        split_seed=int(args.split_seed),
        val_size=int(args.val_size_mlp if arch == "mlp" else args.val_size_resnet),
    )
    test_loader = _make_test_loader(
        setup.test_ds,
        batch_size=int(args.batch_size_mlp if arch == "mlp" else args.batch_size_resnet50 if arch == "resnet50" else args.batch_size_resnet),
        num_workers=int(args.num_workers),
    )

    exp_dir   = out_root / f"{arch}_{dataset_name.lower()}"
    tuning_dir = exp_dir / "tuning"
    final_dir  = exp_dir / "final"
    lmc_dir    = exp_dir / "lmc"
    figs_dir   = exp_dir / "figs"
    exp_dir.mkdir(parents=True, exist_ok=True)

    manifest: Dict[str, Any] = {
        "arch": arch,
        "dataset": dataset_name,
        "seeds": seeds,
        "tuning_seed": int(args.tuning_seed),
        "optimizers": list(OPTIMIZERS),
        "selection_rule": "max val accuracy, tie-break min val loss",
        "tuning": {},
        "final": {opt: {} for opt in OPTIMIZERS},
        "lmc": {"same_optimizer": {opt: {} for opt in OPTIMIZERS}, "cross_optimizer": {}},
    }

    # Use the full train+val for final training; keep tune split separate.
    full_indices = setup.train_indices + setup.val_indices
    train_full_ds = Subset(setup.train_full, full_indices)
    monitor_ds    = Subset(setup.eval_full,  full_indices)

    # ------------------------------------------------------------------ #
    # Phase 1 — Hyperparameter tuning (single tuning seed per optimizer) #
    # ------------------------------------------------------------------ #
    best_hparams: Dict[str, Dict[str, float]] = {}
    for opt_name in OPTIMIZERS:
        tuning_summary_path = tuning_dir / opt_name / "tuning_summary.json"

        if args.tune_mode.lower() == "off":
            best_lr, best_wd = _default_hparams_off(arch, opt_name)
            tuning_summary: Dict[str, Any] = {
                "selection_rule": "manual defaults (tune_mode=off)",
                "best_lr": best_lr,
                "best_weight_decay": best_wd,
                "trials": [],
            }
            save_json(tuning_summary_path, tuning_summary)

        elif tuning_summary_path.exists():
            # Resume: tuning already done on a previous run.
            print(f"[skip] Tuning for {opt_name} already done — loading saved summary.")
            with tuning_summary_path.open(encoding="utf-8") as f:
                tuning_summary = json.load(f)

        else:
            template = _template_cfg(
                arch=arch, dataset_name=dataset_name,
                optimizer_name=opt_name, args=args,
            )
            template.model_seed = int(args.tuning_seed)
            grid = default_hparam_grid(arch=arch, optimizer_name=opt_name, mode=args.tune_mode)
            tuning_summary = tune_grid(
                out_dir=tuning_dir / opt_name,
                setup=setup,
                template_cfg=template,
                lr_grid=grid["lr"],
                wd_grid=grid["wd"],
                run_prefix=f"{arch}_{dataset_name}_{opt_name}_tune",
            )

        best_hparams[opt_name] = {
            "lr": float(tuning_summary["best_lr"]),
            "wd": float(tuning_summary["best_weight_decay"]),
        }
        manifest["tuning"][opt_name] = {
            "best_lr": best_hparams[opt_name]["lr"],
            "best_weight_decay": best_hparams[opt_name]["wd"],
            "summary_path": str(tuning_summary_path),
        }

    # ------------------------------------------------------------------ #
    # Phase 2 — Train N seeds per optimizer with best hparams            #
    # ------------------------------------------------------------------ #
    # final_ckpts[opt_name][seed] = path to _final.pth
    final_ckpts: Dict[str, Dict[int, str]] = {opt: {} for opt in OPTIMIZERS}

    for opt_name in OPTIMIZERS:
        for seed in seeds:
            cfg = _template_cfg(
                arch=arch, dataset_name=dataset_name,
                optimizer_name=opt_name, args=args,
            )
            cfg.lr           = best_hparams[opt_name]["lr"]
            cfg.weight_decay = best_hparams[opt_name]["wd"]
            cfg.model_seed   = seed
            cfg.save_every   = int(max(args.save_every, cfg.epochs))
            cfg.save_last    = True

            run_name = f"{arch}_{dataset_name}_{opt_name}_seed{seed}"
            run_dir  = final_dir / opt_name / f"seed_{seed}"

            summary = train_run(
                run_dir=run_dir,
                run_name=run_name,
                train_ds=train_full_ds,
                val_ds=monitor_ds,
                test_loader=test_loader,
                cfg=cfg,
                wandb_project=getattr(args, "wandb_project", None),
                wandb_entity=getattr(args, "wandb_entity", None),
                wandb_tags=[arch, dataset_name, opt_name],
            )
            ckpt_path = str(run_dir / f"{run_name}_final.pth")
            final_ckpts[opt_name][seed] = ckpt_path
            manifest["final"][opt_name][f"seed_{seed}"] = {
                "ckpt":         ckpt_path,
                "best_ckpt":    str(run_dir / f"{run_name}_best.pth"),
                "lr":           cfg.lr,
                "weight_decay": cfg.weight_decay,
                "summary":      summary,
            }

    # ------------------------------------------------------------------ #
    # Phase 3 — LMC for all pairs                                        #
    # ------------------------------------------------------------------ #
    if not args.skip_lmc:
        width_for_plot = int(args.resnet_width) if arch == "resnet20" else None
        _ = width_for_plot  # used below

        _wandb_lmc_run = None
        if getattr(args, "wandb_project", None):
            try:
                import wandb  # type: ignore
                _wandb_lmc_run = wandb.init(
                    project=args.wandb_project,
                    entity=getattr(args, "wandb_entity", None),
                    name=f"{arch}_{dataset_name}_lmc",
                    tags=[arch, dataset_name, "lmc"],
                    reinit=True,
                )
            except ImportError:
                pass

        # --- Same-optimizer pairs (C(n_seeds, 2) per optimizer) ---
        for opt_name in OPTIMIZERS:
            for s1, s2 in combinations(seeds, 2):
                pair_tag = f"{opt_name}_seed{s1}_vs_seed{s2}"
                pair_dir = lmc_dir / "same_optimizer" / opt_name / f"seed{s1}_vs_seed{s2}"
                manifest["lmc"]["same_optimizer"][opt_name][f"seed{s1}_vs_seed{s2}"] = _run_lmc_pair(
                    arch=arch,
                    dataset_name=dataset_name,
                    ckpt_a=final_ckpts[opt_name][s1],
                    ckpt_b=final_ckpts[opt_name][s2],
                    pair_dir=pair_dir,
                    figs_dir=figs_dir,
                    pair_tag=pair_tag,
                    width_for_plot=width_for_plot,
                    args=args,
                    wandb_lmc_run=_wandb_lmc_run,
                )

        # --- Cross-optimizer pairs (all SGD seeds × all AdamW seeds) ---
        for s_sgd in seeds:
            for s_adamw in seeds:
                pair_tag = f"sgd_seed{s_sgd}_vs_adamw_seed{s_adamw}"
                pair_dir = lmc_dir / "cross_optimizer" / f"sgd_seed{s_sgd}_vs_adamw_seed{s_adamw}"
                manifest["lmc"]["cross_optimizer"][pair_tag] = _run_lmc_pair(
                    arch=arch,
                    dataset_name=dataset_name,
                    ckpt_a=final_ckpts["sgd"][s_sgd],
                    ckpt_b=final_ckpts["adamw"][s_adamw],
                    pair_dir=pair_dir,
                    figs_dir=figs_dir,
                    pair_tag=pair_tag,
                    width_for_plot=width_for_plot,
                    args=args,
                    wandb_lmc_run=_wandb_lmc_run,
                )

        if _wandb_lmc_run is not None:
            _wandb_lmc_run.finish()

    save_json(exp_dir / "manifest.json", manifest)
    return manifest


# --------------------------------------------------------------------------- #
# CLI                                                                          #
# --------------------------------------------------------------------------- #

def main() -> None:
    p = argparse.ArgumentParser(
        description=(
            "SGD vs AdamW experiment:\n"
            "  1. Tune lr/wd per optimizer (single tuning seed).\n"
            "  2. Train N seeds per optimizer with best hparams.\n"
            "  3. Compute LMC for all same-optimizer and cross-optimizer pairs."
        )
    )
    p.add_argument("--out-dir",    type=str, default="./SGDvsAdam_out")
    p.add_argument("--data-root",  type=str, default="./data")
    p.add_argument("--which",      type=str, default="both", choices=["mlp", "resnet20", "resnet50", "both"])

    p.add_argument("--mlp-dataset",    type=str, default=DEFAULTS["mlp"]["dataset"],
                   choices=["MNIST", "FASHIONMNIST", "CIFAR10"])
    p.add_argument("--resnet-dataset", type=str, default=DEFAULTS["resnet20"]["dataset"],
                   choices=["CIFAR10", "CIFAR100"])

    p.add_argument("--epochs-mlp",       type=int,   default=DEFAULTS["mlp"]["epochs"])
    p.add_argument("--epochs-resnet",    type=int,   default=DEFAULTS["resnet20"]["epochs"])
    p.add_argument("--batch-size-mlp",   type=int,   default=DEFAULTS["mlp"]["batch_size"])
    p.add_argument("--batch-size-resnet",type=int,   default=DEFAULTS["resnet20"]["batch_size"])
    p.add_argument("--batch-size-resnet50", type=int, default=DEFAULTS["resnet50"]["batch_size"])
    p.add_argument("--val-size-mlp",     type=int,   default=DEFAULTS["mlp"]["val_size"])
    p.add_argument("--val-size-resnet",  type=int,   default=DEFAULTS["resnet20"]["val_size"])
    p.add_argument("--num-workers",      type=int,   default=4)

    p.add_argument("--seeds",       type=str, default="0,1,2",
                   help="Comma-separated model seeds for the main training runs (default: 0,1,2).")
    p.add_argument("--tuning-seed", type=int, default=99,
                   help="Model seed used during hyperparameter tuning (default: 99, kept separate from main seeds).")
    p.add_argument("--split-seed",  type=int, default=50)
    p.add_argument("--save-every",  type=int, default=10)

    p.add_argument("--mlp-hidden",  type=int,   default=512)
    p.add_argument("--mlp-dropout", type=float, default=0.0)

    p.add_argument("--resnet-width",    type=int, default=16)
    p.add_argument("--resnet-norm",     type=str, default="flax_ln",
                   choices=["bn", "ln", "flax_ln", "none"])
    p.add_argument("--resnet-shortcut", type=str, default="C", choices=["A", "B", "C"])
    p.add_argument("--sgd-momentum",    type=float, default=0.9)
    p.add_argument("--warmup-epochs",   type=int,   default=1)
    p.add_argument("--warmup-lr-start", type=float, default=1e-6)

    p.add_argument("--tune-mode", type=str, default="quick", choices=["off", "quick", "full"])
    p.add_argument("--skip-lmc",  action="store_true")

    p.add_argument("--wandb-project", type=str, default=None,
                   help="W&B project name. If set, final training runs are logged to Weights & Biases.")
    p.add_argument("--wandb-entity",  type=str, default=None,
                   help="W&B entity (username or team). Leave unset to use your default entity.")

    p.add_argument("--eval-samples", type=int, default=0)
    p.add_argument("--num-lambdas",  type=int, default=25)
    p.add_argument("--max-iter",     type=int, default=100)
    p.add_argument("--silent",       action="store_true")
    p.add_argument("--bn-reset-batches", type=int, default=50,
                   help="Batches to recalculate BN stats after each interpolation. 0=disabled.")
    args = p.parse_args()

    out_root = Path(args.out_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    manifests: Dict[str, Any] = {}
    if args.which in ("mlp", "both"):
        manifests["mlp"] = run_one_experiment(
            arch="mlp",
            dataset_name=str(args.mlp_dataset),
            args=args,
            out_root=out_root,
        )
    if args.which in ("resnet20", "both"):
        manifests["resnet20"] = run_one_experiment(
            arch="resnet20",
            dataset_name=str(args.resnet_dataset),
            args=args,
            out_root=out_root,
        )
    if args.which == "resnet50":
        manifests["resnet50"] = run_one_experiment(
            arch="resnet50",
            dataset_name=str(args.resnet_dataset),
            args=args,
            out_root=out_root,
        )

    save_json(out_root / "manifest_all.json", manifests)
    print(json.dumps(manifests, indent=2))


if __name__ == "__main__":
    main()
