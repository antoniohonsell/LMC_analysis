from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

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


DEFAULTS: Dict[str, Dict[str, Any]] = {
    "mlp": {
        "dataset": "MNIST",
        "epochs": 30,
        "batch_size": 256,
        "val_size": 5000,
        "schedule": {"adam": "none", "sgd": "none"},
        "cosine_eta_min": {"adam": 0.0, "sgd": 0.0},
    },
    "resnet20": {
        "dataset": "CIFAR10",
        "epochs": 200,
        "batch_size": 256,
        "val_size": 5000,
        "schedule": {"adam": "cosine", "sgd": "warmup_cosine"},
        "cosine_eta_min": {"adam": 1e-5, "sgd": 0.0},
    },
}


def rerender_named_plots(results_path: Path, out_dir: Path, dataset: str, arch: str, tag: str, width: int | None) -> None:
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
        load_interp_results(str(results_path))["lambdas"],
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


def _make_test_loader(test_ds, batch_size: int, num_workers: int):
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
    batch_size = int(args.batch_size_mlp if arch == "mlp" else args.batch_size_resnet)
    return TrainConfig(
        dataset=dataset_name,
        arch=arch,
        optimizer_name=optimizer_name,
        lr=0.0,
        weight_decay=0.0,
        epochs=epochs,
        batch_size=batch_size,
        num_workers=int(args.num_workers),
        model_seed=int(args.seed),
        split_seed=int(args.split_seed),
        val_size=int(args.val_size_mlp if arch == "mlp" else args.val_size_resnet),
        mlp_hidden=int(args.mlp_hidden),
        mlp_dropout=float(args.mlp_dropout),
        resnet_width=int(args.resnet_width),
        resnet_norm=str(args.resnet_norm),
        resnet_shortcut=str(args.resnet_shortcut),
        sgd_momentum=float(args.sgd_momentum),
        schedule=defaults["schedule"][optimizer_name],
        cosine_eta_min=float(defaults["cosine_eta_min"][optimizer_name]),
        warmup_epochs=int(args.warmup_epochs),
        warmup_lr_start=float(args.warmup_lr_start),
        save_every=int(args.save_every),
        save_last=True,
    )


def run_one_experiment(
    *,
    arch: str,
    dataset_name: str,
    args: argparse.Namespace,
    out_root: Path,
) -> Dict[str, Any]:
    setup = prepare_experiment_setup(
        dataset_name=dataset_name,
        data_root=args.data_root,
        split_seed=int(args.split_seed),
        val_size=int(args.val_size_mlp if arch == "mlp" else args.val_size_resnet),
    )
    test_loader = _make_test_loader(
        setup.test_ds,
        batch_size=int(args.batch_size_mlp if arch == "mlp" else args.batch_size_resnet),
        num_workers=int(args.num_workers),
    )

    exp_dir = out_root / f"{arch}_{dataset_name.lower()}"
    tuning_dir = exp_dir / "tuning"
    final_dir = exp_dir / "final"
    lmc_dir = exp_dir / "lmc"
    figs_dir = exp_dir / "figs"
    exp_dir.mkdir(parents=True, exist_ok=True)

    manifest: Dict[str, Any] = {
        "arch": arch,
        "dataset": dataset_name,
        "selection_rule": "max validation accuracy, tie-break min validation loss",
        "optimizers": {},
    }

    full_indices = setup.train_indices + setup.val_indices
    train_full_ds = Subset(setup.train_full, full_indices)
    monitor_ds = Subset(setup.eval_full, full_indices)
    train_tune_ds = Subset(setup.train_full, setup.train_indices)
    val_tune_ds = Subset(setup.eval_full, setup.val_indices)

    final_ckpts: Dict[str, str] = {}
    for opt_name in ("sgd", "adam"):
        template = _template_cfg(arch=arch, dataset_name=dataset_name, optimizer_name=opt_name, args=args)

        if args.tune_mode.lower() == "off":
            if arch == "mlp" and opt_name == "adam":
                best_lr, best_wd = 1e-3, 0.0
            elif arch == "mlp":
                best_lr, best_wd = 3e-2, 0.0
            elif opt_name == "adam":
                best_lr, best_wd = 3e-4, 5e-4
            else:
                best_lr, best_wd = 1e-1, 5e-4
            tuning_summary = {
                "selection_rule": "manual defaults (tune_mode=off)",
                "best_lr": best_lr,
                "best_weight_decay": best_wd,
                "trials": [],
            }
            save_json(tuning_dir / opt_name / "tuning_summary.json", tuning_summary)
        else:
            grid = default_hparam_grid(arch=arch, optimizer_name=opt_name, mode=args.tune_mode)
            tuning_summary = tune_grid(
                out_dir=tuning_dir / opt_name,
                setup=setup,
                template_cfg=template,
                lr_grid=grid["lr"],
                wd_grid=grid["wd"],
                run_prefix=f"{arch}_{dataset_name}_{opt_name}_tune",
            )

        best_lr = float(tuning_summary["best_lr"])
        best_wd = float(tuning_summary["best_weight_decay"])

        final_cfg = _template_cfg(arch=arch, dataset_name=dataset_name, optimizer_name=opt_name, args=args)
        final_cfg.lr = best_lr
        final_cfg.weight_decay = best_wd
        final_cfg.save_every = int(max(args.save_every, final_cfg.epochs))
        final_cfg.save_last = True

        run_name = f"{arch}_{dataset_name}_{opt_name}_final"
        run_dir = final_dir / opt_name
        summary = train_run(
            run_dir=run_dir,
            run_name=run_name,
            train_ds=train_full_ds,
            val_ds=monitor_ds,
            test_loader=test_loader,
            cfg=final_cfg,
        )
        final_ckpt = str(run_dir / f"{run_name}_final.pth")
        final_ckpts[opt_name] = final_ckpt
        manifest["optimizers"][opt_name] = {
            "best_lr": best_lr,
            "best_weight_decay": best_wd,
            "tuning_summary": str(tuning_dir / opt_name / "tuning_summary.json"),
            "final_summary": summary,
            "final_ckpt": final_ckpt,
            "best_ckpt": str(run_dir / f"{run_name}_best.pth"),
        }

    if not args.skip_lmc:
        width_for_plot = int(args.resnet_width) if arch == "resnet20" else None
        results = run_weight_matching_interp(
            arch=arch,
            dataset_name=dataset_name,
            ckpt_a=final_ckpts["sgd"],
            ckpt_b=final_ckpts["adam"],
            out_dir=str(lmc_dir),
            data_root=args.data_root,
            batch_size=int(args.batch_size_resnet if arch == "resnet20" else args.batch_size_mlp),
            num_workers=int(args.num_workers),
            eval_samples=int(args.eval_samples),
            num_lambdas=int(args.num_lambdas),
            seed=int(args.seed),
            max_iter=int(args.max_iter),
            width_multiplier=(int(args.resnet_width) if arch == "resnet20" else None),
            shortcut_option=(str(args.resnet_shortcut) if arch == "resnet20" else None),
            norm=(str(args.resnet_norm) if arch == "resnet20" else None),
            silent=bool(args.silent),
        )
        rerender_named_plots(
            results_path=lmc_dir / "interp_results.pt",
            out_dir=figs_dir,
            dataset=dataset_name,
            arch=arch,
            tag="sgd_vs_adam",
            width=width_for_plot,
        )
        manifest["lmc"] = {
            "results_pt": str(lmc_dir / "interp_results.pt"),
            "results_json": str(lmc_dir / "interp_results.json"),
            "raw_loss_plot": str(lmc_dir / "interp_loss.png"),
            "raw_acc_plot": str(lmc_dir / "interp_acc.png"),
            "figs_dir": str(figs_dir),
            "summary": {
                "mid_test_loss_naive": float(results["test_loss_naive"][len(results["test_loss_naive"]) // 2]),
                "mid_test_loss_perm": float(results["test_loss_perm"][len(results["test_loss_perm"]) // 2]),
            },
        }

    save_json(exp_dir / "manifest.json", manifest)
    return manifest


def main() -> None:
    p = argparse.ArgumentParser(description="Tune/train SGD vs Adam and compute LMC curves.")
    p.add_argument("--out-dir", type=str, default="./SGDvsAdam_out")
    p.add_argument("--data-root", type=str, default="./data")
    p.add_argument("--which", type=str, default="both", choices=["mlp", "resnet20", "both"])

    p.add_argument("--mlp-dataset", type=str, default=DEFAULTS["mlp"]["dataset"], choices=["MNIST", "FASHIONMNIST", "CIFAR10"])
    p.add_argument("--resnet-dataset", type=str, default=DEFAULTS["resnet20"]["dataset"], choices=["CIFAR10", "CIFAR100"])

    p.add_argument("--epochs-mlp", type=int, default=DEFAULTS["mlp"]["epochs"])
    p.add_argument("--epochs-resnet", type=int, default=DEFAULTS["resnet20"]["epochs"])
    p.add_argument("--batch-size-mlp", type=int, default=DEFAULTS["mlp"]["batch_size"])
    p.add_argument("--batch-size-resnet", type=int, default=DEFAULTS["resnet20"]["batch_size"])
    p.add_argument("--val-size-mlp", type=int, default=DEFAULTS["mlp"]["val_size"])
    p.add_argument("--val-size-resnet", type=int, default=DEFAULTS["resnet20"]["val_size"])
    p.add_argument("--num-workers", type=int, default=4)

    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--split-seed", type=int, default=50)
    p.add_argument("--save-every", type=int, default=10)

    p.add_argument("--mlp-hidden", type=int, default=512)
    p.add_argument("--mlp-dropout", type=float, default=0.0)

    p.add_argument("--resnet-width", type=int, default=16)
    p.add_argument("--resnet-norm", type=str, default="flax_ln", choices=["bn", "ln", "flax_ln", "none"])
    p.add_argument("--resnet-shortcut", type=str, default="C", choices=["A", "B", "C"])
    p.add_argument("--sgd-momentum", type=float, default=0.9)
    p.add_argument("--warmup-epochs", type=int, default=1)
    p.add_argument("--warmup-lr-start", type=float, default=1e-6)

    p.add_argument("--tune-mode", type=str, default="quick", choices=["off", "quick", "full"])
    p.add_argument("--skip-lmc", action="store_true")

    p.add_argument("--eval-samples", type=int, default=0)
    p.add_argument("--num-lambdas", type=int, default=25)
    p.add_argument("--max-iter", type=int, default=100)
    p.add_argument("--silent", action="store_true")
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

    save_json(out_root / "manifest_all.json", manifests)
    print(json.dumps(manifests, indent=2))


if __name__ == "__main__":
    main()
