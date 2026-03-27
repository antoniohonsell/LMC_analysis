from __future__ import annotations

import csv
import json
import math
import os
import re
import random
import sys
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, Subset

THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import architectures  # type: ignore
import datasets  # type: ignore
import train_loop  # type: ignore
import utils  # type: ignore


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def seed_worker(worker_id: int) -> None:
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def save_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, sort_keys=True)


def save_history_csv(path: Path, history: Dict[str, List[float]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    n = len(history.get("train_loss", []))
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["epoch", "train_loss", "train_accuracy", "val_loss", "val_accuracy"])
        for i in range(n):
            w.writerow([
                i + 1,
                history["train_loss"][i],
                history["train_accuracy"][i],
                history["val_loss"][i],
                history["val_accuracy"][i],
            ])


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, criterion: nn.Module, device: torch.device) -> Dict[str, float]:
    model.eval()
    total = len(loader.dataset)
    loss_sum = 0.0
    correct = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss = criterion(logits, y)
        loss_sum += float(loss.item()) * x.size(0)
        pred = logits.argmax(dim=1)
        correct += int((pred == y).sum().item())
    return {"loss": loss_sum / max(1, total), "accuracy": correct / max(1, total)}


def extract_targets(ds: Any) -> List[int]:
    if isinstance(ds, Subset):
        base = extract_targets(ds.dataset)
        return [base[i] for i in ds.indices]

    if hasattr(ds, "targets"):
        t = getattr(ds, "targets")
        if torch.is_tensor(t):
            return t.detach().cpu().numpy().astype(int).tolist()
        return list(map(int, t))

    for attr in ("labels", "train_labels", "test_labels"):
        if hasattr(ds, attr):
            t = getattr(ds, attr)
            if torch.is_tensor(t):
                return t.detach().cpu().numpy().astype(int).tolist()
            return list(map(int, t))

    raise AttributeError("Could not extract targets from dataset.")


def stratified_train_val_split(
    targets: List[int],
    val_size: int,
    seed: int,
    num_classes: int,
) -> Tuple[List[int], List[int]]:
    rng = np.random.default_rng(seed)
    targets_arr = np.asarray(targets)
    all_indices = np.arange(len(targets_arr))

    per_class = {c: all_indices[targets_arr == c].tolist() for c in range(num_classes)}
    for c in range(num_classes):
        rng.shuffle(per_class[c])

    base = val_size // num_classes
    rem = val_size % num_classes
    val_counts = {c: base + (1 if c < rem else 0) for c in range(num_classes)}

    val_indices: List[int] = []
    train_indices: List[int] = []
    for c in range(num_classes):
        k = val_counts[c]
        val_indices.extend(per_class[c][:k])
        train_indices.extend(per_class[c][k:])

    rng.shuffle(val_indices)
    rng.shuffle(train_indices)
    return train_indices, val_indices


@dataclass
class WarmupCosineConfig:
    lr_start: float = 1e-6
    lr_peak: float = 1e-1
    lr_min: float = 0.0
    warmup_epochs: int = 1


class WarmupCosineSchedule:
    def __init__(self, total_steps: int, warmup_steps: int, cfg: WarmupCosineConfig):
        if total_steps <= 0:
            raise ValueError("total_steps must be > 0")
        self.total_steps = total_steps
        self.warmup_steps = max(0, warmup_steps)
        self.cfg = cfg

    def lr_at(self, step: int) -> float:
        step = int(max(0, min(step, self.total_steps - 1)))
        if self.warmup_steps > 0 and step < self.warmup_steps:
            if self.warmup_steps == 1:
                return float(self.cfg.lr_peak)
            t = step / (self.warmup_steps - 1)
            return float(self.cfg.lr_start + t * (self.cfg.lr_peak - self.cfg.lr_start))

        cos_steps = max(1, self.total_steps - self.warmup_steps)
        if cos_steps == 1:
            return float(self.cfg.lr_min)

        t = (step - self.warmup_steps) / (cos_steps - 1)
        return float(
            self.cfg.lr_min + 0.5 * (self.cfg.lr_peak - self.cfg.lr_min) * (1.0 + math.cos(math.pi * t))
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_steps": self.total_steps,
            "warmup_steps": self.warmup_steps,
            "cfg": asdict(self.cfg),
        }


class ScheduledOptimizer:
    def __init__(self, optimizer: torch.optim.Optimizer, schedule: WarmupCosineSchedule):
        self.optimizer = optimizer
        self.schedule = schedule
        self.step_num = 0
        self.last_lr: Optional[float] = None

    @property
    def param_groups(self):
        return self.optimizer.param_groups

    def zero_grad(self, set_to_none: bool = False):
        self.optimizer.zero_grad(set_to_none=set_to_none)

    def step(self, closure=None):
        lr = self.schedule.lr_at(self.step_num)
        for pg in self.optimizer.param_groups:
            pg["lr"] = lr
        self.last_lr = lr
        out = self.optimizer.step(closure=closure) if closure is not None else self.optimizer.step()
        self.step_num += 1
        return out

    def state_dict(self) -> Dict[str, Any]:
        return {
            "optimizer": self.optimizer.state_dict(),
            "step_num": self.step_num,
            "schedule": self.schedule.to_dict(),
        }

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        self.optimizer.load_state_dict(state["optimizer"])
        self.step_num = int(state.get("step_num", 0))


def build_sgd_with_param_groups(
    model: nn.Module,
    lr: float,
    weight_decay: float,
    momentum: float,
) -> torch.optim.Optimizer:
    decay: List[torch.nn.Parameter] = []
    no_decay: List[torch.nn.Parameter] = []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if p.ndim == 1 or name.endswith(".bias") or "norm" in name or "bn" in name or "ln" in name:
            no_decay.append(p)
        else:
            decay.append(p)

    return torch.optim.SGD(
        [
            {"params": decay, "weight_decay": weight_decay},
            {"params": no_decay, "weight_decay": 0.0},
        ],
        lr=lr,
        momentum=momentum,
    )


def build_model(
    *,
    arch: str,
    dataset_name: str,
    mlp_hidden: int = 512,
    mlp_dropout: float = 0.0,
    resnet_width: int = 16,
    resnet_norm: str = "flax_ln",
    resnet_shortcut: str = "C",
) -> nn.Module:
    stats = datasets.DATASET_STATS[dataset_name]
    num_classes = int(stats["num_classes"])
    in_channels = int(stats["in_channels"])
    image_size = tuple(stats["image_size"])
    input_shape = (in_channels, int(image_size[0]), int(image_size[1]))

    arch_l = arch.lower()
    if arch_l == "mlp":
        return architectures.build_model(
            "mlp",
            num_classes=num_classes,
            in_channels=in_channels,
            input_shape=input_shape,
            hidden=int(mlp_hidden),
            dropout=float(mlp_dropout),
        )
    if arch_l == "resnet20":
        return architectures.build_model(
            "resnet20",
            num_classes=num_classes,
            in_channels=in_channels,
            norm=str(resnet_norm),
            width_multiplier=int(resnet_width),
            shortcut_option=str(resnet_shortcut),
        )
    if arch_l == "resnet50":
        return architectures.build_model(
            "resnet50",
            num_classes=num_classes,
            in_channels=in_channels,
            norm="bn",  # ResNet50 always uses BatchNorm
        )
    raise ValueError(f"Unsupported arch: {arch}")


@dataclass
class TrainConfig:
    dataset: str
    arch: str
    optimizer_name: str
    lr: float
    weight_decay: float
    epochs: int
    batch_size: int
    num_workers: int
    model_seed: int
    split_seed: int
    val_size: int
    mlp_hidden: int = 512
    mlp_dropout: float = 0.0
    resnet_width: int = 16
    resnet_norm: str = "flax_ln"
    resnet_shortcut: str = "C"
    sgd_momentum: float = 0.9
    muon_momentum: float = 0.95
    muon_ns_steps: int = 5
    schedule: str = "none"  # none | cosine | warmup_cosine
    cosine_eta_min: float = 0.0
    warmup_epochs: int = 1
    warmup_lr_start: float = 1e-6
    save_every: int = 10
    save_last: bool = True


@dataclass
class TrialResult:
    lr: float
    weight_decay: float
    best_val_accuracy: float
    best_val_loss: float
    best_epoch: int
    run_dir: str


@dataclass
class ExperimentSetup:
    train_full: Dataset
    eval_full: Dataset
    test_ds: Dataset
    train_indices: List[int]
    val_indices: List[int]
    dataset_name: str


def prepare_experiment_setup(dataset_name: str, data_root: str, split_seed: int, val_size: int) -> ExperimentSetup:
    train_full, eval_full, test_ds = datasets.build_datasets(
        dataset_name,
        root=data_root,
        download=True,
        augment_train=None,
        normalize=True,
    )
    targets = extract_targets(train_full)
    num_classes = int(datasets.DATASET_STATS[dataset_name]["num_classes"])
    train_indices, val_indices = stratified_train_val_split(targets, val_size, split_seed, num_classes)
    return ExperimentSetup(
        train_full=train_full,
        eval_full=eval_full,
        test_ds=test_ds,
        train_indices=train_indices,
        val_indices=val_indices,
        dataset_name=dataset_name,
    )


def _make_optimizer_and_scheduler(model: nn.Module, cfg: TrainConfig, steps_per_epoch: int):
    opt_name = cfg.optimizer_name.lower()
    scheduler = None
    if opt_name == "adam":
        optimizer: Any = torch.optim.Adam(model.parameters(), lr=float(cfg.lr), weight_decay=float(cfg.weight_decay))
        if cfg.schedule == "cosine":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=int(cfg.epochs), eta_min=float(cfg.cosine_eta_min)
            )
        elif cfg.schedule != "none":
            raise ValueError(f"Unsupported Adam schedule: {cfg.schedule}")
        return optimizer, scheduler

    if opt_name == "adamw":
        optimizer = torch.optim.AdamW(model.parameters(), lr=float(cfg.lr), weight_decay=float(cfg.weight_decay))
        if cfg.schedule == "cosine":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=int(cfg.epochs), eta_min=float(cfg.cosine_eta_min)
            )
        elif cfg.schedule != "none":
            raise ValueError(f"Unsupported AdamW schedule: {cfg.schedule}")
        return optimizer, scheduler

    if opt_name == "sgd":
        if cfg.schedule == "warmup_cosine":
            base_optim = build_sgd_with_param_groups(
                model, lr=float(cfg.warmup_lr_start), weight_decay=float(cfg.weight_decay), momentum=float(cfg.sgd_momentum)
            )
            total_steps = int(steps_per_epoch) * int(cfg.epochs)
            warmup_steps = int(steps_per_epoch) * int(cfg.warmup_epochs)
            schedule = WarmupCosineSchedule(
                total_steps=total_steps,
                warmup_steps=warmup_steps,
                cfg=WarmupCosineConfig(
                    lr_start=float(cfg.warmup_lr_start),
                    lr_peak=float(cfg.lr),
                    lr_min=float(cfg.cosine_eta_min),
                    warmup_epochs=int(cfg.warmup_epochs),
                ),
            )
            return ScheduledOptimizer(base_optim, schedule), None

        optimizer = build_sgd_with_param_groups(
            model, lr=float(cfg.lr), weight_decay=float(cfg.weight_decay), momentum=float(cfg.sgd_momentum)
        )
        if cfg.schedule == "cosine":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=int(cfg.epochs), eta_min=float(cfg.cosine_eta_min)
            )
        elif cfg.schedule != "none":
            raise ValueError(f"Unsupported SGD schedule: {cfg.schedule}")
        return optimizer, scheduler

    if opt_name == "muon":
        from muon import build_muon  # type: ignore
        optimizer = build_muon(
            model,
            lr=float(cfg.lr),
            momentum=float(cfg.muon_momentum),
            ns_steps=int(cfg.muon_ns_steps),
            weight_decay=float(cfg.weight_decay),
        )
        if cfg.schedule == "cosine":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=int(cfg.epochs), eta_min=float(cfg.cosine_eta_min)
            )
        elif cfg.schedule != "none":
            raise ValueError(f"Unsupported Muon schedule: {cfg.schedule}")
        return optimizer, scheduler

    raise ValueError(f"Unsupported optimizer: {cfg.optimizer_name}")


def _find_resume_checkpoint(run_dir: Path, run_name: str) -> Optional[str]:
    """
    Returns the path of the latest epoch checkpoint to resume from, or None.
    Returns the string "done" if training already completed (_final.pth exists).
    """
    final = run_dir / f"{run_name}_final.pth"
    if final.exists():
        return "done"

    epoch_re = re.compile(rf"^{re.escape(run_name)}_epoch(\d+)\.pth$")
    best_epoch = -1
    best_path: Optional[str] = None
    for p in run_dir.glob(f"{run_name}_epoch*.pth"):
        m = epoch_re.match(p.name)
        if m:
            ep = int(m.group(1))
            if ep > best_epoch:
                best_epoch = ep
                best_path = str(p)
    return best_path


def train_run(
    *,
    run_dir: Path,
    run_name: str,
    train_ds: Dataset,
    val_ds: Dataset,
    test_loader: Optional[DataLoader],
    cfg: TrainConfig,
    wandb_project: Optional[str] = None,
    wandb_entity: Optional[str] = None,
    wandb_tags: Optional[List[str]] = None,
) -> Dict[str, Any]:
    run_dir.mkdir(parents=True, exist_ok=True)

    # Auto-resume: if the final checkpoint already exists, reload the saved
    # summary and return immediately without re-training.
    resume_ckpt = _find_resume_checkpoint(run_dir, run_name)
    if resume_ckpt == "done":
        summary_path = run_dir / "summary.json"
        if summary_path.exists():
            with summary_path.open(encoding="utf-8") as f:
                print(f"[skip] '{run_name}' already complete — loading saved summary.")
                return json.load(f)

    set_seed(int(cfg.model_seed))

    _wandb_run = None
    if wandb_project is not None:
        try:
            import wandb  # type: ignore
            _wandb_run = wandb.init(
                project=wandb_project,
                entity=wandb_entity,
                name=run_name,
                config=asdict(cfg),
                tags=wandb_tags or [],
                reinit=True,
            )
        except ImportError:
            print("[wandb] wandb not installed — skipping W&B logging.")

    device = utils.get_device()
    g_loader = torch.Generator().manual_seed(int(cfg.model_seed))

    train_loader = DataLoader(
        train_ds,
        batch_size=int(cfg.batch_size),
        shuffle=True,
        num_workers=int(cfg.num_workers),
        worker_init_fn=seed_worker if int(cfg.num_workers) > 0 else None,
        generator=g_loader,
        pin_memory=(device.type == "cuda"),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=int(cfg.batch_size),
        shuffle=False,
        num_workers=int(cfg.num_workers),
        pin_memory=(device.type == "cuda"),
    )

    model = build_model(
        arch=cfg.arch,
        dataset_name=cfg.dataset,
        mlp_hidden=int(cfg.mlp_hidden),
        mlp_dropout=float(cfg.mlp_dropout),
        resnet_width=int(cfg.resnet_width),
        resnet_norm=str(cfg.resnet_norm),
        resnet_shortcut=str(cfg.resnet_shortcut),
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer, scheduler = _make_optimizer_and_scheduler(model, cfg, steps_per_epoch=len(train_loader))

    t0 = time.time()
    history = train_loop.train(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=int(cfg.epochs),
        device=device,
        save_dir=str(run_dir),
        run_name=run_name,
        save_every=int(cfg.save_every),
        save_last=bool(cfg.save_last),
        resume_from=resume_ckpt,
        wandb_run=_wandb_run,
    )
    t1 = time.time()

    torch.save({"history": history}, run_dir / "history.pt")
    save_json(run_dir / "history.json", history)
    save_history_csv(run_dir / "history.csv", history)
    save_json(run_dir / "config.json", asdict(cfg))

    metrics: Dict[str, Any] = {"train_wallclock_sec": float(t1 - t0)}
    if test_loader is not None:
        final_metrics = evaluate(model, test_loader, criterion, device)
        best_metrics: Optional[Dict[str, float]] = None
        best_path = run_dir / f"{run_name}_best.pth"
        if best_path.exists():
            ckpt = torch.load(best_path, map_location=device)
            model.load_state_dict(ckpt["state_dict"])
            best_metrics = evaluate(model, test_loader, criterion, device)
        metrics.update({"final": final_metrics, "best": best_metrics})

    save_json(run_dir / "test_metrics.json", metrics)

    val_accs = [float(x) for x in history.get("val_accuracy", [])]
    val_losses = [float(x) for x in history.get("val_loss", [])]
    best_epoch_idx = int(np.nanargmax(val_accs)) if val_accs else -1
    best_val_acc = float(val_accs[best_epoch_idx]) if best_epoch_idx >= 0 else float("nan")
    best_val_loss = float(val_losses[best_epoch_idx]) if best_epoch_idx >= 0 else float("nan")

    summary = {
        "run_name": run_name,
        "best_epoch": best_epoch_idx + 1 if best_epoch_idx >= 0 else None,
        "best_val_accuracy": best_val_acc,
        "best_val_loss": best_val_loss,
        "history_last": {k: v[-1] if len(v) else None for k, v in history.items()},
        "metrics": metrics,
        "best_ckpt": str(run_dir / f"{run_name}_best.pth"),
        "final_ckpt": str(run_dir / f"{run_name}_final.pth"),
    }
    save_json(run_dir / "summary.json", summary)

    if _wandb_run is not None:
        _wandb_run.summary["best_val_accuracy"] = best_val_acc
        _wandb_run.summary["best_val_loss"] = best_val_loss
        _wandb_run.summary["best_epoch"] = summary["best_epoch"]
        if test_loader is not None and "final" in metrics and metrics["final"]:
            _wandb_run.summary["test/final_accuracy"] = metrics["final"]["accuracy"]
            _wandb_run.summary["test/final_loss"] = metrics["final"]["loss"]
        if test_loader is not None and "best" in metrics and metrics["best"]:
            _wandb_run.summary["test/best_accuracy"] = metrics["best"]["accuracy"]
            _wandb_run.summary["test/best_loss"] = metrics["best"]["loss"]
        try:
            import wandb  # type: ignore
            artifact = wandb.Artifact(name=run_name, type="model")
            best_path = run_dir / f"{run_name}_best.pth"
            final_path = run_dir / f"{run_name}_final.pth"
            if best_path.exists():
                artifact.add_file(str(best_path), name="best.pth")
            if final_path.exists():
                artifact.add_file(str(final_path), name="final.pth")
            _wandb_run.log_artifact(artifact)
        except Exception as e:
            print(f"[wandb] Failed to upload checkpoint artifact: {e}")
        _wandb_run.finish()

    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return summary


def tune_grid(
    *,
    out_dir: Path,
    setup: ExperimentSetup,
    template_cfg: TrainConfig,
    lr_grid: Sequence[float],
    wd_grid: Sequence[float],
    run_prefix: str,
) -> Dict[str, Any]:
    device = utils.get_device()
    del device  # only to trigger any backend init symmetrically

    train_ds = Subset(setup.train_full, setup.train_indices)
    val_ds = Subset(setup.eval_full, setup.val_indices)

    trials: List[TrialResult] = []
    best: Optional[TrialResult] = None

    for lr in lr_grid:
        for wd in wd_grid:
            cfg = TrainConfig(**asdict(template_cfg))
            cfg.lr = float(lr)
            cfg.weight_decay = float(wd)
            cfg.save_every = cfg.epochs  # no intermediate checkpoints during tuning
            trial_tag = f"lr{lr:.6g}_wd{wd:.6g}".replace("-", "m")
            trial_dir = out_dir / trial_tag
            trial_name = f"{run_prefix}_{trial_tag}"
            summary = train_run(
                run_dir=trial_dir,
                run_name=trial_name,
                train_ds=train_ds,
                val_ds=val_ds,
                test_loader=None,
                cfg=cfg,
            )
            tr = TrialResult(
                lr=float(lr),
                weight_decay=float(wd),
                best_val_accuracy=float(summary["best_val_accuracy"]),
                best_val_loss=float(summary["best_val_loss"]),
                best_epoch=int(summary["best_epoch"] or 0),
                run_dir=str(trial_dir),
            )
            trials.append(tr)
            if best is None:
                best = tr
            else:
                if (tr.best_val_accuracy > best.best_val_accuracy) or (
                    tr.best_val_accuracy == best.best_val_accuracy and tr.best_val_loss < best.best_val_loss
                ):
                    best = tr

    assert best is not None
    payload = {
        "selection_rule": "max val_accuracy, tie-break min val_loss",
        "best_lr": best.lr,
        "best_weight_decay": best.weight_decay,
        "best_val_accuracy": best.best_val_accuracy,
        "best_val_loss": best.best_val_loss,
        "trials": [asdict(t) for t in trials],
    }
    save_json(out_dir / "tuning_summary.json", payload)
    return payload


def default_hparam_grid(arch: str, optimizer_name: str, mode: str = "quick") -> Dict[str, Sequence[float]]:
    mode = mode.lower()
    arch = arch.lower()
    optimizer_name = optimizer_name.lower()
    if mode == "off":
        raise ValueError("No grid for mode=off")

    if arch == "mlp" and optimizer_name == "adam":
        if mode == "quick":
            return {"lr": [3e-4, 1e-3, 3e-3], "wd": [0.0, 1e-4, 1e-3]}
        return {"lr": [1e-4, 3e-4, 1e-3, 3e-3], "wd": [0.0, 1e-5, 1e-4, 1e-3]}

    if arch == "mlp" and optimizer_name == "sgd":
        if mode == "quick":
            return {"lr": [1e-2, 3e-2, 1e-1], "wd": [0.0, 1e-4, 1e-3]}
        return {"lr": [3e-3, 1e-2, 3e-2, 1e-1], "wd": [0.0, 1e-5, 1e-4, 1e-3]}

    if arch == "resnet20" and optimizer_name == "adam":
        if mode == "quick":
            return {"lr": [1e-4, 3e-4, 1e-3], "wd": [1e-4, 5e-4, 1e-3]}
        return {"lr": [1e-4, 3e-4, 1e-3, 3e-3], "wd": [1e-5, 1e-4, 5e-4, 1e-3]}

    if arch == "resnet20" and optimizer_name == "sgd":
        if mode == "quick":
            return {"lr": [3e-2, 1e-1, 2e-1], "wd": [5e-4, 1e-3, 5e-3]}
        return {"lr": [1e-2, 3e-2, 1e-1, 2e-1], "wd": [5e-4, 1e-3, 5e-3]}

    # AdamW grids: same lr ranges as Adam, but weight decay ranges are higher
    # because AdamW applies true decoupled weight decay (not L2 reg on adapted grad).
    if arch == "mlp" and optimizer_name == "adamw":
        if mode == "quick":
            return {"lr": [3e-4, 1e-3, 3e-3], "wd": [1e-4, 1e-3, 1e-2]}
        return {"lr": [1e-4, 3e-4, 1e-3, 3e-3], "wd": [1e-5, 1e-4, 1e-3, 1e-2]}

    if arch == "resnet20" and optimizer_name == "adamw":
        if mode == "quick":
            return {"lr": [1e-4, 3e-4, 1e-3], "wd": [1e-3, 5e-3, 1e-2]}
        return {"lr": [1e-4, 3e-4, 1e-3, 3e-3], "wd": [1e-4, 1e-3, 5e-3, 1e-2]}

    if arch == "resnet50" and optimizer_name == "sgd":
        if mode == "quick":
            return {"lr": [3e-2, 1e-1, 2e-1], "wd": [1e-4, 5e-4, 1e-3]}
        return {"lr": [1e-2, 3e-2, 1e-1, 2e-1], "wd": [1e-4, 5e-4, 1e-3, 5e-3]}

    if arch == "resnet50" and optimizer_name in ("adam", "adamw"):
        if mode == "quick":
            return {"lr": [1e-4, 3e-4, 1e-3], "wd": [1e-3, 5e-3, 1e-2]}
        return {"lr": [1e-4, 3e-4, 1e-3, 3e-3], "wd": [1e-4, 1e-3, 5e-3, 1e-2]}

    if optimizer_name == "muon":
        # Muon: lr is the main knob; weight_decay is typically 0
        if arch == "mlp":
            if mode == "quick":
                return {"lr": [5e-3, 1e-2, 2e-2], "wd": [0.0]}
            return {"lr": [2e-3, 5e-3, 1e-2, 2e-2, 5e-2], "wd": [0.0]}
        if arch == "resnet20":
            if mode == "quick":
                return {"lr": [5e-3, 1e-2, 2e-2], "wd": [0.0]}
            return {"lr": [2e-3, 5e-3, 1e-2, 2e-2, 5e-2], "wd": [0.0]}

    raise ValueError(f"No default grid for arch={arch}, optimizer={optimizer_name}")


__all__ = [
    "TrainConfig",
    "Muon",
    "TrialResult",
    "ExperimentSetup",
    "WarmupCosineConfig",
    "WarmupCosineSchedule",
    "ScheduledOptimizer",
    "build_model",
    "default_hparam_grid",
    "evaluate",
    "extract_targets",
    "prepare_experiment_setup",
    "save_history_csv",
    "save_json",
    "set_seed",
    "stratified_train_val_split",
    "train_run",
    "tune_grid",
    "utils",
    "datasets",
]
