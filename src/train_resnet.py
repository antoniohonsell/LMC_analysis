#!/usr/bin/env python3
"""
train_resnet.py

Trains ResNet20 on CIFAR-10 and CIFAR-100 using:
- Architecture tweak: BatchNorm -> true LayerNorm2d (channelwise LN for NCHW)
- Augmentation: random resize (0.8–1.2), random 32x32 crop, hflip, rotation ±30°
- Optimizer: SGD(momentum=0.9, weight_decay=5e-4; no decay on 1D/bias/norm params)
- LR schedule: linear warmup (1 epoch) from 1e-6 -> 1e-1, then single cosine decay

Runs:
1) Full training split (train=50k-val, val=5000 stratified), for multiple seeds
2) Disjoint balanced Subset A and Subset B from training indices, for multiple seeds

Uses repo modules:
- architectures.py (build_model)
- train_loop.py (train loop + checkpointing)
- utils.py (get_device)

Outputs per run:
- checkpoints saved by train_loop.py: *_best.pth, *_epoch{n}.pth, *_final.pth
- config.json
- history.pt / history.json / history.csv
- test_metrics.json (final + best)
- split indices .pt (once per dataset)

HOW TO RUN:
python train_resnet.py --out_dir ./runs_resnet20_ln_warmcos

"""
from __future__ import annotations

import argparse
import csv
import json
import os
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset
from torchvision.transforms import functional as TF
from torchvision.transforms import InterpolationMode

import architectures
import train_loop
import utils


# ---------------------------
# Dataset stats
# ---------------------------
DATASET_STATS = {
    "CIFAR10": {
        "mean": (0.49139968, 0.48215841, 0.44653091),
        "std":  (0.24703223, 0.24348513, 0.26158784),
        "num_classes": 10,
    },
    "CIFAR100": {
        "mean": (0.50707516, 0.48654887, 0.44091784),
        "std":  (0.26733429, 0.25643846, 0.27615047),
        "num_classes": 100,
    },
}


# ---------------------------
# Reproducibility helpers
# ---------------------------
def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # Best-effort determinism
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def seed_worker(worker_id: int) -> None:
    # Ensures dataloader workers are deterministically seeded
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


# ---------------------------
# Splits (stratified val + balanced disjoint subsets)
# ---------------------------
def stratified_train_val_split(
    targets: List[int],
    val_size: int,
    seed: int,
    num_classes: int,
) -> Tuple[List[int], List[int]]:
    """
    Deterministic stratified split: val gets ~val_size samples total, balanced across classes.
    Returns: (train_indices, val_indices) indices into the full training dataset.
    """
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


def split_train_into_two_balanced_subsets(
    train_indices: List[int],
    targets: List[int],
    seed: int,
    num_classes: int,
) -> Tuple[List[int], List[int], Dict[str, Any]]:
    """
    Given train_indices into the full dataset, return two disjoint balanced subsets A/B.
    Balanced means: each subset has the same number per class, and classes are equally represented.
    Uses all possible samples subject to exact balance constraints.
    """
    rng = np.random.default_rng(seed)
    targets_arr = np.asarray(targets)
    train_indices_arr = np.asarray(train_indices)

    per_class = {
        c: train_indices_arr[targets_arr[train_indices_arr] == c].tolist()
        for c in range(num_classes)
    }
    for c in range(num_classes):
        rng.shuffle(per_class[c])

    min_count = min(len(per_class[c]) for c in range(num_classes))
    k = (min_count // 2) * 2  # largest even <= min_count
    half = k // 2

    subset_a: List[int] = []
    subset_b: List[int] = []
    for c in range(num_classes):
        cls = per_class[c][:k]
        subset_a.extend(cls[:half])
        subset_b.extend(cls[half:])

    rng.shuffle(subset_a)
    rng.shuffle(subset_b)

    meta = {
        "per_class_used": half,
        "dropped_per_class": {c: len(per_class[c]) - k for c in range(num_classes)},
    }
    return subset_a, subset_b, meta


# ---------------------------
# Augmentation
# ---------------------------
class RandomResizeThenCrop:
    """
    Randomly resize a PIL image to (round(base_size * s), round(base_size * s))
    with s ~ Uniform(scale[0], scale[1]), then random crop crop_size x crop_size.
    If resized image is smaller than crop_size, pads symmetrically.
    """
    def __init__(
        self,
        base_size: int = 32,
        scale: Tuple[float, float] = (0.8, 1.2),
        crop_size: int = 32,
        interpolation: InterpolationMode = InterpolationMode.BILINEAR,
        fill: int = 0,
    ):
        self.base_size = base_size
        self.scale = scale
        self.crop_size = crop_size
        self.interpolation = interpolation
        self.fill = fill

    def __call__(self, img):
        s = random.uniform(self.scale[0], self.scale[1])
        new_size = int(round(self.base_size * s))
        new_size = max(1, new_size)

        img = TF.resize(img, [new_size, new_size], interpolation=self.interpolation)

        # Pad to at least crop_size
        w, h = img.size  # PIL: (W, H)
        pad_w = max(0, self.crop_size - w)
        pad_h = max(0, self.crop_size - h)
        if pad_w > 0 or pad_h > 0:
            left = pad_w // 2
            right = pad_w - left
            top = pad_h // 2
            bottom = pad_h - top
            img = TF.pad(img, [left, top, right, bottom], fill=self.fill)

        # Random crop
        w, h = img.size
        if w == self.crop_size and h == self.crop_size:
            return img
        max_left = w - self.crop_size
        max_top = h - self.crop_size
        left = random.randint(0, max_left)
        top = random.randint(0, max_top)
        img = TF.crop(img, top=top, left=left, height=self.crop_size, width=self.crop_size)
        return img


def build_transforms(dataset_name: str) -> Tuple[transforms.Compose, transforms.Compose]:
    stats = DATASET_STATS[dataset_name]
    normalize = transforms.Normalize(stats["mean"], stats["std"])

    train_tf = transforms.Compose([
        RandomResizeThenCrop(base_size=32, scale=(0.8, 1.2), crop_size=32),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(degrees=30, interpolation=InterpolationMode.BILINEAR, fill=0),
        transforms.ToTensor(),
        normalize,
    ])

    eval_tf = transforms.Compose([
        transforms.ToTensor(),
        normalize,
    ])

    return train_tf, eval_tf


# ---------------------------
# LayerNorm2d + BN->LN replacement
# ---------------------------
class LayerNorm2d(nn.Module):
    """
    True LayerNorm over channels for NCHW tensors by applying nn.LayerNorm(C)
    at each spatial location (i.e., over the channel dimension only).
    """
    def __init__(self, num_channels: int, eps: float = 1e-5, affine: bool = True):
        super().__init__()
        self.ln = nn.LayerNorm(num_channels, eps=eps, elementwise_affine=affine)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (N, C, H, W) -> (N, H, W, C) -> LN(C) -> back
        x = x.permute(0, 2, 3, 1).contiguous()
        x = self.ln(x)
        x = x.permute(0, 3, 1, 2).contiguous()
        return x


def replace_batchnorm_with_layernorm2d(model: nn.Module) -> nn.Module:
    """
    Recursively replace nn.BatchNorm2d modules with LayerNorm2d(C).
    """
    for name, child in model.named_children():
        if isinstance(child, nn.BatchNorm2d):
            setattr(model, name, LayerNorm2d(child.num_features))
        else:
            replace_batchnorm_with_layernorm2d(child)
    return model


# ---------------------------
# LR schedule (per-step warmup + cosine) implemented inside optimizer wrapper
# ---------------------------
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
        step = int(step)
        step = max(0, min(step, self.total_steps - 1))

        # Warmup: step in [0, warmup_steps-1]
        if self.warmup_steps > 0 and step < self.warmup_steps:
            if self.warmup_steps == 1:
                return float(self.cfg.lr_peak)
            # map 0 -> lr_start, warmup_steps-1 -> lr_peak
            t = step / (self.warmup_steps - 1)
            return float(self.cfg.lr_start + t * (self.cfg.lr_peak - self.cfg.lr_start))

        # Cosine: step in [warmup_steps, total_steps-1]
        cos_steps = max(1, self.total_steps - self.warmup_steps)
        if cos_steps == 1:
            return float(self.cfg.lr_min)

        t = (step - self.warmup_steps) / (cos_steps - 1)  # 0..1
        # cosine from lr_peak -> lr_min
        lr = self.cfg.lr_min + 0.5 * (self.cfg.lr_peak - self.cfg.lr_min) * (1.0 + np.cos(np.pi * t))
        return float(lr)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_steps": self.total_steps,
            "warmup_steps": self.warmup_steps,
            "cfg": asdict(self.cfg),
        }


class ScheduledOptimizer:
    """
    Thin wrapper that updates LR *every optimizer.step()* according to WarmupCosineSchedule.
    Compatible with train_loop.py which calls optimizer.step() per batch.
    """
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
        # schedule is not rebuilt here; assume caller uses same schedule config.


# ---------------------------
# Optimizer building (SGD, wd only on "decay" params)
# ---------------------------
def build_sgd_with_param_groups(model: nn.Module, lr: float, weight_decay: float, momentum: float) -> torch.optim.Optimizer:
    decay, no_decay = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        # no weight decay for 1D params (norm scale), biases, and explicit norm identifiers
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


# ---------------------------
# Evaluation + stats saving
# ---------------------------
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
    return {"loss": loss_sum / total, "accuracy": correct / total}


def save_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, sort_keys=True)


def save_history_csv(path: Path, history: Dict[str, List[float]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    n = len(history["train_loss"])
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


def lr_summary_per_epoch(schedule: WarmupCosineSchedule, steps_per_epoch: int, epochs: int) -> List[Dict[str, float]]:
    out = []
    for e in range(epochs):
        start_step = e * steps_per_epoch
        end_step = min((e + 1) * steps_per_epoch - 1, schedule.total_steps - 1)
        out.append({
            "epoch": e + 1,
            "lr_start": schedule.lr_at(start_step),
            "lr_end": schedule.lr_at(end_step),
        })
    return out


# ---------------------------
# Training orchestration
# ---------------------------
@dataclass
class RunConfig:
    dataset: str
    regime: str  # "full" | "subset_A" | "subset_B"
    base_seed: int
    model_seed: int
    split_seed: int
    subset_seed: int
    val_size: int
    epochs: int
    batch_size: int
    num_workers: int
    weight_decay: float
    momentum: float
    width_multiplier: int
    shortcut_option: str
    lr_schedule: Dict[str, Any]


def train_one_run(
    *,
    run_dir: Path,
    run_name: str,
    dataset_name: str,
    train_ds: Subset,
    val_ds: Subset,
    test_loader: DataLoader,
    device: torch.device,
    epochs: int,
    batch_size: int,
    num_workers: int,
    model_seed: int,
    lr_cfg: WarmupCosineConfig,
    weight_decay: float,
    momentum: float,
    width_multiplier: int,
    shortcut_option: str,
    save_every: int = 1,
) -> Dict[str, Any]:
    run_dir.mkdir(parents=True, exist_ok=True)

    set_seed(model_seed)
    g_loader = torch.Generator().manual_seed(model_seed)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        worker_init_fn=seed_worker if num_workers > 0 else None,
        generator=g_loader,
        pin_memory=(device.type == "cuda"),
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
    )

    num_classes = int(DATASET_STATS[dataset_name]["num_classes"])
    # Build ResNet20 from architectures.py, then do BN->LN replacement
    model = architectures.build_model(
            "resnet20",
            num_classes=num_classes,
            norm="flax_ln",
            width_multiplier=width_multiplier,
            shortcut_option=shortcut_option,
            ).to(device)
    # model = replace_batchnorm_with_layernorm2d(model).to(device)

    criterion = nn.CrossEntropyLoss()

    # Warmup+cosine is per optimizer step, so no epoch scheduler is needed.
    base_optim = build_sgd_with_param_groups(model, lr=lr_cfg.lr_start, weight_decay=weight_decay, momentum=momentum)

    steps_per_epoch = len(train_loader)
    total_steps = steps_per_epoch * epochs
    warmup_steps = steps_per_epoch * max(0, int(lr_cfg.warmup_epochs))

    schedule = WarmupCosineSchedule(total_steps=total_steps, warmup_steps=warmup_steps, cfg=lr_cfg)
    optimizer = ScheduledOptimizer(base_optim, schedule)

    # Train
    t0 = time.time()
    history = train_loop.train(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=None,  # IMPORTANT: per-step schedule is inside optimizer.step()
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=epochs,
        device=device,
        save_dir=str(run_dir),
        run_name=run_name,
        save_every=epochs +1, # usually I put to save_every=save_every, to save every epoch 
        save_last=False, # usually I put it to false 
    )
    t1 = time.time()

    # Save history in multiple formats
    torch.save({"history": history}, run_dir / "history.pt")
    save_json(run_dir / "history.json", history)
    save_history_csv(run_dir / "history.csv", history)

    # LR summary (epoch endpoints)
    save_json(run_dir / "lr_per_epoch.json", lr_summary_per_epoch(schedule, steps_per_epoch, epochs))

    # Evaluate final model on test set
    final_metrics = evaluate(model, test_loader, criterion, device)

    # Evaluate best checkpoint on test set
    best_path = run_dir / f"{run_name}_best.pth"
    best_metrics: Optional[Dict[str, float]] = None
    if best_path.exists():
        ckpt = torch.load(best_path, map_location=device)
        model.load_state_dict(ckpt["state_dict"])
        best_metrics = evaluate(model, test_loader, criterion, device)

    metrics = {
        "final": final_metrics,
        "best": best_metrics,
        "train_wallclock_sec": float(t1 - t0),
    }
    save_json(run_dir / "test_metrics.json", metrics)

    # Save a lightweight summary
    summary = {
        "run_name": run_name,
        "train_size": len(train_ds),
        "val_size": len(val_ds),
        "test_size": len(test_loader.dataset),
        "history_last": {k: v[-1] if len(v) else None for k, v in history.items()},
        "test_metrics": metrics,
        "schedule": schedule.to_dict(),
    }
    save_json(run_dir / "summary.json", summary)

    # Cleanup
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()

    return summary


def load_cifar_datasets(dataset_name: str, data_root: str) -> Tuple[Any, Any, Any, List[int]]:
    train_tf, eval_tf = build_transforms(dataset_name)

    if dataset_name == "CIFAR10":
        train_full = torchvision.datasets.CIFAR10(root=data_root, train=True, download=True, transform=train_tf)
        eval_full = torchvision.datasets.CIFAR10(root=data_root, train=True, download=True, transform=eval_tf)
        test_ds = torchvision.datasets.CIFAR10(root=data_root, train=False, download=True, transform=eval_tf)
    elif dataset_name == "CIFAR100":
        train_full = torchvision.datasets.CIFAR100(root=data_root, train=True, download=True, transform=train_tf)
        eval_full = torchvision.datasets.CIFAR100(root=data_root, train=True, download=True, transform=eval_tf)
        test_ds = torchvision.datasets.CIFAR100(root=data_root, train=False, download=True, transform=eval_tf)
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")

    targets = list(train_full.targets)  # 50k labels
    return train_full, eval_full, test_ds, targets


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", default=["CIFAR10", "CIFAR100"], choices=["CIFAR10", "CIFAR100"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1])

    parser.add_argument("--data_root", type=str, default="./data")
    parser.add_argument("--out_dir", type=str, default="./runs_resnet20_ln_warmcos")

    parser.add_argument("--split_seed", type=int, default=50)     # fixed train/val split across all runs
    parser.add_argument("--subset_seed", type=int, default=None)  # fixed A/B membership; default = split_seed
    parser.add_argument("--val_size", type=int, default=5000)

    parser.add_argument("--epochs_cifar10", type=int, default=128)
    parser.add_argument("--epochs_cifar100", type=int, default=200)

    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--num_workers", type=int, default=8)

    parser.add_argument("--weight_decay", type=float, default=5e-4)
    parser.add_argument("--momentum", type=float, default=0.9)

    parser.add_argument("--lr_start", type=float, default=1e-6)
    parser.add_argument("--lr_peak", type=float, default=1e-1)
    parser.add_argument("--lr_min", type=float, default=0.0)
    parser.add_argument("--warmup_epochs", type=int, default=1)

    parser.add_argument("--width_multiplier", type=int, default=1)
    parser.add_argument("--shortcut_option", type=str, default="C", choices=["A", "B", "C"])

    parser.add_argument("--skip_full", action="store_true")
    parser.add_argument("--skip_disjoint", action="store_true")

    args = parser.parse_args()

    if args.subset_seed is None:
        args.subset_seed = args.split_seed

    device = utils.get_device()

    out_root = Path(args.out_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    lr_cfg = WarmupCosineConfig(
        lr_start=float(args.lr_start),
        lr_peak=float(args.lr_peak),
        lr_min=float(args.lr_min),
        warmup_epochs=int(args.warmup_epochs),
    )

    for dataset_name in args.datasets:
        if dataset_name not in DATASET_STATS:
            raise ValueError(f"Unsupported dataset: {dataset_name}")

        epochs = int(args.epochs_cifar10 if dataset_name == "CIFAR10" else args.epochs_cifar100)
        stats = DATASET_STATS[dataset_name]
        num_classes = int(stats["num_classes"])

        # Datasets (separate objects so val has no augmentation)
        train_full, eval_full, test_ds, targets = load_cifar_datasets(dataset_name, args.data_root)

        # Fixed stratified train/val split (balanced per class)
        train_indices, val_indices = stratified_train_val_split(
            targets=targets,
            val_size=int(args.val_size),
            seed=int(args.split_seed),
            num_classes=num_classes,
        )

        # Disjoint balanced A/B subsets from TRAIN indices
        subset_a_idx, subset_b_idx, subset_meta = split_train_into_two_balanced_subsets(
            train_indices=train_indices,
            targets=targets,
            seed=int(args.subset_seed),
            num_classes=num_classes,
        )

        # Save split indices once per dataset
        ds_root = out_root / dataset_name
        ds_root.mkdir(parents=True, exist_ok=True)
        split_path = ds_root / f"indices_{dataset_name}_splitseed{args.split_seed}_subsetseed{args.subset_seed}_val{args.val_size}.pt"
        if not split_path.exists():
            torch.save(
                {
                    "dataset": dataset_name,
                    "split_seed": int(args.split_seed),
                    "subset_seed": int(args.subset_seed),
                    "val_size": int(args.val_size),
                    "train_indices": train_indices,
                    "val_indices": val_indices,
                    "subset_a_indices": subset_a_idx,
                    "subset_b_indices": subset_b_idx,
                    "subset_meta": subset_meta,
                },
                split_path,
            )

        # Build Subset datasets
        train_ds_full = Subset(train_full, train_indices)
        subset_a = Subset(train_full, subset_a_idx)
        subset_b = Subset(train_full, subset_b_idx)
        val_ds = Subset(eval_full, val_indices)

        # Test loader
        test_loader = DataLoader(
            test_ds,
            batch_size=int(args.batch_size),
            shuffle=False,
            num_workers=int(args.num_workers),
            pin_memory=(device.type == "cuda"),
        )

        # 1) Full regime
        if not args.skip_full:
            full_root = ds_root / "full"
            full_root.mkdir(parents=True, exist_ok=True)

            # Dataset-level config
            save_json(full_root / "dataset_config.json", {
                "dataset": dataset_name,
                "epochs": epochs,
                "batch_size": int(args.batch_size),
                "val_size": int(args.val_size),
                "split_seed": int(args.split_seed),
                "lr_cfg": asdict(lr_cfg),
                "optimizer": {
                    "type": "SGD",
                    "momentum": float(args.momentum),
                    "weight_decay": float(args.weight_decay),
                },
                "model": {
                    "name": "resnet20",
                    "width_multiplier": int(args.width_multiplier),
                    "shortcut_option": str(args.shortcut_option),
                    "norm": "BN->LayerNorm2d replacement",
                },
                "device": str(device),
            })

            for base_seed in args.seeds:
                model_seed = int(base_seed)
                run_dir = full_root / f"seed_{base_seed}"
                run_name = f"resnet20_{dataset_name}_full_seed{base_seed}"

                rc = RunConfig(
                    dataset=dataset_name,
                    regime="full",
                    base_seed=int(base_seed),
                    model_seed=model_seed,
                    split_seed=int(args.split_seed),
                    subset_seed=int(args.subset_seed),
                    val_size=int(args.val_size),
                    epochs=epochs,
                    batch_size=int(args.batch_size),
                    num_workers=int(args.num_workers),
                    weight_decay=float(args.weight_decay),
                    momentum=float(args.momentum),
                    width_multiplier=int(args.width_multiplier),
                    shortcut_option=str(args.shortcut_option),
                    lr_schedule={
                        "type": "warmup+cosine (per-step)",
                        **asdict(lr_cfg),
                    },
                )
                save_json(run_dir / "config.json", asdict(rc))

                train_one_run(
                    run_dir=run_dir,
                    run_name=run_name,
                    dataset_name=dataset_name,
                    train_ds=train_ds_full,
                    val_ds=val_ds,
                    test_loader=test_loader,
                    device=device,
                    epochs=epochs,
                    batch_size=int(args.batch_size),
                    num_workers=int(args.num_workers),
                    model_seed=model_seed,
                    lr_cfg=lr_cfg,
                    weight_decay=float(args.weight_decay),
                    momentum=float(args.momentum),
                    width_multiplier=int(args.width_multiplier),
                    shortcut_option=str(args.shortcut_option),
                    save_every=1,
                )

        # 2) Disjoint subsets regime
        if not args.skip_disjoint:
            dis_root = ds_root / "disjoint"
            dis_root.mkdir(parents=True, exist_ok=True)

            save_json(dis_root / "dataset_config.json", {
                "dataset": dataset_name,
                "epochs": epochs,
                "batch_size": int(args.batch_size),
                "val_size": int(args.val_size),
                "split_seed": int(args.split_seed),
                "subset_seed": int(args.subset_seed),
                "subset_meta": subset_meta,
                "lr_cfg": asdict(lr_cfg),
                "optimizer": {
                    "type": "SGD",
                    "momentum": float(args.momentum),
                    "weight_decay": float(args.weight_decay),
                },
                "model": {
                    "name": "resnet20",
                    "width_multiplier": int(args.width_multiplier),
                    "shortcut_option": str(args.shortcut_option),
                    "norm": "BN->LayerNorm2d replacement",
                },
                "device": str(device),
            })

            for base_seed in args.seeds:
                # Follow the repo’s convention of distinct model seeds per subset
                # (base_seed*1000 + offset) as seen in the disjoint runner :contentReference[oaicite:5]{index=5}.
                for subset_name, subset_ds, offset in [("A", subset_a, 1), ("B", subset_b, 2)]:
                    model_seed = int(base_seed) * 1000 + offset
                    run_dir = dis_root / f"seed_{base_seed}" / f"subset_{subset_name}"
                    run_name = f"resnet20_{dataset_name}_seed{base_seed}_subset{subset_name}"

                    rc = RunConfig(
                        dataset=dataset_name,
                        regime=f"subset_{subset_name}",
                        base_seed=int(base_seed),
                        model_seed=model_seed,
                        split_seed=int(args.split_seed),
                        subset_seed=int(args.subset_seed),
                        val_size=int(args.val_size),
                        epochs=epochs,
                        batch_size=int(args.batch_size),
                        num_workers=int(args.num_workers),
                        weight_decay=float(args.weight_decay),
                        momentum=float(args.momentum),
                        width_multiplier=int(args.width_multiplier),
                        shortcut_option=str(args.shortcut_option),
                        lr_schedule={
                            "type": "warmup+cosine (per-step)",
                            **asdict(lr_cfg),
                        },
                    )
                    save_json(run_dir / "config.json", asdict(rc))

                    train_one_run(
                        run_dir=run_dir,
                        run_name=run_name,
                        dataset_name=dataset_name,
                        train_ds=subset_ds,
                        val_ds=val_ds,
                        test_loader=test_loader,
                        device=device,
                        epochs=epochs,
                        batch_size=int(args.batch_size),
                        num_workers=int(args.num_workers),
                        model_seed=model_seed,
                        lr_cfg=lr_cfg,
                        weight_decay=float(args.weight_decay),
                        momentum=float(args.momentum),
                        width_multiplier=int(args.width_multiplier),
                        shortcut_option=str(args.shortcut_option),
                        save_every=1,
                    )
                break


if __name__ == "__main__":
    main()
