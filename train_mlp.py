#!/usr/bin/env python3
"""
train_mlp.py

Trains an MLP on:
  - MNIST
  - CIFAR-10

Across all combinations:
  - seed ∈ {0, 1}
  - subset ∈ {A, B}   (disjoint, balanced subsets carved from the training indices)

Optimizer:
  - Adam(lr=1e-3)

This script is designed to match the repo conventions used by train_resnet.py:
  - Uses datasets.py for dataset creation (train_full / eval_full / test_ds)
  - Uses train_loop.py for training + checkpointing
  - Saves run configs + history + test metrics in a structured out_dir

Output structure (default --out_dir ./runs_mlp):
  runs_mlp/
    MNIST/
      disjoint/
        indices_MNIST_splitseed50_subsetseed50_val5000.pt
        seed_0/subset_A/...
        seed_0/subset_B/...
        seed_1/subset_A/...
        seed_1/subset_B/...
    CIFAR10/
      disjoint/
        indices_CIFAR10_splitseed50_subsetseed50_val5000.pt
        ...

Usage:
  python train_mlp.py
  python train_mlp.py --out_dir ./runs_mlp --val_size 5000 --epochs_mnist 40 --epochs_cifar10 100
  python train_mlp.py --arch MLP --out_dir ./runs_mlp --val_size 5000 --epochs_mnist 40
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

import architectures
import datasets
import train_loop
import utils


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
# Small utilities
# ---------------------------
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
    return {"loss": loss_sum / total, "accuracy": correct / total}


def extract_targets(ds: Any) -> List[int]:
    """
    Works for torchvision datasets and Subset wrappers.
    """
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

    raise AttributeError("Could not extract targets from dataset (expected .targets/.labels or Subset thereof).")


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
        "per_class_used_per_subset": half,
        "dropped_per_class": {c: len(per_class[c]) - k for c in range(num_classes)},
    }
    return subset_a, subset_b, meta


# ---------------------------
# Model building (prefer repo MLP; fallback to internal MLP)
# ---------------------------
class FallbackMLP(nn.Module):
    def __init__(
        self,
        *,
        input_dim: int,
        num_classes: int,
        hidden_dims: Sequence[int] = (512,),
        dropout: float = 0.0,
    ):
        super().__init__()
        dims = [input_dim] + list(hidden_dims) + [num_classes]
        layers: List[nn.Module] = []
        for i in range(len(dims) - 2):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(p=dropout))
        layers.append(nn.Linear(dims[-2], dims[-1]))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.view(x.size(0), -1)
        return self.net(x)


def _safe_instantiate(builder: Any, kwargs: Dict[str, Any]) -> nn.Module:
    """
    Instantiate builder with only accepted kwargs (robust to signature differences).
    """
    import inspect

    if isinstance(builder, type) and issubclass(builder, nn.Module):
        sig = inspect.signature(builder.__init__)
        params = set(sig.parameters.keys())
        params.discard("self")
        filtered = {k: v for k, v in kwargs.items() if k in params}
        return builder(**filtered)  # type: ignore[misc]

    sig = inspect.signature(builder)
    params = set(sig.parameters.keys())
    filtered = {k: v for k, v in kwargs.items() if k in params}
    return builder(**filtered)


def build_mlp_model(
    *,
    preferred_arch: str,
    num_classes: int,
    in_channels: int,
    image_size: Tuple[int, int],
    hidden_dims: Sequence[int],
    dropout: float,
) -> nn.Module:
    """
    Tries, in order:
      1) architectures.build_model(preferred_arch, ...)
      2) architectures registry direct builder (if present)
      3) fallback internal MLP
    """
    c, (h, w) = in_channels, image_size
    input_shape = (c, h, w)
    input_dim = c * h * w

    # Candidate kwargs to support multiple possible MLP constructor styles
    candidate_kwargs: Dict[str, Any] = {
        "name": preferred_arch,
        "num_classes": num_classes,
        "in_channels": in_channels,
        "input_shape": input_shape,
        "input_dim": input_dim,
        "hidden": int(hidden_dims[0]) if len(hidden_dims) else 512,
        "hidden_dims": tuple(int(x) for x in hidden_dims),
        "hidden_sizes": tuple(int(x) for x in hidden_dims),
        "dropout": float(dropout),
    }

    # 1) Preferred: build_model if it works
    if hasattr(architectures, "build_model"):
        try:
            # Keep args minimal-first; pass extras only if accepted by builder, if possible.
            # We try "rich" call; if it TypeErrors due to signature, we fall back below.
            return architectures.build_model(  # type: ignore[attr-defined]
                preferred_arch,
                num_classes=num_classes,
                in_channels=in_channels,
                input_shape=input_shape,
                input_dim=input_dim,
                hidden=int(hidden_dims[0]) if len(hidden_dims) else 512,
                hidden_dims=tuple(int(x) for x in hidden_dims),
                hidden_sizes=tuple(int(x) for x in hidden_dims),
                dropout=float(dropout),
            )
        except Exception:
            pass

    # 2) Direct registry access (if present)
    builder = None
    if hasattr(architectures, "_MODEL_REGISTRY"):
        reg = getattr(architectures, "_MODEL_REGISTRY")
        if isinstance(reg, dict) and preferred_arch in reg:
            builder = reg[preferred_arch]

    if builder is None and hasattr(architectures, preferred_arch):
        builder = getattr(architectures, preferred_arch)

    if builder is not None:
        try:
            # Many builders won't want 'name'
            filtered = {k: v for k, v in candidate_kwargs.items() if k != "name"}
            return _safe_instantiate(builder, filtered)
        except Exception:
            pass

    # 3) Fallback internal MLP (always works)
    return FallbackMLP(input_dim=input_dim, num_classes=num_classes, hidden_dims=hidden_dims, dropout=dropout)

from dataclasses import dataclass
from typing import Callable, Sequence, Dict, Any, List, Optional
import numpy as np

@dataclass
class LRTuningTrial:
    lr: float
    best_val_acc: float
    best_val_loss: float
    best_epoch: int
    run_dir: str

def tune_lr_mlp(
    *,
    out_dir: Path,
    run_prefix: str,
    model_fn: Callable[[], nn.Module],     # builds a fresh model each trial
    train_ds: torch.utils.data.Dataset,    # Subset(train_full, train_indices)
    val_ds: torch.utils.data.Dataset,      # Subset(eval_full, val_indices)
    device: torch.device,
    model_seed: int,
    epochs: int,
    batch_size: int,
    num_workers: int,
    lr_grid: Sequence[float],
    weight_decay: float,
) -> Dict[str, Any]:
    trials: List[LRTuningTrial] = []
    best_lr: Optional[float] = None
    best_acc = -1.0
    best_loss = float("inf")

    for lr in lr_grid:
        trial_name = f"{run_prefix}_tune_lr{lr:.6g}"
        trial_dir = out_dir / f"lr_{lr:.6g}"
        model = model_fn()

        # IMPORTANT: pass test_loader=None so tuning never looks at test
        summary = train_one_run(
            run_dir=trial_dir,
            run_name=trial_name,
            model=model,
            train_ds=train_ds,
            val_ds=val_ds,
            test_loader=None,                 # requires your small refactor
            device=device,
            model_seed=model_seed,
            epochs=epochs,
            batch_size=batch_size,
            num_workers=num_workers,
            lr=float(lr),
            weight_decay=float(weight_decay),
            save_every=epochs + 1,
        )

        # easiest: read back history.json you already save in train_one_run
        hist = json.load((trial_dir / "history.json").open("r", encoding="utf-8"))
        val_accs = [float(x) for x in hist["val_accuracy"]]
        val_losses = [float(x) for x in hist["val_loss"]]

        best_epoch_idx = int(np.nanargmax(val_accs))
        trial_best_acc = float(val_accs[best_epoch_idx])
        trial_best_loss = float(val_losses[best_epoch_idx])

        trials.append(LRTuningTrial(
            lr=float(lr),
            best_val_acc=trial_best_acc,
            best_val_loss=trial_best_loss,
            best_epoch=best_epoch_idx + 1,
            run_dir=str(trial_dir),
        ))

        # pick best by val acc; tie-break by val loss (same pattern as your tuning script) :contentReference[oaicite:4]{index=4}
        if (trial_best_acc > best_acc) or (trial_best_acc == best_acc and trial_best_loss < best_loss):
            best_acc = trial_best_acc
            best_loss = trial_best_loss
            best_lr = float(lr)

    assert best_lr is not None
    payload = {
        "best_lr": best_lr,
        "best_val_accuracy": best_acc,
        "best_val_loss": best_loss,
        "trials": [t.__dict__ for t in trials],
    }
    save_json(out_dir / "lr_tuning.json", payload)
    return payload

def train_full_after_tuning(
    *,
    out_dir: Path,
    run_name: str,
    model_fn: Callable[[], nn.Module],
    train_full_ds: torch.utils.data.Dataset,   # Subset(train_full, all_indices)
    monitor_ds: torch.utils.data.Dataset,      # Subset(eval_full, all_indices) for clean metrics
    test_loader: DataLoader,
    device: torch.device,
    model_seed: int,
    epochs: int,
    batch_size: int,
    num_workers: int,
    lr: float,
    weight_decay: float,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    set_seed(model_seed)
    g = torch.Generator().manual_seed(model_seed)

    train_loader = DataLoader(train_full_ds, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, worker_init_fn=seed_worker if num_workers>0 else None,
                              generator=g, pin_memory=(device.type=="cuda"))
    monitor_loader = DataLoader(monitor_ds, batch_size=batch_size, shuffle=False,
                                num_workers=num_workers, pin_memory=(device.type=="cuda"))

    model = model_fn().to(device)
    crit = nn.CrossEntropyLoss()
    opt = torch.optim.Adam(model.parameters(), lr=float(lr), weight_decay=float(weight_decay))

    history = train_loop.train(
        model=model,
        criterion=crit,
        optimizer=opt,
        scheduler=None,
        train_loader=train_loader,
        val_loader=monitor_loader,   # not a true “val”, just monitoring
        epochs=epochs,
        device=device,
        save_dir=str(out_dir),
        run_name=run_name,
        save_every=max(1, epochs),   # keep files minimal
        save_last=True,              # ensures *_final.pth exists :contentReference[oaicite:7]{index=7}
    )

    save_json(out_dir / "history.json", history)

    # optional: evaluate final (or best) on test once, here (this is now “post-selection”)
    final_ckpt = out_dir / f"{run_name}_final.pth"
    ckpt = torch.load(final_ckpt, map_location=device)
    model.load_state_dict(ckpt["state_dict"])
    test_metrics = evaluate(model, test_loader, crit, device)
    save_json(out_dir / "test_metrics.json", {"final": test_metrics})


# ---------------------------
# Training orchestration
# ---------------------------
@dataclass
class RunConfig:
    dataset: str
    subset: str  # "A" or "B"
    base_seed: int
    model_seed: int
    split_seed: int
    subset_seed: int
    val_size: int
    epochs: int
    batch_size: int
    num_workers: int
    optimizer: Dict[str, Any]
    model: Dict[str, Any]
    device: str


def train_one_run(
    *,
    run_dir: Path,
    run_name: str,
    model: nn.Module,
    train_ds: Subset,
    val_ds: Subset,
    test_loader: DataLoader,
    device: torch.device,
    model_seed: int,
    epochs: int,
    batch_size: int,
    num_workers: int,
    lr: float,
    weight_decay: float,
    save_every: int,
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

    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=float(lr), weight_decay=float(weight_decay))

    t0 = time.time()
    history = train_loop.train(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=None,
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=epochs,
        device=device,
        save_dir=str(run_dir),
        run_name=run_name,
        save_every=epochs +1, # put it to save_every=save_every if you want to save each epoch
        save_last=False, # put it to True if you want to save the last  
    )
    t1 = time.time()

    # Save history in multiple formats
    torch.save({"history": history}, run_dir / "history.pt")
    save_json(run_dir / "history.json", history)
    save_history_csv(run_dir / "history.csv", history)

    # Evaluate final model on test set (skip if test_loader is None, e.g., during LR tuning)
    metrics = {}
    summary_test_size = 0
    if test_loader is not None:
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
        summary_test_size = len(test_loader.dataset)
    
    save_json(run_dir / "test_metrics.json", metrics)

    summary = {
        "run_name": run_name,
        "train_size": len(train_ds),
        "val_size": len(val_ds),
        "test_size": summary_test_size,
        "history_last": {k: v[-1] if len(v) else None for k, v in history.items()},
        "test_metrics": metrics,
    }
    save_json(run_dir / "summary.json", summary)

    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()

    return summary


def parse_hidden_dims(s: str) -> Tuple[int, ...]:
    s = (s or "").strip()
    if not s:
        return (512,)
    parts = [p.strip() for p in s.replace(";", ",").split(",") if p.strip()]
    return tuple(int(p) for p in parts)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", default=["MNIST"], choices=["MNIST", "FASHIONMNIST",  "CIFAR10"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[0])

    parser.add_argument("--data_root", type=str, default="./data")
    parser.add_argument("--out_dir", type=str, default="./runs_mlp")

    # Fixed splits across all runs (like train_resnet.py)
    parser.add_argument("--split_seed", type=int, default=50)      # fixed train/val split
    parser.add_argument("--subset_seed", type=int, default=None)   # fixed A/B membership; default = split_seed
    parser.add_argument("--val_size", type=int, default=5000)

    parser.add_argument("--epochs_mnist", type=int, default=20)
    parser.add_argument("--epochs_fashionmnist", type=int, default=40)
    parser.add_argument("--epochs_cifar10", type=int, default=50)

    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--num_workers", type=int, default=4)

    # MLP specifics
    parser.add_argument("--arch", type=str, default="mlp",
                        help="Preferred architecture key in architectures.py (fallbacks: MLP/lightnet/LightNet).")
    parser.add_argument("--hidden_dims", type=str, default="512",
                        help="Comma-separated hidden layer sizes, e.g. '512' or '1024,1024'.")
    parser.add_argument("--dropout", type=float, default=0.0)

    # Optimizer specifics (user request: Adam + lr=1e-3)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=0.0)

    parser.add_argument("--augment_train", action="store_true",
                        help="Force enable train augmentation (overrides datasets.py default).")
    parser.add_argument("--no_augment_train", action="store_true",
                        help="Force disable train augmentation (overrides datasets.py default).")

    parser.add_argument("--save_every", type=int, default=1)

    # Regime + LR tuning
    parser.add_argument("--regime", type=str, default="disjoint", choices=["disjoint", "full"],
                        help="Training regime: 'disjoint' trains on A/B subsets, 'full' trains on full dataset with optional LR tuning.")
    parser.add_argument("--tune_lr", action="store_true",
                        help="Enable learning rate tuning in 'full' regime.")
    parser.add_argument("--lr_grid", nargs="+", type=float, default=[],
                        help="Grid of learning rates to search over (e.g., 1e-4 3e-4 1e-3).")

    args = parser.parse_args()

    if args.subset_seed is None:
        args.subset_seed = int(args.split_seed)

    hidden_dims = parse_hidden_dims(args.hidden_dims)
    device = utils.get_device()

    out_root = Path(args.out_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    # Decide augmentation override
    augment_override: Optional[bool] = None
    if args.augment_train and args.no_augment_train:
        raise ValueError("Pick at most one of --augment_train / --no_augment_train")
    if args.augment_train:
        augment_override = True
    if args.no_augment_train:
        augment_override = False

    # Choose a sensible preferred arch if user gave something not present
    preferred_arch = str(args.arch).strip()
    candidate_arches = [preferred_arch, preferred_arch.upper(), preferred_arch.lower(), "MLP", "mlp", "LightNet", "lightnet"]
    if hasattr(architectures, "available_models"):
        avail = set(architectures.available_models())  # type: ignore[attr-defined]
        for k in candidate_arches:
            if k in avail:
                preferred_arch = k
                break

    for dataset_name in args.datasets:
        # Build datasets via repo utility
        train_full, eval_full, test_ds = datasets.build_datasets(
            dataset_name,
            root=args.data_root,
            download=True,
            augment_train=augment_override,
            normalize=True,
        )

        stats = datasets.DATASET_STATS[dataset_name]
        num_classes = int(stats["num_classes"])
        in_channels = int(stats["in_channels"])
        image_size = tuple(stats["image_size"])

        # Extract targets from the full training split
        targets = extract_targets(train_full)

        # Fixed stratified train/val split
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

        # Build Subset datasets (train uses augmentation; val uses eval_full without augmentation)
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

        # Save split indices once per dataset
        regime_dir = "disjoint" if args.regime == "disjoint" else "full"
        ds_root = out_root / dataset_name / regime_dir
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

        # Epochs per dataset
        if dataset_name == "MNIST":
            epochs = int(args.epochs_mnist)
        elif dataset_name == "FASHIONMNIST":
            epochs = int(args.epochs_fashionmnist)
        else:
            epochs = int(args.epochs_cifar10)

        # Dataset-level config
        save_json(ds_root / "dataset_config.json", {
            "dataset": dataset_name,
            "epochs": epochs,
            "batch_size": int(args.batch_size),
            "val_size": int(args.val_size),
            "split_seed": int(args.split_seed),
            "subset_seed": int(args.subset_seed),
            "subset_meta": subset_meta,
            "optimizer": {"type": "Adam", "lr": float(args.lr), "weight_decay": float(args.weight_decay)},
            "model": {"preferred_arch": preferred_arch, "hidden_dims": list(hidden_dims), "dropout": float(args.dropout)},
            "device": str(device),
        })

        # Run all combinations based on regime
        if args.regime == "disjoint":
            # Run disjoint A/B subset training
            for base_seed in args.seeds:
                for subset_name, subset_ds, offset in [("A", subset_a, 1), ("B", subset_b, 2)]:
                    # Keep the same convention as train_resnet.py: distinct model seeds per subset
                    model_seed = int(base_seed) * 1000 + offset

                    run_dir = ds_root / f"seed_{base_seed}" / f"subset_{subset_name}"
                    run_name = f"{preferred_arch}_{dataset_name}_subset{subset_name}_seed{base_seed}"

                    # Build model (prefer repo MLP, else fallback)
                    model = build_mlp_model(
                        preferred_arch=preferred_arch,
                        num_classes=num_classes,
                        in_channels=in_channels,
                        image_size=image_size,  # (H, W)
                        hidden_dims=hidden_dims,
                        dropout=float(args.dropout),
                    )

                    rc = RunConfig(
                        dataset=dataset_name,
                        subset=subset_name,
                        base_seed=int(base_seed),
                        model_seed=int(model_seed),
                        split_seed=int(args.split_seed),
                        subset_seed=int(args.subset_seed),
                        val_size=int(args.val_size),
                        epochs=epochs,
                        batch_size=int(args.batch_size),
                        num_workers=int(args.num_workers),
                        optimizer={"type": "Adam", "lr": float(args.lr), "weight_decay": float(args.weight_decay)},
                        model={"arch": preferred_arch, "hidden_dims": list(hidden_dims), "dropout": float(args.dropout)},
                        device=str(device),
                    )
                    save_json(run_dir / "config.json", asdict(rc))

                    train_one_run(
                        run_dir=run_dir,
                        run_name=run_name,
                        model=model,
                        train_ds=subset_ds,
                        val_ds=val_ds,
                        test_loader=test_loader,
                        device=device,
                        model_seed=model_seed,
                        epochs=epochs,
                        batch_size=int(args.batch_size),
                        num_workers=int(args.num_workers),
                        lr=float(args.lr),
                        weight_decay=float(args.weight_decay),
                        save_every=int(args.save_every),
                    )

        elif args.regime == "full":
            # Run full dataset training with optional LR tuning
            for base_seed in args.seeds:
                model_seed = base_seed * 1000  # model seed for full regime
                run_dir = ds_root / f"seed_{base_seed}"
                run_name = f"{preferred_arch}_{dataset_name}_full_seed{base_seed}"

                def model_fn() -> nn.Module:
                    return build_mlp_model(
                        preferred_arch=preferred_arch,
                        num_classes=num_classes,
                        in_channels=in_channels,
                        image_size=image_size,
                        hidden_dims=hidden_dims,
                        dropout=float(args.dropout),
                    )

                if args.tune_lr and len(args.lr_grid) > 0:
                    # 1) Tune LR on train_indices vs val_indices
                    train_ds_for_tuning = Subset(train_full, train_indices)
                    val_ds_for_tuning = Subset(eval_full, val_indices)

                    tune_dir = run_dir / "lr_tuning"
                    tune_result = tune_lr_mlp(
                        out_dir=tune_dir,
                        run_prefix=run_name,
                        model_fn=model_fn,
                        train_ds=train_ds_for_tuning,
                        val_ds=val_ds_for_tuning,
                        device=device,
                        model_seed=model_seed,
                        epochs=epochs,
                        batch_size=int(args.batch_size),
                        num_workers=int(args.num_workers),
                        lr_grid=args.lr_grid,
                        weight_decay=float(args.weight_decay),
                    )
                    best_lr = tune_result["best_lr"]
                else:
                    best_lr = float(args.lr)

                # 2) Train on full dataset (train_indices + val_indices) with selected LR
                all_indices = train_indices + val_indices
                train_full_ds = Subset(train_full, all_indices)
                monitor_ds = Subset(eval_full, all_indices)

                train_final_dir = run_dir / "final_train"
                train_final_dir.mkdir(parents=True, exist_ok=True)

                rc = RunConfig(
                    dataset=dataset_name,
                    subset="full",
                    base_seed=int(base_seed),
                    model_seed=int(model_seed),
                    split_seed=int(args.split_seed),
                    subset_seed=int(args.subset_seed),
                    val_size=int(args.val_size),
                    epochs=epochs,
                    batch_size=int(args.batch_size),
                    num_workers=int(args.num_workers),
                    optimizer={"type": "Adam", "lr": float(best_lr), "weight_decay": float(args.weight_decay)},
                    model={"arch": preferred_arch, "hidden_dims": list(hidden_dims), "dropout": float(args.dropout)},
                    device=str(device),
                )
                save_json(train_final_dir / "config.json", asdict(rc))

                train_full_after_tuning(
                    out_dir=train_final_dir,
                    run_name=run_name,
                    model_fn=model_fn,
                    train_full_ds=train_full_ds,
                    monitor_ds=monitor_ds,
                    test_loader=test_loader,
                    device=device,
                    model_seed=model_seed,
                    epochs=epochs,
                    batch_size=int(args.batch_size),
                    num_workers=int(args.num_workers),
                    lr=float(best_lr),
                    weight_decay=float(args.weight_decay),
                )


if __name__ == "__main__":
    main()
