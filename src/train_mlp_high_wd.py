#!/usr/bin/env python3
"""
train_mlp_high_wd.py

Trains MLPs on MNIST subset A and subset B with strong weight decay.

Usage:
  python train_mlp_high_wd.py
  python train_mlp_high_wd.py --weight_decay 0.1 --epochs 50
"""

import argparse
import random
import time
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

import architectures
import datasets
import train_loop
import utils


def set_seed(seed: int) -> None:
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def seed_worker(worker_id: int) -> None:
    """Ensures dataloader workers are deterministically seeded."""
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def extract_targets(dataset):
    """Extract target labels from dataset."""
    targets = []
    for i in range(len(dataset)):
        _, target = dataset[i]
        targets.append(target)
    return targets


def stratified_train_val_split(
    targets: List[int],
    val_size: int,
    seed: int,
    num_classes: int,
) -> Tuple[List[int], List[int]]:
    """Stratified split of indices into train and validation."""
    rng = np.random.RandomState(seed)
    targets_arr = np.asarray(targets)
    train_indices_list = []
    val_indices_list = []

    for c in range(num_classes):
        class_indices = np.where(targets_arr == c)[0].tolist()
        rng.shuffle(class_indices)
        
        # Distribute proportionally to validation
        n_class = len(class_indices)
        n_val_class = max(1, int(val_size * n_class / len(targets)))
        
        val_indices_list.extend(class_indices[:n_val_class])
        train_indices_list.extend(class_indices[n_val_class:])

    rng.shuffle(train_indices_list)
    rng.shuffle(val_indices_list)

    # Trim to exact val_size
    if len(val_indices_list) > val_size:
        val_indices_list = val_indices_list[:val_size]
    if len(val_indices_list) < val_size and len(train_indices_list) > 0:
        move_count = min(val_size - len(val_indices_list), len(train_indices_list))
        val_indices_list.extend(train_indices_list[:move_count])
        train_indices_list = train_indices_list[move_count:]

    return train_indices_list, val_indices_list


def split_train_into_two_balanced_subsets(
    train_indices: List[int],
    targets: List[int],
    seed: int,
    num_classes: int,
) -> Tuple[List[int], List[int], dict]:
    """Split training indices into two balanced subsets A and B."""
    rng = np.random.RandomState(seed)
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


class SimpleMLP(nn.Module):
    """Simple MLP architecture."""
    def __init__(self, input_dim: int, num_classes: int, hidden_dims: Tuple[int, ...] = (512,)):
        super().__init__()
        dims = [input_dim] + list(hidden_dims) + [num_classes]
        layers: List[nn.Module] = []
        for i in range(len(dims) - 2):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(dims[-2], dims[-1]))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.view(x.size(0), -1)
        return self.net(x)


def train_mlp(
    subset_name: str,
    model: nn.Module,
    train_ds: Subset,
    val_ds: Subset,
    test_loader: DataLoader,
    device: torch.device,
    lr: float,
    weight_decay: float,
    epochs: int,
    batch_size: int,
    num_workers: int,
    seed: int,
    out_dir: Path,
) -> None:
    """Train a single MLP model."""
    out_dir.mkdir(parents=True, exist_ok=True)

    set_seed(seed)
    g_loader = torch.Generator().manual_seed(seed)

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
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    print(f"\n{'='*60}")
    print(f"Training MLP - Subset {subset_name}")
    print(f"{'='*60}")
    print(f"LR: {lr}, Weight Decay: {weight_decay}")
    print(f"Training samples: {len(train_ds)}, Validation samples: {len(val_ds)}")
    print(f"Epochs: {epochs}, Batch size: {batch_size}")

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
        save_dir=str(out_dir),
        run_name=f"mlp_subset_{subset_name}",
        save_every=epochs + 1,  # Only save best
        save_last=True,
    )
    t1 = time.time()

    print(f"\nTraining completed in {t1 - t0:.1f} seconds")
    print(f"Final val accuracy: {history['val_accuracy'][-1]:.4f}")
    print(f"Final val loss: {history['val_loss'][-1]:.4f}")

    # Save the model
    best_ckpt_path = out_dir / f"mlp_subset_{subset_name}_best.pth"
    if best_ckpt_path.exists():
        final_weights = torch.load(best_ckpt_path, map_location=device)
        model.load_state_dict(final_weights["state_dict"])
        torch.save(model.state_dict(), out_dir / f"mlp_subset_{subset_name}_weights.pt")
    else:
        # Save final model if best wasn't saved
        torch.save(model.state_dict(), out_dir / f"mlp_subset_{subset_name}_weights.pt")

    # Evaluate on test set
    model.eval()
    test_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            test_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

    test_loss /= total
    test_acc = correct / total
    print(f"Test accuracy: {test_acc:.4f}, Test loss: {test_loss:.4f}")

    # Save history
    torch.save(history, out_dir / f"history_subset_{subset_name}.pt")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train MLPs on MNIST subsets with strong weight decay")
    parser.add_argument("--data_root", type=str, default="./data", help="Root directory for datasets")
    parser.add_argument("--out_dir", type=str, default="./runs_mlp_high_wd", help="Output directory for trained models")
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=256, help="Batch size")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of dataloader workers")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-2, help="Weight decay (L2 regularization)")
    parser.add_argument("--val_size", type=int, default=5000, help="Validation set size")
    parser.add_argument("--split_seed", type=int, default=50, help="Seed for train/val split")
    parser.add_argument("--subset_seed", type=int, default=50, help="Seed for A/B subset split")
    parser.add_argument("--seed", type=int, default=0, help="Model training seed")
    parser.add_argument("--hidden_dims", type=str, default="512", help="Hidden dimensions (comma-separated)")

    args = parser.parse_args()

    # Parse hidden dimensions
    hidden_dims = tuple(int(x.strip()) for x in args.hidden_dims.split(",") if x.strip())

    device = utils.get_device()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Device: {device}")
    print(f"Output directory: {out_dir}")

    # Load MNIST dataset
    train_full, eval_full, test_ds = datasets.build_datasets(
        "MNIST",
        root=args.data_root,
        download=True,
        augment_train=False,
        normalize=True,
    )

    # Extract targets
    targets = extract_targets(train_full)

    # Create train/val split
    train_indices, val_indices = stratified_train_val_split(
        targets=targets,
        val_size=args.val_size,
        seed=args.split_seed,
        num_classes=10,
    )

    # Create A/B subsets from training indices
    subset_a_idx, subset_b_idx, subset_meta = split_train_into_two_balanced_subsets(
        train_indices=train_indices,
        targets=targets,
        seed=args.subset_seed,
        num_classes=10,
    )

    # Create datasets
    subset_a_ds = Subset(train_full, subset_a_idx)
    subset_b_ds = Subset(train_full, subset_b_idx)
    val_ds = Subset(eval_full, val_indices)

    # Create test loader
    test_loader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )

    print(f"\nDataset splits:")
    print(f"  Subset A: {len(subset_a_ds)} samples")
    print(f"  Subset B: {len(subset_b_ds)} samples")
    print(f"  Validation: {len(val_ds)} samples")
    print(f"  Test: {len(test_ds)} samples")

    # Train Subset A
    model_a = SimpleMLP(input_dim=784, num_classes=10, hidden_dims=hidden_dims)
    train_mlp(
        subset_name="A",
        model=model_a,
        train_ds=subset_a_ds,
        val_ds=val_ds,
        test_loader=test_loader,
        device=device,
        lr=args.lr,
        weight_decay=args.weight_decay,
        epochs=args.epochs,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        seed=args.seed,
        out_dir=out_dir,
    )

    # Train Subset B
    model_b = SimpleMLP(input_dim=784, num_classes=10, hidden_dims=hidden_dims)
    train_mlp(
        subset_name="B",
        model=model_b,
        train_ds=subset_b_ds,
        val_ds=val_ds,
        test_loader=test_loader,
        device=device,
        lr=args.lr,
        weight_decay=args.weight_decay,
        epochs=args.epochs,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        seed=args.seed + 1,
        out_dir=out_dir,
    )

    print(f"\n{'='*60}")
    print(f"Training completed. Models saved to: {out_dir}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
