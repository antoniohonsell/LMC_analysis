import os
import random
import argparse
import sys
from collections import Counter
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_SRC  = _HERE.parent.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Subset

import old_architectures.resnet18_arch_BatchNorm as resnet18_arch_BatchNorm
import old_architectures.resnet20_arch_BatchNorm as resnet20_arch_BatchNorm
import old_architectures.resnet20_arch_LayerNorm as resnet20_arch_LayerNorm
import train_loop
import utils


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


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # Determinism knobs (best-effort; some backends/ops may still be nondeterministic)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def seed_worker(worker_id: int):
    # Ensures dataloader workers are deterministically seeded
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def lr_lambda(epoch: int):
    # Same schedule as main_non_disjoint.py (CIFAR-style step drops)
    if epoch < 80 - 1:
        return 1.0
    elif epoch < 120 - 1:
        return 0.1
    else:
        return 0.01


def stratified_train_val_split(targets, val_size: int, seed: int, num_classes: int):
    """
    Deterministic stratified split: val gets ~val_size samples total, balanced across classes.
    Returns: (train_indices, val_indices) indices into the full train dataset.
    """
    rng = np.random.default_rng(seed)
    targets = np.asarray(targets)

    all_indices = np.arange(len(targets))
    per_class = {c: all_indices[targets == c].tolist() for c in range(num_classes)}
    for c in range(num_classes):
        rng.shuffle(per_class[c])

    # Allocate val counts per class
    base = val_size // num_classes
    rem = val_size % num_classes
    val_counts = {c: base + (1 if c < rem else 0) for c in range(num_classes)}

    val_indices = []
    train_indices = []
    for c in range(num_classes):
        k = val_counts[c]
        val_indices.extend(per_class[c][:k])
        train_indices.extend(per_class[c][k:])

    rng.shuffle(val_indices)
    rng.shuffle(train_indices)
    return train_indices, val_indices


def split_train_into_two_balanced_subsets(train_indices, targets, seed: int, num_classes: int):
    """
    Given train_indices into the full dataset, return two disjoint balanced subsets A/B.
    Balanced means: each subset has the same number per class, and classes are equally represented.
    Uses all possible samples subject to exact balance constraints.
    """
    rng = np.random.default_rng(seed)
    targets = np.asarray(targets)

    train_indices = np.asarray(train_indices)
    per_class = {c: train_indices[targets[train_indices] == c].tolist() for c in range(num_classes)}
    for c in range(num_classes):
        rng.shuffle(per_class[c])

    # To be exactly balanced across classes and between subsets, use:
    # k = min class count in train, then make it even so it can split into 2 equal halves.
    min_count = min(len(per_class[c]) for c in range(num_classes))
    k = (min_count // 2) * 2  # largest even <= min_count
    half = k // 2

    subset_a = []
    subset_b = []
    for c in range(num_classes):
        cls = per_class[c][:k]          # truncate to common even count
        subset_a.extend(cls[:half])
        subset_b.extend(cls[half:])

    rng.shuffle(subset_a)
    rng.shuffle(subset_b)
    return subset_a, subset_b, {
        "per_class_used": half,
        "dropped_per_class": {c: len(per_class[c]) - k for c in range(num_classes)},
    }


def class_counts(indices, targets, num_classes: int):
    targets = np.asarray(targets)
    ctr = Counter(targets[np.asarray(indices)].tolist())
    return [ctr.get(c, 0) for c in range(num_classes)]


def build_model(model_name: str, num_classes: int):
    if model_name == "resnet18":
        return resnet18_arch_BatchNorm.resnet_18_cifar(num_classes=num_classes)
    if model_name == "resnet20":
        # Keep parity with main_non_disjoint.py (LayerNorm variant)
        return resnet20_arch_LayerNorm.resnet20(num_classes=num_classes)
        # If you want BatchNorm ResNet20 instead, swap to:
        # return resnet20_arch_BatchNorm.resnet20(num_classes=num_classes)
    raise ValueError(f"Unsupported model: {model_name}")


def build_optimizer(model: torch.nn.Module, lr: float):
    """Match main_non_disjoint.py: no weight decay for 1D params (bias/norm scale)."""
    decay, no_decay = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        # no weight decay for normalization params and biases
        if p.ndim == 1 or name.endswith(".bias") or "norm" in name or "bn" in name or "ln" in name:
            no_decay.append(p)
        else:
            decay.append(p)

    return optim.SGD(
        [
            {"params": decay, "weight_decay": 5e-4},
            {"params": no_decay, "weight_decay": 0.0},
        ],
        lr=lr,
        momentum=0.9,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="CIFAR10")
    parser.add_argument("--model", type=str, default="resnet20")
    parser.add_argument("--seeds", nargs="+", type=int, default=list(range(0, 2)))
    parser.add_argument("--split_seed", type=int, default=50)        # fixed train/val split across all runs
    parser.add_argument("--subset_seed", type=int, default=None)     # fixed A/B membership; default = split_seed
    parser.add_argument("--epochs", type=int, default=128)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--out_dir", type=str, default=None)
    parser.add_argument("--val_size", type=int, default=5000)
    args = parser.parse_args()

    if args.dataset not in DATASET_STATS:
        raise ValueError(f"Unsupported dataset: {args.dataset}")
    stats = DATASET_STATS[args.dataset]

    if args.subset_seed is None:
        args.subset_seed = args.split_seed

    if args.out_dir is None:
        args.out_dir = f"./runs_{args.model}_{args.dataset}_disjoint"

    device = utils.get_device()
    os.makedirs(args.out_dir, exist_ok=True)

    normalize = transforms.Normalize(stats["mean"], stats["std"])

    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ])

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        normalize,
    ])

    # Use separate dataset objects so validation has no augmentation
    if args.dataset == "CIFAR10":
        train_full = torchvision.datasets.CIFAR10(root="./data", train=True, download=True, transform=train_transform)
        eval_full = torchvision.datasets.CIFAR10(root="./data", train=True, download=True, transform=test_transform)
        test_ds = torchvision.datasets.CIFAR10(root="./data", train=False, download=True, transform=test_transform)
    else:  # CIFAR100
        train_full = torchvision.datasets.CIFAR100(root="./data", train=True, download=True, transform=train_transform)
        eval_full = torchvision.datasets.CIFAR100(root="./data", train=True, download=True, transform=test_transform)
        test_ds = torchvision.datasets.CIFAR100(root="./data", train=False, download=True, transform=test_transform)

    targets = train_full.targets  # length 50k
    num_classes = stats["num_classes"]

    # Stratified train/val split (balanced per class)
    train_indices, val_indices = stratified_train_val_split(
        targets=targets,
        val_size=args.val_size,
        seed=args.split_seed,
        num_classes=num_classes,
    )

    # Two disjoint balanced subsets of the TRAIN indices
    subset_a_idx, subset_b_idx, subset_meta = split_train_into_two_balanced_subsets(
        train_indices=train_indices,
        targets=targets,
        seed=args.subset_seed,
        num_classes=num_classes,
    )

    # Save split indices for reproducibility
    split_path = os.path.join(
        args.out_dir,
        f"indices_{args.dataset}_splitseed{args.split_seed}_subsetseed{args.subset_seed}_val{args.val_size}.pt",
    )
    if not os.path.exists(split_path):
        torch.save(
            {
                "dataset": args.dataset,
                "split_seed": args.split_seed,
                "subset_seed": args.subset_seed,
                "val_size": args.val_size,
                "train_indices": train_indices,
                "val_indices": val_indices,
                "subset_a_indices": subset_a_idx,
                "subset_b_indices": subset_b_idx,
                "subset_meta": subset_meta,
            },
            split_path,
        )

    # Build datasets
    subset_a = Subset(train_full, subset_a_idx)
    subset_b = Subset(train_full, subset_b_idx)
    val_ds = Subset(eval_full, val_indices)

    # Sanity prints (class-balanced?)
    a_counts = class_counts(subset_a_idx, targets, num_classes=num_classes)
    b_counts = class_counts(subset_b_idx, targets, num_classes=num_classes)
    v_counts = class_counts(val_indices, targets, num_classes=num_classes)
    print("Subset A per-class counts:", a_counts)
    print("Subset B per-class counts:", b_counts)
    print("Val     per-class counts:", v_counts)
    print(
        f"Subset A size: {len(subset_a_idx)} | "
        f"Subset B size: {len(subset_b_idx)} | "
        f"Val size: {len(val_indices)}"
    )

    # Test loader never shuffled (created for parity with main_non_disjoint.py)
    test_loader = torch.utils.data.DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )

    # Shared val loader (no shuffle)
    val_loader = torch.utils.data.DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )

    for seed in args.seeds:
        print(f"\n==============================\nRunning seed = {seed}\n==============================")

        run_dir = os.path.join(args.out_dir, f"seed_{seed}")
        os.makedirs(run_dir, exist_ok=True)

        for subset_name, subset_ds, model_offset in [
            ("A", subset_a, 1),
            ("B", subset_b, 2),
        ]:
            model_dir = os.path.join(run_dir, f"subset_{subset_name}")
            os.makedirs(model_dir, exist_ok=True)

            # Deterministic but distinct seed stream per model
            model_seed = seed * 1000 + model_offset
            set_seed(model_seed)
            g_loader = torch.Generator().manual_seed(model_seed)

            train_loader = torch.utils.data.DataLoader(
                subset_ds,
                batch_size=args.batch_size,
                shuffle=True,
                num_workers=args.num_workers,
                worker_init_fn=seed_worker if args.num_workers > 0 else None,
                generator=g_loader,
                pin_memory=(device.type == "cuda"),
            )

            model = build_model(args.model, num_classes=num_classes).to(device)
            criterion = nn.CrossEntropyLoss()
            optimizer = build_optimizer(model, lr=args.lr)
            scheduler = LambdaLR(optimizer, lr_lambda=lr_lambda)

            history = train_loop.train(
                model=model,
                criterion=criterion,
                optimizer=optimizer,
                scheduler=scheduler,
                train_loader=train_loader,
                val_loader=val_loader,
                epochs=args.epochs,
                device=device,
                save_dir=model_dir,
                run_name=f"{args.model}_{args.dataset}_seed{seed}_subset{subset_name}",
                save_every=1,
                save_last=True,
            )

            torch.save(
                {
                    "base_seed": seed,
                    "model_seed": model_seed,
                    "subset": subset_name,
                    "dataset": args.dataset,
                    "model": args.model,
                    "history": history,
                },
                os.path.join(model_dir, "history.pt"),
            )

        # Optional: evaluate on test after training (you already created test_loader)
        # Add a small test() helper similar to validate() if needed.


if __name__ == "__main__":
    main()
