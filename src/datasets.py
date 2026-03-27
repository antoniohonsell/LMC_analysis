
"""
datasets.py

Dataset utilities for the generalization_experiments repo.

Provides:
- Standard dataset creation for: CIFAR10, CIFAR100, MNIST, FashionMNIST
- A stratified (distribution-preserving) splitter that partitions a training set
  into K disjoint parts (K=2/3/4 or any K>=2), with per-class proportions as
  equal as possible across parts.

This module is designed to replace the repeated dataset/transforms logic that
currently appears in training scripts (e.g., CIFAR RandomCrop+Flip+Normalize). 
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset, Subset
import torchvision
import torchvision.transforms as transforms


# ---------------------------------------------------------------------
# Dataset stats (match existing CIFAR stats used in repo training scripts)
# ---------------------------------------------------------------------

DATASET_STATS: Dict[str, Dict[str, Any]] = {
    "CIFAR10": {
        "mean": (0.49139968, 0.48215841, 0.44653091),
        "std":  (0.24703223, 0.24348513, 0.26158784),
        "num_classes": 10,
        "in_channels": 3,
        "image_size": (32, 32),
    },
    "CIFAR100": {
        "mean": (0.50707516, 0.48654887, 0.44091784),
        "std":  (0.26733429, 0.25643846, 0.27615047),
        "num_classes": 100,
        "in_channels": 3,
        "image_size": (32, 32),
    },
    # Standard MNIST normalization constants (commonly used baseline).
    "MNIST": {
        "mean": (0.1307,),
        "std":  (0.3081,),
        "num_classes": 10,
        "in_channels": 1,
        "image_size": (28, 28),
    },
    # Fashion-MNIST (grayscale 28x28, 10 classes).
    # NOTE: mean/std below are commonly used defaults; you can recompute them if you want exact values.
    "FASHIONMNIST": {
        "mean": (0.2860,),
        "std":  (0.3530,),
        "num_classes": 10,
        "in_channels": 1,
        "image_size": (28, 28),
    },
}


# ---------------------------------------------------------------------
# Transforms
# ---------------------------------------------------------------------

def build_transforms(
    dataset: str,
    *,
    train: bool,
    augment: Optional[bool] = None,
    normalize: bool = True,
) -> transforms.Compose:
    """
    Create torchvision transform pipelines consistent with existing scripts.

    CIFAR:
      - train: RandomCrop(32, padding=4) + RandomHorizontalFlip + ToTensor + Normalize
      - eval:  ToTensor + Normalize

    MNIST:
      - default: ToTensor + Normalize
      - if augment=True: RandomAffine (small) + ToTensor + Normalize
        (kept conservative to avoid degrading MNIST by aggressive cropping)

    Args:
      dataset: "CIFAR10" | "CIFAR100" | "MNIST"
      train: whether to create train transforms
      augment: if None, defaults to True for CIFAR and False for MNIST
      normalize: apply Normalize(mean,std)

    Returns:
      torchvision.transforms.Compose
    """
    dataset = dataset.strip().upper()
    if dataset not in DATASET_STATS:
        raise ValueError(f"Unsupported dataset: {dataset} (supported: {sorted(DATASET_STATS)})")

    if augment is None:
        augment = (dataset in ("CIFAR10", "CIFAR100"))  # default matches repo CIFAR scripts

    ops: List[Any] = []

    if dataset in ("CIFAR10", "CIFAR100"):
        if train and augment:
            ops.extend([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
            ])
        ops.append(transforms.ToTensor())

    else:  # MNIST / FashionMNIST 
        if train and augment:
            ops.append(transforms.RandomAffine(
                 degrees=10,
                 translate=(0.05, 0.05),
                 scale=(0.95, 1.05),
            ))
        ops.append(transforms.ToTensor())

    if normalize:
        stats = DATASET_STATS[dataset]
        ops.append(transforms.Normalize(stats["mean"], stats["std"]))

    return transforms.Compose(ops)


# ---------------------------------------------------------------------
# Dataset creation
# ---------------------------------------------------------------------

def build_datasets(
    dataset: str,
    *,
    root: str = "./data",
    download: bool = True,
    augment_train: Optional[bool] = None,
    normalize: bool = True,
) -> Tuple[Dataset, Dataset, Dataset]:
    """
    Create (train_full, eval_full, test_ds) for a supported dataset.

    Why both train_full and eval_full?
      - Many training scripts use augmentation in training but want validation
        on the same underlying examples without augmentation.
      - This function mirrors that pattern by returning two dataset objects
        pointing to the *same split* (train=True) but different transforms.

    Returns:
      train_full: train split with optional augmentation
      eval_full:  train split without augmentation (for validation subsets)
      test_ds:    test split without augmentation
    """
    dataset = dataset.strip().upper()
    if dataset not in DATASET_STATS:
        raise ValueError(f"Unsupported dataset: {dataset} (supported: {sorted(DATASET_STATS)})")

    train_transform = build_transforms(dataset, train=True, augment=augment_train, normalize=normalize)
    eval_transform  = build_transforms(dataset, train=False, augment=False, normalize=normalize)

    if dataset == "CIFAR10":
        train_full = torchvision.datasets.CIFAR10(root=root, train=True, download=download, transform=train_transform)
        eval_full  = torchvision.datasets.CIFAR10(root=root, train=True, download=download, transform=eval_transform)
        test_ds    = torchvision.datasets.CIFAR10(root=root, train=False, download=download, transform=eval_transform)
    elif dataset == "CIFAR100":
        train_full = torchvision.datasets.CIFAR100(root=root, train=True, download=download, transform=train_transform)
        eval_full  = torchvision.datasets.CIFAR100(root=root, train=True, download=download, transform=eval_transform)
        test_ds    = torchvision.datasets.CIFAR100(root=root, train=False, download=download, transform=eval_transform)
    elif dataset == "MNIST":
        train_full = torchvision.datasets.MNIST(root=root, train=True, download=download, transform=train_transform)
        eval_full  = torchvision.datasets.MNIST(root=root, train=True, download=download, transform=eval_transform)
        test_ds    = torchvision.datasets.MNIST(root=root, train=False, download=download, transform=eval_transform)
    else:  # FASHIONMNIST
        train_full = torchvision.datasets.FashionMNIST(root=root, train=True, download=download, transform=train_transform)
        eval_full  = torchvision.datasets.FashionMNIST(root=root, train=True, download=download, transform=eval_transform)
        test_ds    = torchvision.datasets.FashionMNIST(root=root, train=False, download=download, transform=eval_transform)

    return train_full, eval_full, test_ds


def num_classes(dataset: str) -> int:
    dataset = dataset.strip().upper()
    if dataset not in DATASET_STATS:
        raise ValueError(f"Unsupported dataset: {dataset}")
    return int(DATASET_STATS[dataset]["num_classes"])


# ---------------------------------------------------------------------
# Balanced / stratified splitting
# ---------------------------------------------------------------------

def _extract_targets(ds: Dataset) -> np.ndarray:
    """
    Extract targets/labels for common torchvision datasets AND for Subset wrappers.

    Returns:
      numpy array of shape [len(ds)] containing integer class labels.
    """
    if isinstance(ds, Subset):
        base_targets = _extract_targets(ds.dataset)
        idx = np.asarray(ds.indices)
        return base_targets[idx]

    if hasattr(ds, "targets"):
        t = getattr(ds, "targets")
        if torch.is_tensor(t):
            return t.detach().cpu().numpy().astype(int)
        return np.asarray(t, dtype=int)

    for attr in ("labels", "train_labels", "test_labels"):
        if hasattr(ds, attr):
            t = getattr(ds, attr)
            if torch.is_tensor(t):
                return t.detach().cpu().numpy().astype(int)
            return np.asarray(t, dtype=int)

    raise AttributeError(
        "Cannot extract targets from this dataset. Expected .targets/.labels "
        "or a torchvision dataset / Subset thereof."
    )


@dataclass(frozen=True)
class SplitMeta:
    num_parts: int
    num_classes: int
    sizes: Tuple[int, ...]
    per_class_counts: Tuple[Tuple[int, ...], ...]  # shape: [num_parts][num_classes]


def split_dataset_stratified(
    dataset: Dataset,
    *,
    num_parts: int,
    seed: int = 0,
    exact: bool = False,
    shuffle_within_split: bool = True,
) -> Tuple[List[Subset], List[List[int]], SplitMeta]:
    """
    Split a dataset into K disjoint, stratified parts.

    Stratified here means: for each class c, its samples are divided as evenly
    as possible across the K parts. This preserves the class distribution in
    each part up to rounding.

    Args:
      dataset: any torchvision-like dataset or Subset, provided targets are extractable.
      num_parts: K (recommended 2/3/4; any integer >=2 is supported)
      seed: controls deterministic shuffling
      exact: if True, enforces *exactly equal* per-class counts across parts by
             dropping per-class remainders so count(class) is divisible by K.
             If False (default), uses all samples and distributes remainders.
      shuffle_within_split: shuffle final indices for each part

    Returns:
      subsets: list[Subset(dataset, indices_part_i)]
      indices: list[list[int]] indices into the provided dataset
      meta: SplitMeta with sizes and per-class counts
    """
    if num_parts < 2:
        raise ValueError("num_parts must be >= 2")
    rng = np.random.default_rng(seed)

    y = _extract_targets(dataset)
    n = len(y)
    all_pos = np.arange(n)

    classes = np.unique(y)
    num_classes_ = int(classes.size)

    parts: List[List[int]] = [[] for _ in range(num_parts)]

    for c in classes:
        cls_pos = all_pos[y == c]
        rng.shuffle(cls_pos)

        if exact:
            usable = (len(cls_pos) // num_parts) * num_parts
            cls_pos = cls_pos[:usable]

        chunks = np.array_split(cls_pos, num_parts)
        for i in range(num_parts):
            parts[i].extend(chunks[i].tolist())

    if shuffle_within_split:
        for i in range(num_parts):
            rng.shuffle(parts[i])

    subsets = [Subset(dataset, idxs) for idxs in parts]

    per_class_counts: List[Tuple[int, ...]] = []
    for i in range(num_parts):
        yi = y[np.asarray(parts[i], dtype=int)]
        counts = [int(np.sum(yi == c)) for c in classes]
        per_class_counts.append(tuple(counts))

    meta = SplitMeta(
        num_parts=num_parts,
        num_classes=num_classes_,
        sizes=tuple(len(p) for p in parts),
        per_class_counts=tuple(per_class_counts),
    )

    return subsets, parts, meta


def split_trainset_balanced(
    train_full: Dataset,
    eval_full: Optional[Dataset] = None,
    *,
    num_parts: int,
    seed: int = 0,
    exact: bool = False,
) -> Dict[str, Any]:
    """
    Convenience helper for the common pattern:
      - train_full: train split with augmentation
      - eval_full:  train split without augmentation (optional)

    Produces:
      - train_parts: list[Subset] (subsets of train_full)
      - eval_parts:  list[Subset] (subsets of eval_full, if provided)
      - indices:     list[list[int]] indices into train_full/eval_full
      - meta:        SplitMeta
    """
    train_parts, indices, meta = split_dataset_stratified(
        train_full, num_parts=num_parts, seed=seed, exact=exact
    )

    eval_parts = None
    if eval_full is not None:
        eval_parts = [Subset(eval_full, idxs) for idxs in indices]

    return {
        "train_parts": train_parts,
        "eval_parts": eval_parts,
        "indices": indices,
        "meta": meta,
    }


__all__ = [
    "DATASET_STATS",
    "build_transforms",
    "build_datasets",
    "num_classes",
    "split_dataset_stratified",
    "split_trainset_balanced",
    "SplitMeta",
]
