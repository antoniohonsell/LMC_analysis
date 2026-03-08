#!/usr/bin/env python3
"""
compute_hessian_jacobian_sweep.py

Computes H (representation second moment) and Z (weight product) matrices for backward RWA analysis.

For each run directory:
  1. Load the model from model.pth
  2. Load the training data
  3. Compute H = E[h_a h_a^T] where h_a are activations from penultimate layer
  4. Compute Z = W^T W where W is the final layer weight matrix
  5. Save H.pt and Z.pt

This implements the "backward RWA" setup from the paper:
  - H_a = E[h_a(x_i) h_a(x_i)^T] = (1/N) R^T R (second moment of activations)
  - Z_a = W^T W (weight product from final layer)
  - RWA is H_a ∝ Z_a (their alignment)

Usage:
  export PYTHONPATH="$(pwd)"
  python CRH/compute_hessian_jacobian_sweep.py \
    --sweep_root runs_sweep_full/mnist_mlp_reg \
    --run_glob "**/final_train" \
    --batch_size 256
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# Assuming these modules exist in the repo
try:
    from architectures import build_model
    from datasets import build_datasets, DATASET_STATS
except ImportError:
    print("Error: Could not import architectures or datasets. Make sure PYTHONPATH is set.")
    raise


def load_config(run_dir: Path) -> Dict[str, Any]:
    """Load config.json from run directory."""
    cfg_path = run_dir / "config.json"
    if not cfg_path.exists():
        return {}
    with open(cfg_path, "r") as f:
        return json.load(f)


def load_model(run_dir: Path, config: Dict[str, Any]) -> nn.Module:
    """Load the trained model from checkpoint."""
    # Try multiple possible checkpoint names
    checkpoint_names = [
        "model.pth",
        "*_final.pth",
        "*_best.pth",
        "*_epoch40.pth",
        "*.pth",
    ]
    
    model_path = None
    for pattern in checkpoint_names:
        matches = list(run_dir.glob(pattern))
        if matches:
            model_path = matches[0]  # Take the first match
            break
    
    if model_path is None or not model_path.exists():
        raise FileNotFoundError(f"Could not find any checkpoint in {run_dir}")
    
    # Infer dataset and architecture from config or path
    dataset = config.get("dataset", "MNIST")
    arch = config.get("arch", "mnist_mlp_reg")  # Use lowercase default
    
    # Normalize architecture name to lowercase for compatibility
    arch = arch.lower()
    
    # Get num_classes from dataset stats
    num_classes = DATASET_STATS[dataset]["num_classes"]
    in_channels = DATASET_STATS[dataset]["in_channels"]
    
    # Build model
    model = build_model(arch, num_classes=num_classes, in_channels=in_channels)
    checkpoint = torch.load(model_path, map_location="cpu")
    
    # Extract the actual state_dict from checkpoint
    if isinstance(checkpoint, dict):
        if "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        else:
            state_dict = checkpoint
    else:
        state_dict = checkpoint
    
    model.load_state_dict(state_dict)
    
    print(f"  Loaded model from: {model_path.name}")
    return model.eval()


def load_training_data(run_dir: Path, config: Dict[str, Any], batch_size: int = 256) -> DataLoader:
    """Load training dataset for the run."""
    dataset = config.get("dataset", "MNIST")
    
    # Use build_datasets which returns (train_full, eval_full, test_ds)
    train_ds, eval_ds, test_ds = build_datasets(dataset, root="./data", download=False, 
                                                 augment_train=False, normalize=True)
    
    loader = DataLoader(train_ds, batch_size=batch_size, shuffle=False, num_workers=0)
    return loader


def compute_representation_matrix(model: nn.Module, data_loader: DataLoader, 
                                  device: str = "cpu") -> torch.Tensor:
    """
    Compute H = E[h_a h_a^T], the second moment of activations from penultimate layer.
    
    For an MLP, this is the Gram matrix of representations before the final layer.
    Use activations from the layer before the output layer.
    
    Returns shape (d, d) where d is the hidden dimension.
    """
    model = model.to(device)
    model.eval()
    
    # Hook to capture activations
    activations = []
    
    def hook_fn(module, input, output):
        # For linear layers, input is the activation
        if isinstance(module, nn.Linear):
            activations.append(input[0].detach().cpu())
    
    # Register hook on the last layer to capture its input (penultimate activations)
    last_layer = None
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            last_layer = module
    
    if last_layer is None:
        raise ValueError("Could not find Linear layer in model")
    
    handle = last_layer.register_forward_hook(hook_fn)
    
    print("  Computing activation Gram matrix...")
    with torch.no_grad():
        for batch_idx, (x, y) in enumerate(data_loader):
            x = x.to(device)
            _ = model(x)
            if batch_idx % 10 == 0:
                print(f"    Processed {batch_idx + 1} batches...")
    
    handle.remove()
    
    # Stack all activations and compute Gram matrix H = (1/N) R^T R
    R = torch.cat(activations, dim=0)  # (N, d)
    print(f"  Activation matrix shape: {R.shape}")
    
    # H = (1/N) R^T R (uncentered second moment, matching paper definition)
    H = (R.T @ R) / R.shape[0]
    print(f"  Representation Gram matrix shape: {H.shape}")
    return H


def compute_weight_product_matrix(model: nn.Module) -> torch.Tensor:
    """
    Compute Z = W^T W from the final layer weights.
    
    Returns shape (d, d) where d is the hidden dimension.
    """
    print("  Computing weight product matrix...")
    
    # Get the last linear layer (final layer)
    last_layer = None
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            last_layer = module
    
    if last_layer is None:
        raise ValueError("Could not find Linear layer in model")
    
    W = last_layer.weight.data  # (out_features, in_features)
    
    # Z = W^T W (shape: in_features x in_features)
    Z = W.T @ W
    print(f"  Weight product matrix shape: {Z.shape}")
    return Z.cpu()


def infer_dataset_from_path(run_dir: Path) -> str:
    """Infer dataset name from run directory path."""
    path_str = str(run_dir).lower()
    if "mnist" in path_str:
        return "MNIST"
    elif "cifar10" in path_str:
        return "CIFAR10"
    elif "cifar100" in path_str:
        return "CIFAR100"
    else:
        return "MNIST"  # default


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--sweep_root", type=str, required=True,
                   help="Root folder containing parameter sweep directories.")
    p.add_argument("--run_glob", type=str, default="**/final_train",
                   help="Glob pattern to match run directories.")
    p.add_argument("--batch_size", type=int, default=256,
                   help="Batch size for loading data.")
    p.add_argument("--device", type=str, default="mps" if torch.backends.mps.is_available() else "cpu",
                   help="Device to use (cpu, cuda, mps).")
    p.add_argument("--overwrite", action="store_true",
                   help="Overwrite existing H.pt/Z.pt files.")
    args = p.parse_args()
    sweep_root = Path(args.sweep_root).expanduser().resolve()
    if not sweep_root.exists():
        print(f"Error: sweep_root does not exist: {sweep_root}")
        return
    
    run_dirs = sorted(sweep_root.glob(args.run_glob))
    print(f"Found {len(run_dirs)} run directories")
    
    for idx, run_dir in enumerate(run_dirs):
        H_path = run_dir / "H.pt"
        Z_path = run_dir / "Z.pt"
        
        # Skip if already computed
        if H_path.exists() and Z_path.exists() and not args.overwrite:
            print(f"[{idx+1}/{len(run_dirs)}] SKIP {run_dir.name} (H/Z already exist)")
            continue
        
        print(f"\n[{idx+1}/{len(run_dirs)}] Processing {run_dir}")
        
        try:
            # Load config
            config = load_config(run_dir.parent)  # config usually at parent level
            
            # Infer dataset from path if not in config
            if "dataset" not in config:
                config["dataset"] = infer_dataset_from_path(run_dir)
            
            print(f"  Dataset: {config.get('dataset')}")
            print(f"  Architecture: {config.get('arch', 'MNISTMLPReg')}")
            
            # Load model
            print("  Loading model...")
            model = load_model(run_dir, config)
            
            # Load data
            print("  Loading training data...")
            data_loader = load_training_data(run_dir, config, batch_size=args.batch_size)
            
            # Compute H (representation Gram matrix) and Z (weight product)
            print("  Computing representation Gram matrix H = E[h_a h_a^T]...")
            H = compute_representation_matrix(model, data_loader, device=args.device)
            
            print("  Computing weight product matrix Z = W^T W...")
            Z = compute_weight_product_matrix(model)
            
            # Save
            print(f"  Saving H to {H_path}...")
            torch.save(H, H_path)
            print(f"  Saving Z to {Z_path}...")
            torch.save(Z, Z_path)
            print(f"  Done!")
            
        except Exception as e:
            print(f"  ERROR: {e}")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    main()
