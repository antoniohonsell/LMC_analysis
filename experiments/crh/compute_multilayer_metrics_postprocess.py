#!/usr/bin/env python3
"""
Post-process script to compute multi-layer CRH metrics for already-trained models.
This loads trained checkpoints and computes metrics without retraining.
"""

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any, Dict, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

_HERE = Path(__file__).resolve().parent
_SRC  = _HERE.parent.parent / "src"
for _p in (str(_SRC), str(_HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import architectures
import datasets
from train_track_crh_pah_mlp_v2 import (
    compute_crh_matrices_all_layers,
    find_all_linear_layers,
    flatten_pair,
)
from hz_metrics import compute_all_metrics


def set_seed(seed: int) -> None:
    import random
    random.seed(seed)
    import numpy as np
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def trace_remove(A: torch.Tensor) -> torch.Tensor:
    n = A.shape[0]
    tr = torch.trace(A)
    return A - (tr / float(n)) * torch.eye(n, dtype=A.dtype, device=A.device)


def fro_norm(A: torch.Tensor) -> float:
    return float(torch.linalg.norm(A, ord="fro").item())


def compute_metrics_for_checkpoint(
    model_path: Path,
    dataset_name: str,
    data_root: Path,
    device: torch.device,
    hidden_dims: tuple = (2048, 512),
    dropout: float = 0.0,
    label_smoothing: float = 0.05,
    k: int = 20,
    eig_tol: float = 1e-12,
    alpha_min: float = 0.25,
    alpha_max: float = 3.0,
    alpha_steps: int = 56,
    max_batches: Optional[int] = None,
    g_mode: str = "batch",
) -> Dict[str, Any]:
    """
    Load a trained model and compute multi-layer CRH metrics.
    
    Args:
        model_path: Path to the saved model checkpoint
        dataset_name: "MNIST" or "FASHIONMNIST"
        data_root: Path to data directory
        device: torch device
        hidden_dims: Hidden dimensions for MLP
        dropout: Dropout rate
        label_smoothing: Label smoothing rate
        k: Number of principal angles to compute
        eig_tol: Eigenvalue tolerance
        alpha_min, alpha_max, alpha_steps: Power-law fitting parameters
        max_batches: Max batches for metrics computation (None = all)
        g_mode: "batch" or "sample" gradient mode
    
    Returns:
        Dictionary with metrics for all layers
    """
    
    # Load model
    model = architectures.mnist_mlp_reg(
        hidden_dims=hidden_dims,
        dropout=dropout,
        label_smoothing=label_smoothing,
    )
    
    checkpoint = torch.load(model_path, map_location=device)
    if isinstance(checkpoint, dict) and "model" in checkpoint:
        model.load_state_dict(checkpoint["model"])
    else:
        model.load_state_dict(checkpoint)
    
    model = model.to(device)
    model.eval()
    
    # Load dataset
    train_data, val_data, _ = datasets.load(
        dataset_name,
        data_root,
        do_augment=False,  # No augmentation for metrics
    )
    
    val_loader = DataLoader(
        val_data,
        batch_size=128,
        shuffle=False,
        num_workers=0,
    )
    
    # Compute metrics using all layers
    all_layer_results = compute_crh_matrices_all_layers(
        model,
        val_loader,
        device,
        max_batches=max_batches,
        g_mode=g_mode,
    )
    
    # Define metrics computation function
    def M(A: torch.Tensor, B: torch.Tensor, allow_indefinite: bool) -> Dict[str, Any]:
        return compute_all_metrics(
            A, B,
            sym=True,
            k=k,
            eig_tol=eig_tol,
            alpha_min=alpha_min,
            alpha_max=alpha_max,
            alpha_steps=alpha_steps,
            allow_indefinite=allow_indefinite,
        )
    
    # Process results for each layer
    results = {}
    
    for layer_name, mats in all_layer_results.items():
        if layer_name == "__last__":
            continue
        
        Ha, Hb = mats["Ha"], mats["Hb"]
        HaM, HbM = mats["Ha_means"], mats["Hb_means"]
        WTW, WWT = mats["WTW"], mats["WWT"]
        Ga, Gb = mats["Ga"], mats["Gb"]
        
        # Compute alignment metrics
        m_Ha_WTW = M(Ha, WTW, allow_indefinite=False)
        m_HaM_WTW = M(HaM, WTW, allow_indefinite=False)
        m_Hb_WWT = M(Hb, WWT, allow_indefinite=False)
        m_HbM_WWT = M(HbM, WWT, allow_indefinite=False)
        m_Ga_WTW = M(Ga, WTW, allow_indefinite=False)
        m_Gb_WWT = M(Gb, WWT, allow_indefinite=False)
        
        # Trace-removed versions
        Ha_tr, WTW_tr = trace_remove(Ha), trace_remove(WTW)
        HaM_tr = trace_remove(HaM)
        Hb_tr, WWT_tr = trace_remove(Hb), trace_remove(WWT)
        HbM_tr = trace_remove(HbM)
        
        m_Ha_WTW_tr = M(Ha_tr, WTW_tr, allow_indefinite=True)
        m_HaM_WTW_tr = M(HaM_tr, WTW_tr, allow_indefinite=True)
        m_Hb_WWT_tr = M(Hb_tr, WWT_tr, allow_indefinite=True)
        m_HbM_WWT_tr = M(HbM_tr, WWT_tr, allow_indefinite=True)
        
        HaB, HbB = mats["Ha_between"], mats["Hb_between"]
        m_HaB_WTW = M(HaB, WTW, allow_indefinite=False)
        m_HbB_WWT = M(HbB, WWT, allow_indefinite=False)
        
        HaB_tr = trace_remove(HaB)
        HbB_tr = trace_remove(HbB)
        m_HaB_WTW_tr = M(HaB_tr, WTW_tr, allow_indefinite=True)
        m_HbB_WWT_tr = M(HbB_tr, WWT_tr, allow_indefinite=True)
        
        # Flatten metrics
        layer_metrics = {}
        layer_metrics.update({f"{layer_name}_{k}": v for k, v in flatten_pair("HaBetween_WTW", m_HaB_WTW).items()})
        layer_metrics.update({f"{layer_name}_{k}": v for k, v in flatten_pair("HbBetween_WWT", m_HbB_WWT).items()})
        layer_metrics.update({f"{layer_name}_{k}": v for k, v in flatten_pair("HaBetween_WTW_tr", m_HaB_WTW_tr).items()})
        layer_metrics.update({f"{layer_name}_{k}": v for k, v in flatten_pair("HbBetween_WWT_tr", m_HbB_WWT_tr).items()})
        
        layer_metrics.update({f"{layer_name}_{k}": v for k, v in flatten_pair("Ha_WTW", m_Ha_WTW).items()})
        layer_metrics.update({f"{layer_name}_{k}": v for k, v in flatten_pair("HaMeans_WTW", m_HaM_WTW).items()})
        layer_metrics.update({f"{layer_name}_{k}": v for k, v in flatten_pair("Hb_WWT", m_Hb_WWT).items()})
        layer_metrics.update({f"{layer_name}_{k}": v for k, v in flatten_pair("HbMeans_WWT", m_HbM_WWT).items()})
        
        layer_metrics.update({f"{layer_name}_{k}": v for k, v in flatten_pair("Ga_WTW", m_Ga_WTW).items()})
        layer_metrics.update({f"{layer_name}_{k}": v for k, v in flatten_pair("Gb_WWT", m_Gb_WWT).items()})
        
        layer_metrics.update({f"{layer_name}_{k}": v for k, v in flatten_pair("Ha_WTW_tr", m_Ha_WTW_tr).items()})
        layer_metrics.update({f"{layer_name}_{k}": v for k, v in flatten_pair("HaMeans_WTW_tr", m_HaM_WTW_tr).items()})
        layer_metrics.update({f"{layer_name}_{k}": v for k, v in flatten_pair("Hb_WWT_tr", m_Hb_WWT_tr).items()})
        layer_metrics.update({f"{layer_name}_{k}": v for k, v in flatten_pair("HbMeans_WWT_tr", m_HbM_WWT_tr).items()})
        
        # Add norms
        layer_metrics[f"{layer_name}_||Ha_tr||F"] = fro_norm(Ha_tr)
        layer_metrics[f"{layer_name}_||WTW_tr||F"] = fro_norm(WTW_tr)
        layer_metrics[f"{layer_name}_||Hb_tr||F"] = fro_norm(Hb_tr)
        layer_metrics[f"{layer_name}_||WWT_tr||F"] = fro_norm(WWT_tr)
        
        results[layer_name] = layer_metrics
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Compute multi-layer metrics for trained models")
    parser.add_argument("--model_path", type=str, required=True, help="Path to trained model checkpoint")
    parser.add_argument("--dataset", type=str, default="MNIST", choices=["MNIST", "FASHIONMNIST"])
    parser.add_argument("--data_root", type=str, default="data")
    parser.add_argument("--hidden_dims", type=str, default="2048,512", help="Hidden dimensions (comma-separated)")
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--label_smoothing", type=float, default=0.0)
    parser.add_argument("--output", type=str, help="Output JSON file (default: model_path_multilayer_metrics.json)")
    parser.add_argument("--max_batches", type=int, default=None, help="Max batches for metrics")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"])
    
    args = parser.parse_args()
    
    set_seed(args.seed)
    device = torch.device(args.device)
    
    # Parse hidden dims
    hidden_dims = tuple(int(x) for x in args.hidden_dims.split(","))
    
    # Set label smoothing based on dataset
    label_smoothing = args.label_smoothing
    if args.dataset == "FASHIONMNIST" and label_smoothing == 0.0:
        label_smoothing = 0.05
    
    print(f"Loading model from: {args.model_path}")
    print(f"Dataset: {args.dataset}, Hidden dims: {hidden_dims}")
    
    # Compute metrics
    metrics = compute_metrics_for_checkpoint(
        Path(args.model_path),
        args.dataset,
        Path(args.data_root),
        device,
        hidden_dims=hidden_dims,
        dropout=args.dropout,
        label_smoothing=label_smoothing,
        max_batches=args.max_batches,
    )
    
    # Save results
    output_path = args.output or f"{args.model_path}_multilayer_metrics.json"
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, "w") as f:
        json.dump(metrics, f, indent=2, default=str)
    
    print(f"✓ Metrics saved to: {output_path}")
    print(f"  Layers analyzed: {list(metrics.keys())}")
    print(f"  Metrics per layer: {len(metrics[list(metrics.keys())[0]])} metrics")


if __name__ == "__main__":
    main()
