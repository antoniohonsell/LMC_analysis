#!/usr/bin/env python3
"""
Compute CRH metrics for all linear layers in the model, not just the last one.

Usage:
    python compute_multilayer_metrics.py \
        --checkpoint runs_sweep_full/mnist_mlp_reg/.../final.pth \
        --dataset FASHIONMNIST \
        --data_root ./data \
        --output_dir ./multilayer_results
"""

import argparse
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from typing import Dict, Tuple, List, Optional, Any
import architectures
import datasets
import utils

from CRH.hz_metrics import compute_all_metrics


def fro_norm(A: torch.Tensor) -> float:
    return float(torch.linalg.norm(A, ord="fro").item())


def trace_remove(A: torch.Tensor) -> torch.Tensor:
    n = A.shape[0]
    tr = torch.trace(A)
    return A - (tr / float(n)) * torch.eye(n, dtype=A.dtype, device=A.device)


def _accum_second_moment(X: torch.Tensor) -> torch.Tensor:
    return X.T @ X


def get_all_linear_layers(model: nn.Module) -> List[Tuple[str, nn.Linear]]:
    """Get all linear layers in the model with their names."""
    layers = []
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            layers.append((name, module))
    return layers


def compute_crh_matrices_for_layer(
    model: nn.Module,
    loader,
    device: torch.device,
    linear_layer: nn.Linear,
    layer_name: str,
    max_batches: Optional[int] = None,
    g_mode: str = "batch",
) -> Dict[str, Any]:
    """
    Compute CRH matrices for a specific linear layer.
    
    Args:
        model: The neural network model
        loader: DataLoader for the data
        device: torch device
        linear_layer: The specific linear layer to analyze
        layer_name: Name of the layer for logging
        max_batches: Max batches to process
        g_mode: "batch" or "sample"
    
    Returns:
        Dictionary with Ha, Hb, Ga, Gb matrices and statistics
    """
    model = model.to(device)
    model.eval()

    W = linear_layer.weight.detach()  # (C, d)
    b = linear_layer.bias.detach() if linear_layer.bias is not None else None
    C, d = W.shape

    cache: Dict[str, torch.Tensor] = {}

    def hook_fn(module, inputs, outputs):
        cache["h_a"] = inputs[0].detach()
        cache["h_b_raw"] = outputs.detach()  # Capture layer output

    handle = linear_layer.register_forward_hook(hook_fn)

    Ha = torch.zeros((d, d), dtype=torch.float32, device=device)
    Hb = torch.zeros((C, C), dtype=torch.float32, device=device)
    Ga = torch.zeros((d, d), dtype=torch.float32, device=device)
    Gb = torch.zeros((C, C), dtype=torch.float32, device=device)

    # class sums for means (use 10 classes for label-based grouping)
    num_y_classes = 10
    sum_ha_by_class = torch.zeros((num_y_classes, d), dtype=torch.float32, device=device)
    sum_hb_by_class = torch.zeros((num_y_classes, C), dtype=torch.float32, device=device)
    count_by_class = torch.zeros((num_y_classes,), dtype=torch.float32, device=device)

    # stats
    sum_ha_n2 = 0.0
    sum_ha_n2_sq = 0.0
    n_ha = 0

    sum_gb_n2 = 0.0
    sum_gb_n2_sq = 0.0
    n_gb = 0

    n_samples = 0
    n_batches_used = 0

    with torch.no_grad():
        for bi, (x, y) in enumerate(loader):
            if max_batches is not None and bi >= int(max_batches):
                break

            x = x.to(device)
            y = y.to(device)

            cache.clear()
            logits = model(x)       # output shape depends on layer position
            h_a = cache["h_a"]      # input to the layer
            h_b_raw = cache.get("h_b_raw", logits)  # output of the layer

            # h_b is the pre-activation (before bias), so we use the layer output directly
            h_b = h_b_raw

            B = x.size(0)
            n_samples += B

            Ha += _accum_second_moment(h_a)
            Hb += _accum_second_moment(h_b)

            # class means accum
            onehot = F.one_hot(y, num_classes=num_y_classes).to(dtype=torch.float32)
            sum_ha_by_class += onehot.T @ h_a
            sum_hb_by_class += onehot.T @ h_b
            count_by_class += onehot.sum(dim=0)

            # Assume assumptions for ||h_a||^2
            ha_n2 = h_a.pow(2).sum(dim=1)
            sum_ha_n2 += float(ha_n2.sum().item())
            sum_ha_n2_sq += float((ha_n2 * ha_n2).sum().item())
            n_ha += int(B)

            # Gradient computation - only valid for last layer with cross-entropy
            # For intermediate layers, use identity as proxy (gradient approximation)
            is_last_layer = (C == 10)  # Last layer has output dim 10
            
            if is_last_layer:
                p = torch.softmax(logits, dim=1)
                y_oh = F.one_hot(y, num_classes=C).to(dtype=p.dtype)
                gb_sample = (p - y_oh)
            else:
                # For intermediate layers, use ReLU proxy: max(0, h_b)
                # This is an approximation
                gb_sample = torch.relu(h_b) / (torch.relu(h_b).mean(dim=0, keepdim=True) + 1e-8)

            if g_mode == "sample":
                Gb += _accum_second_moment(gb_sample)
                ga_sample = gb_sample @ W
                Ga += _accum_second_moment(ga_sample)

                gb_n2 = gb_sample.pow(2).sum(dim=1)
                sum_gb_n2 += float(gb_n2.sum().item())
                sum_gb_n2_sq += float((gb_n2 * gb_n2).sum().item())
                n_gb += int(B)
            else:
                gb_batch = gb_sample.mean(dim=0)
                ga_batch = gb_batch @ W
                Gb += torch.outer(gb_batch, gb_batch)
                Ga += torch.outer(ga_batch, ga_batch)

                gb_n2 = float(gb_batch.pow(2).sum().item())
                sum_gb_n2 += gb_n2
                sum_gb_n2_sq += gb_n2 * gb_n2
                n_gb += 1
                n_batches_used += 1

    handle.remove()

    if n_samples == 0:
        raise RuntimeError(f"No samples processed for layer {layer_name}")

    Ha = (Ha / float(n_samples)).detach().cpu()
    Hb = (Hb / float(n_samples)).detach().cpu()

    if g_mode == "sample":
        Ga = (Ga / float(n_samples)).detach().cpu()
        Gb = (Gb / float(n_samples)).detach().cpu()
    else:
        denom = float(max(1, n_batches_used))
        Ga = (Ga / denom).detach().cpu()
        Gb = (Gb / denom).detach().cpu()

    # class means
    count_cpu = count_by_class.detach().cpu()
    sum_ha_cpu = sum_ha_by_class.detach().cpu()
    sum_hb_cpu = sum_hb_by_class.detach().cpu()

    count_safe = torch.clamp(count_cpu, min=1.0)
    mu_a = sum_ha_cpu / count_safe.unsqueeze(1)
    mu_b = sum_hb_cpu / count_safe.unsqueeze(1)

    Ha_means = (mu_a.T @ mu_a)
    Hb_means = (mu_b.T @ mu_b)

    # between-class
    p = count_cpu / count_cpu.sum()
    mu_a_global = (p.unsqueeze(1) * mu_a).sum(dim=0)
    mu_b_global = (p.unsqueeze(1) * mu_b).sum(dim=0)

    Ha_between = torch.zeros_like(Ha)
    Hb_between = torch.zeros_like(Hb)
    for c in range(num_y_classes):
        ha_c_centered = mu_a[c] - mu_a_global
        hb_c_centered = mu_b[c] - mu_b_global
        Ha_between += p[c] * torch.outer(ha_c_centered, ha_c_centered)
        Hb_between += p[c] * torch.outer(hb_c_centered, hb_c_centered)

    WTW = (W.T @ W).detach().cpu()  # W^T @ W -> (d, d)
    WWT = (W @ W.T).detach().cpu()  # W @ W^T -> (C, C)

    # Compute assumption stats
    def _finalize_stats(sum_x, sum_x2, n):
        import math
        if n <= 1:
            return {"mean": float("nan"), "std": float("nan"), "cv": float("nan")}
        mean = sum_x / n
        var = max(0.0, (sum_x2 / n) - mean * mean)
        std = math.sqrt(var)
        cv = std / mean if mean > 0 else float("nan")
        return {"mean": float(mean), "std": float(std), "cv": float(cv)}

    return {
        "Ha": Ha,
        "Hb": Hb,
        "Ga": Ga,
        "Gb": Gb,
        "Ha_means": Ha_means,
        "Hb_means": Hb_means,
        "Ha_between": Ha_between,
        "Hb_between": Hb_between,
        "WTW": WTW,
        "WWT": WWT,
        "assumption_stats": {
            "ha_norm2": _finalize_stats(sum_ha_n2, sum_ha_n2_sq, n_ha),
            "gb_norm2": _finalize_stats(sum_gb_n2, sum_gb_n2_sq, n_gb),
            "g_mode": g_mode,
            "n_samples": n_samples,
            "n_batches_for_g": n_batches_used if g_mode == "batch" else n_samples,
        },
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=str, default=None, help="Path to trained model checkpoint (if None, uses latest available)")
    p.add_argument("--run_dir", type=str, default=None, help="Path to run directory (alternative to --checkpoint)")
    p.add_argument("--dataset", type=str, default="FASHIONMNIST", choices=["MNIST", "FASHIONMNIST"])
    p.add_argument("--data_root", type=str, default="./data")
    p.add_argument("--output_dir", type=str, required=True, help="Output directory for results")
    p.add_argument("--max_batches", type=int, default=200, help="Max batches to process")
    p.add_argument("--g_mode", type=str, default="batch", choices=["batch", "sample"])
    p.add_argument("--k", type=int, default=10, help="Subspace dimension for Hz metrics")
    args = p.parse_args()

    device = utils.get_device()
    print(f"Using device: {device}")

    # Determine checkpoint path
    checkpoint_path = args.checkpoint
    if checkpoint_path is None and args.run_dir is not None:
        # Find latest checkpoint in run directory
        run_dir = Path(args.run_dir)
        checkpoints = sorted(run_dir.glob("ckpt_epoch*.pth"))
        if checkpoints:
            checkpoint_path = str(checkpoints[-1])  # Latest checkpoint
            print(f"No checkpoint specified, using latest: {checkpoint_path}")
        else:
            raise FileNotFoundError(f"No checkpoints found in {run_dir}")
    
    if checkpoint_path is None:
        raise ValueError("Either --checkpoint or --run_dir must be provided")
    
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    # Load model
    print(f"Loading checkpoint from: {checkpoint_path}")
    config_path = checkpoint_path.parent / "config.json"
    with open(config_path) as f:
        config = json.load(f)

    num_classes = 10
    in_channels = 1
    input_shape = (1, 28, 28)
    
    model = architectures.build_model(
        config["arch"],
        num_classes=num_classes,
        in_channels=in_channels,
        input_shape=input_shape,
        hidden_dims=tuple(config["hidden_dims"]),
        dropout=float(config["dropout"]),
    ).to(device)

    checkpoint = torch.load(str(checkpoint_path), map_location=device)
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        model.load_state_dict(checkpoint["state_dict"])
    else:
        model.load_state_dict(checkpoint)

    # Load data
    dataset_name = args.dataset.upper()
    stats = datasets.DATASET_STATS[dataset_name]
    train_full, eval_full, _test_ds = datasets.build_datasets(
        dataset_name, root=args.data_root, download=True, augment_train=False, normalize=True
    )

    from torch.utils.data import DataLoader
    loader = DataLoader(
        train_full,
        batch_size=256,
        shuffle=False,
        num_workers=0,
        pin_memory=(device.type == "cuda"),
    )

    # Get all linear layers
    print("\nLinear layers in model:")
    linear_layers = get_all_linear_layers(model)
    for i, (name, layer) in enumerate(linear_layers):
        print(f"  {i}: {name} - {layer.in_features} -> {layer.out_features}")

    # Compute metrics for each layer
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results = {}
    for layer_name, layer in linear_layers:
        print(f"\nComputing metrics for layer: {layer_name}")
        mats = compute_crh_matrices_for_layer(
            model, loader, device, layer,
            layer_name=layer_name,
            max_batches=args.max_batches,
            g_mode=args.g_mode
        )

        Ha, Hb = mats["Ha"], mats["Hb"]
        Ha_tr = trace_remove(Ha)
        Hb_tr = trace_remove(Hb)
        WTW = mats["WTW"]
        WWT = mats["WWT"]

        # Compute Hz metrics
        metrics = compute_all_metrics(
            Ha, WTW,
            sym=True,
            k=int(args.k),
            eig_tol=1e-12,
            alpha_min=0.0,
            alpha_max=3.0,
            alpha_steps=121,
        )

        results[layer_name] = {
            "Ha_shape": list(Ha.shape),
            "Hb_shape": list(Hb.shape),
            "Ha_Frobenius": float(fro_norm(Ha)),
            "Hb_Frobenius": float(fro_norm(Hb)),
            "Ha_tr_Frobenius": float(fro_norm(Ha_tr)),
            "Hb_tr_Frobenius": float(fro_norm(Hb_tr)),
            "metrics": metrics,
        }

        print(f"  ✓ Ha_WTW eps_lin: {metrics.get('eps_lin', 'N/A'):.4f}")
        print(f"  ✓ Ha_WTW cosine_sim_fro: {metrics.get('cosine_sim_fro', 'N/A'):.4f}")

    # Save results
    output_path = output_dir / "multilayer_metrics.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n✓ Results saved to: {output_path}")


if __name__ == "__main__":
    main()
