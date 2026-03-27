#!/usr/bin/env python3
# CRH/train_track_crh_pah_mlp_v2.py

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

_HERE = Path(__file__).resolve().parent
_SRC  = _HERE.parent.parent / "src"
for _p in (str(_SRC), str(_HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

import architectures
import datasets
import train_loop
import utils

from hz_metrics import compute_all_metrics


def save_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True), encoding="utf-8")


def set_seed(seed: int) -> None:
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def seed_worker(worker_id: int) -> None:
    import random
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def parse_int_list(s: str) -> Tuple[int, ...]:
    items = [x.strip() for x in s.split(",") if x.strip()]
    if not items:
        raise ValueError("--hidden_dims must be a comma-separated list like '2048,512'")
    return tuple(int(x) for x in items)


def trace_remove(A: torch.Tensor) -> torch.Tensor:
    # A must be square (n x n)
    n = A.shape[0]
    tr = torch.trace(A)
    return A - (tr / float(n)) * torch.eye(n, dtype=A.dtype, device=A.device)


def fro_norm(A: torch.Tensor) -> float:
    return float(torch.linalg.norm(A, ord="fro").item())


def find_last_linear(model: nn.Module) -> nn.Linear:
    last = None
    for m in model.modules():
        if isinstance(m, nn.Linear):
            last = m
    if last is None:
        raise ValueError("Could not find any nn.Linear layer in the model.")
    return last


def find_all_linear_layers(model: nn.Module) -> List[Tuple[str, nn.Linear]]:
    """Find all linear layers in the model with their names."""
    layers = []
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            layers.append((name, module))
    if not layers:
        raise ValueError("Could not find any nn.Linear layer in the model.")
    return layers


@torch.no_grad()
def _accum_second_moment(X: torch.Tensor) -> torch.Tensor:
    return X.T @ X


@dataclass
class MomentStats:
    mean: float
    std: float
    cv: float


def _finalize_stats(sum_x: float, sum_x2: float, n: int) -> MomentStats:
    if n <= 1:
        return MomentStats(mean=float("nan"), std=float("nan"), cv=float("nan"))
    mean = sum_x / n
    var = max(0.0, (sum_x2 / n) - mean * mean)
    std = math.sqrt(var)
    cv = std / mean if mean > 0 else float("nan")
    return MomentStats(mean=float(mean), std=float(std), cv=float(cv))


@torch.no_grad()
def compute_crh_matrices(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    *,
    max_batches: Optional[int] = None,
    g_mode: str = "batch",  # "batch" recommended for theorem-like Gb
) -> Dict[str, Any]:
    """
    Computes:
      - Ha = E[h_a h_a^T], Hb = E[h_b h_b^T]
      - WTW = W^T W, WWT = W W^T
      - Ga, Gb from either batch-gradient vectors (default) or per-sample
      - Class-means matrices:
          mu_a[c] = E[h_a | y=c], M_a rows = mu_a => Ha_means = M_a^T M_a
          mu_b[c] = E[h_b | y=c], M_b rows = mu_b => Hb_means = M_b^T M_b
      - Assumption-1 proxies:
          CV of ||h_a||^2 across samples
          CV of ||g_b||^2 across batches (or samples if g_mode="sample")
    """
    model = model.to(device)
    model.eval()

    last = find_last_linear(model)
    W = last.weight.detach()  # (C, d)
    b = last.bias.detach() if last.bias is not None else None
    C, d = W.shape

    cache: Dict[str, torch.Tensor] = {}

    def hook_fn(module, inputs, outputs):
        cache["h_a"] = inputs[0].detach()

    handle = last.register_forward_hook(hook_fn)

    Ha = torch.zeros((d, d), dtype=torch.float32, device=device)
    Hb = torch.zeros((C, C), dtype=torch.float32, device=device)
    Ga = torch.zeros((d, d), dtype=torch.float32, device=device)
    Gb = torch.zeros((C, C), dtype=torch.float32, device=device)

    # class sums for means
    sum_ha_by_class = torch.zeros((C, d), dtype=torch.float32, device=device)
    sum_hb_by_class = torch.zeros((C, C), dtype=torch.float32, device=device)
    count_by_class = torch.zeros((C,), dtype=torch.float32, device=device)

    # stats
    sum_ha_n2 = 0.0
    sum_ha_n2_sq = 0.0
    n_ha = 0

    sum_gb_n2 = 0.0
    sum_gb_n2_sq = 0.0
    n_gb = 0

    n_samples = 0
    n_batches_used = 0

    for bi, (x, y) in enumerate(loader):
        if max_batches is not None and bi >= int(max_batches):
            break

        x = x.to(device)
        y = y.to(device)

        cache.clear()
        logits = model(x)       # (B, C)
        h_a = cache["h_a"]      # (B, d)
        if b is not None:
            h_b = logits - b.unsqueeze(0)
        else:
            h_b = logits

        B = x.size(0)
        n_samples += B

        Ha += _accum_second_moment(h_a)
        Hb += _accum_second_moment(h_b)

        # class means accum
        onehot = F.one_hot(y, num_classes=C).to(dtype=torch.float32)  # (B,C)
        sum_ha_by_class += onehot.T @ h_a                    # (C,d)
        sum_hb_by_class += onehot.T @ h_b                    # (C,C)
        count_by_class += onehot.sum(dim=0)

        # Assumption-1 proxy for ||h_a||^2
        ha_n2 = h_a.pow(2).sum(dim=1)
        sum_ha_n2 += float(ha_n2.sum().item())
        sum_ha_n2_sq += float((ha_n2 * ha_n2).sum().item())
        n_ha += int(B)

        # g_b (cross-entropy) wrt logits
        p = torch.softmax(logits, dim=1)
        y_oh = F.one_hot(y, num_classes=C).to(dtype=p.dtype)
        gb_sample = (p - y_oh)  # (B,C)

        if g_mode == "sample":
            Gb += _accum_second_moment(gb_sample)
            ga_sample = gb_sample @ W  # (B,d)
            Ga += _accum_second_moment(ga_sample)

            gb_n2 = gb_sample.pow(2).sum(dim=1)
            sum_gb_n2 += float(gb_n2.sum().item())
            sum_gb_n2_sq += float((gb_n2 * gb_n2).sum().item())
            n_gb += int(B)

        else:
            gb_batch = gb_sample.mean(dim=0)      # (C,)
            ga_batch = gb_batch @ W               # (d,)
            Gb += torch.outer(gb_batch, gb_batch)
            Ga += torch.outer(ga_batch, ga_batch)

            gb_n2 = float(gb_batch.pow(2).sum().item())
            sum_gb_n2 += gb_n2
            sum_gb_n2_sq += gb_n2 * gb_n2
            n_gb += 1
            n_batches_used += 1

    handle.remove()

    if n_samples == 0:
        raise RuntimeError("No samples processed while computing CRH matrices.")

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

    # avoid division by zero (shouldn't happen on MNIST/FMNIST)
    count_safe = torch.clamp(count_cpu, min=1.0)
    mu_a = sum_ha_cpu / count_safe.unsqueeze(1)  # (C,d)
    mu_b = sum_hb_cpu / count_safe.unsqueeze(1)  # (C,C)

    Ha_means = (mu_a.T @ mu_a)                   # (d,d)  = M_a^T M_a
    Hb_means = (mu_b.T @ mu_b)                   # (C,C)

    # NEW: centered / between-class (probability-weighted)
    p = count_cpu / count_cpu.sum()                       # (C,)
    mu_a_bar = (p.unsqueeze(0) @ mu_a).squeeze(0)         # (d,)
    mu_b_bar = (p.unsqueeze(0) @ mu_b).squeeze(0)         # (C,)

    mu_a_c = mu_a - mu_a_bar.unsqueeze(0)                 # (C,d)
    mu_b_c = mu_b - mu_b_bar.unsqueeze(0)                 # (C,C)

    # between-class covariance (weighted)
    sqrtp = torch.sqrt(p).unsqueeze(1)                    # (C,1)
    Ha_between = (mu_a_c * sqrtp).T @ (mu_a_c * sqrtp)     # (d,d)
    Hb_between = (mu_b_c * sqrtp).T @ (mu_b_c * sqrtp)     # (C,C)
    
    # weights
    W_cpu = W.detach().cpu().float()
    WTW = (W_cpu.T @ W_cpu)  # (d,d)
    WWT = (W_cpu @ W_cpu.T)  # (C,C)

    ha_stats = _finalize_stats(sum_ha_n2, sum_ha_n2_sq, n_ha)
    gb_stats = _finalize_stats(sum_gb_n2, sum_gb_n2_sq, n_gb)

    return {
        "Ha": Ha, "Hb": Hb,
        "Ha_means": Ha_means, "Hb_means": Hb_means,
        "mu_a": mu_a, "mu_b": mu_b,
        "Ha_between": Ha_between,
        "Hb_between": Hb_between,
        "WTW": WTW, "WWT": WWT,
        "Ga": Ga, "Gb": Gb,
        "assumption_stats": {
            "ha_norm2": asdict(ha_stats),
            "gb_norm2": asdict(gb_stats),
            "g_mode": g_mode,
            "n_samples": int(n_samples),
            "n_batches_for_g": int(n_batches_used if g_mode == "batch" else n_samples),
        },
    }


@torch.no_grad()
def compute_crh_matrices_all_layers(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    *,
    max_batches: Optional[int] = None,
    g_mode: str = "batch",
) -> Dict[str, Dict[str, Any]]:
    """
    Compute CRH metrics for all linear layers in the model simultaneously.
    Returns a dict mapping layer names (e.g., "fc1", "fc2", "fc3") to their metrics.
    The "__last__" key contains the last layer for backward compatibility.
    """
    model = model.to(device)
    model.eval()

    all_layers = find_all_linear_layers(model)
    if not all_layers:
        raise RuntimeError("No linear layers found in model")

    # Cache for activations: {layer_name: {"h_a": tensor, "h_b": tensor, "W": tensor, "b": tensor}}
    cache: Dict[str, Dict[str, torch.Tensor]] = {name: {} for name, _ in all_layers}

    # Register hooks for all layers
    handles = []
    
    def make_hook(layer_name, layer_module):
        def hook_fn(module, inputs, outputs):
            cache[layer_name]["h_a"] = inputs[0].detach()
            cache[layer_name]["h_b"] = outputs.detach()
            cache[layer_name]["W"] = layer_module.weight.detach()
            if layer_module.bias is not None:
                cache[layer_name]["b"] = layer_module.bias.detach()
        return hook_fn

    for layer_name, layer_module in all_layers:
        handle = layer_module.register_forward_hook(make_hook(layer_name, layer_module))
        handles.append(handle)

    # Process batches and accumulate statistics for each layer
    layer_results = {
        name: {
            "Ha": torch.zeros((0, 0), dtype=torch.float32),
            "Hb": torch.zeros((0, 0), dtype=torch.float32),
            "Ha_means": torch.zeros((0, 0), dtype=torch.float32),
            "Hb_means": torch.zeros((0, 0), dtype=torch.float32),
            "Ga": torch.zeros((0, 0), dtype=torch.float32),
            "Gb": torch.zeros((0, 0), dtype=torch.float32),
            "Ha_between": torch.zeros((0, 0), dtype=torch.float32),
            "Hb_between": torch.zeros((0, 0), dtype=torch.float32),
            "WTW": torch.zeros((0, 0), dtype=torch.float32),
            "WWT": torch.zeros((0, 0), dtype=torch.float32),
            "sum_ha_n2": 0.0,
            "sum_ha_n2_sq": 0.0,
            "n_ha": 0,
            "sum_gb_n2": 0.0,
            "sum_gb_n2_sq": 0.0,
            "n_gb": 0,
            "n_samples": 0,
            "n_batches_used": 0,
            "sum_ha_by_class": None,
            "sum_hb_by_class": None,
            "count_by_class": None,
        }
        for name, _ in all_layers
    }

    n_samples = 0
    n_batches_used = 0

    for bi, (x, y) in enumerate(loader):
        if max_batches is not None and bi >= int(max_batches):
            break

        x = x.to(device)
        y = y.to(device)

        # Clear cache and run forward pass
        for name in cache:
            cache[name].clear()
        
        _ = model(x)  # forward pass populates cache

        B = x.size(0)
        n_samples += B

        # Process each layer
        for layer_name, layer_module in all_layers:
            if layer_name not in cache or "h_a" not in cache[layer_name]:
                continue

            h_a = cache[layer_name]["h_a"]  # (B, d_in)
            h_b = cache[layer_name]["h_b"]  # (B, d_out)
            W = cache[layer_name]["W"]      # (d_out, d_in)
            b = cache[layer_name].get("b")   # (d_out,) or None
            
            # Correct h_b if needed (subtract bias to get pre-nonlinearity)
            if b is not None:
                h_b = h_b - b.unsqueeze(0)

            d_in, d_out = h_a.shape[1], h_b.shape[1]
            C = d_out  # treat output dim as number of classes

            # Initialize if needed
            res = layer_results[layer_name]
            if res["Ha"].shape[0] == 0:
                res["Ha"] = torch.zeros((d_in, d_in), dtype=torch.float32, device=device)
                res["Hb"] = torch.zeros((d_out, d_out), dtype=torch.float32, device=device)
                res["Ga"] = torch.zeros((d_in, d_in), dtype=torch.float32, device=device)
                res["Gb"] = torch.zeros((d_out, d_out), dtype=torch.float32, device=device)
                res["sum_ha_by_class"] = torch.zeros((C, d_in), dtype=torch.float32, device=device)
                res["sum_hb_by_class"] = torch.zeros((C, d_out), dtype=torch.float32, device=device)
                res["count_by_class"] = torch.zeros(C, dtype=torch.float32, device=device)

            # Accumulate second moments
            res["Ha"] += _accum_second_moment(h_a)
            res["Hb"] += _accum_second_moment(h_b)

            # Class-wise accumulation (mock: treat first d_out classes)
            min_c = min(C, y.max().item() + 1)
            if min_c > 0:
                onehot = F.one_hot(y, num_classes=C).to(dtype=torch.float32)
                res["sum_ha_by_class"] += onehot.T @ h_a
                res["sum_hb_by_class"] += onehot.T @ h_b
                res["count_by_class"] += onehot.sum(dim=0)

            # Norm stats for h_a
            ha_n2 = h_a.pow(2).sum(dim=1)
            res["sum_ha_n2"] += float(ha_n2.sum().item())
            res["sum_ha_n2_sq"] += float((ha_n2 * ha_n2).sum().item())
            res["n_ha"] += int(B)

            # Gradient computation
            p = torch.softmax(h_b, dim=1) if d_out > 1 else torch.sigmoid(h_b)
            y_oh = F.one_hot(torch.clamp(y, 0, C-1), num_classes=C).to(dtype=p.dtype)
            gb_sample = (p - y_oh)

            if g_mode == "sample":
                res["Ga"] += _accum_second_moment(gb_sample @ W)
                res["Gb"] += _accum_second_moment(gb_sample)
                gb_n2 = gb_sample.pow(2).sum(dim=1)
                res["sum_gb_n2"] += float(gb_n2.sum().item())
                res["sum_gb_n2_sq"] += float((gb_n2 * gb_n2).sum().item())
                res["n_gb"] += int(B)
            else:
                gb_batch = gb_sample.mean(dim=0)
                ga_batch = gb_batch @ W
                res["Gb"] += torch.outer(gb_batch, gb_batch)
                res["Ga"] += torch.outer(ga_batch, ga_batch)
                gb_n2 = float(gb_batch.pow(2).sum().item())
                res["sum_gb_n2"] += gb_n2
                res["sum_gb_n2_sq"] += gb_n2 * gb_n2
                res["n_gb"] += 1
            
            res["n_samples"] = n_samples
            if g_mode == "batch":
                res["n_batches_used"] += 1

    # Remove hooks
    for handle in handles:
        handle.remove()

    # Normalize and finalize results
    final_results = {}
    for layer_name, layer_module in all_layers:
        res = layer_results[layer_name]
        
        if res["n_samples"] == 0:
            print(f"Warning: No samples processed for layer {layer_name}")
            continue

        d_in = res["Ha"].shape[0]
        d_out = res["Hb"].shape[0]
        C = d_out

        # Normalize second moments
        Ha = (res["Ha"] / float(res["n_samples"])).detach().cpu()
        Hb = (res["Hb"] / float(res["n_samples"])).detach().cpu()

        if g_mode == "sample":
            Ga = (res["Ga"] / float(res["n_samples"])).detach().cpu()
            Gb = (res["Gb"] / float(res["n_samples"])).detach().cpu()
        else:
            denom = float(max(1, res["n_batches_used"]))
            Ga = (res["Ga"] / denom).detach().cpu()
            Gb = (res["Gb"] / denom).detach().cpu()

        # Class means
        count_cpu = res["count_by_class"].detach().cpu()
        count_safe = torch.clamp(count_cpu, min=1.0)
        mu_a = (res["sum_ha_by_class"].detach().cpu() / count_safe.unsqueeze(1))
        mu_b = (res["sum_hb_by_class"].detach().cpu() / count_safe.unsqueeze(1))

        Ha_means = (mu_a.T @ mu_a)
        Hb_means = (mu_b.T @ mu_b)

        # Centered means
        p = count_cpu / (count_cpu.sum() + 1e-8)
        mu_a_bar = (p.unsqueeze(0) @ mu_a).squeeze(0)
        mu_b_bar = (p.unsqueeze(0) @ mu_b).squeeze(0)
        mu_a_c = mu_a - mu_a_bar.unsqueeze(0)
        mu_b_c = mu_b - mu_b_bar.unsqueeze(0)
        sqrtp = torch.sqrt(p).unsqueeze(1)
        Ha_between = (mu_a_c * sqrtp).T @ (mu_a_c * sqrtp)
        Hb_between = (mu_b_c * sqrtp).T @ (mu_b_c * sqrtp)

        # Weights
        W_cpu = cache[layer_name]["W"].detach().cpu().float()
        WTW = (W_cpu.T @ W_cpu)
        WWT = (W_cpu @ W_cpu.T)

        ha_stats = _finalize_stats(res["sum_ha_n2"], res["sum_ha_n2_sq"], res["n_ha"])
        gb_stats = _finalize_stats(res["sum_gb_n2"], res["sum_gb_n2_sq"], res["n_gb"])

        final_results[layer_name] = {
            "Ha": Ha, "Hb": Hb,
            "Ha_means": Ha_means, "Hb_means": Hb_means,
            "mu_a": mu_a, "mu_b": mu_b,
            "Ha_between": Ha_between, "Hb_between": Hb_between,
            "WTW": WTW, "WWT": WWT,
            "Ga": Ga, "Gb": Gb,
            "assumption_stats": {
                "ha_norm2": asdict(ha_stats),
                "gb_norm2": asdict(gb_stats),
                "g_mode": g_mode,
                "n_samples": int(n_samples),
                "n_batches_for_g": int(res.get("n_batches_used", n_samples)),
            },
        }

    # Add last layer result for backward compatibility
    if all_layers:
        last_layer_name = all_layers[-1][0]
        final_results["__last__"] = final_results.get(last_layer_name, {})

    return final_results


def flatten_pair(prefix: str, m: Dict[str, Any]) -> Dict[str, Any]:
    """Extract metrics from alignment pair into flattened dict with prefix."""
    pa = m["principal_angles_topk"]
    pl = m["powerlaw_eigs"]
    pw = m["best_power_fit"]
    return {
        f"{prefix}_eps_lin": m["eps_lin"],
        f"{prefix}_c_lin": m["c_lin"],
        f"{prefix}_cosine_sim_fro": m["cosine_sim_fro"],
        f"{prefix}_pearson_correlation": m["pearson_correlation"],
        f"{prefix}_eps_comm": m["eps_comm"],
        f"{prefix}_pa_mean_cos": pa["mean_cos"],
        f"{prefix}_pa_min_cos": pa["min_cos"],
        f"{prefix}_pa_max_angle_deg": pa["max_angle_deg"],
        f"{prefix}_powerlaw_alpha": pl["alpha"],
        f"{prefix}_powerlaw_r2": pl["r2"],
        f"{prefix}_pow_eps": pw["eps_pow"],
        f"{prefix}_pow_alpha": pw["alpha"],
        f"{prefix}_pow_c": pw["c"],
        f"{prefix}_pow_cosine_sim_fro": pw.get("rho_pow_fro", float("nan")),
    }


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    criterion: nn.Module,
) -> float:
    model.train()
    running = 0.0
    n = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()
        running += float(loss.item()) * x.size(0)
        n += int(x.size(0))
    return running / max(1, n)


def build_run_dir(
    *,
    sweep_root: Path,
    arch: str,
    dataset: str,
    epochs: int,
    lr: float,
    wd: float,
    seed: int,
    hidden_dims: Tuple[int, ...],
    dropout: float,
    label_smoothing: float,
) -> Path:
    hd_tag = "-".join(str(x) for x in hidden_dims)
    tag = f"ep{epochs}_lr_{lr:g}_wd_{wd:g}_hd_{hd_tag}_do_{dropout:g}_ls_{label_smoothing:g}"
    return sweep_root / arch / tag / dataset / "full" / f"seed_{seed}" / "final_train"


def main() -> None:
    p = argparse.ArgumentParser()

    p.add_argument("--dataset", type=str, default="MNIST", help="MNIST or FASHIONMNIST (depending on your datasets.py).")
    p.add_argument("--arch", type=str, default="mnist_mlp_reg")
    p.add_argument("--data_root", type=str, default="./data")
    p.add_argument("--sweep_root", type=str, default="runs_sweep_full")

    p.add_argument("--epochs", type=int, default=800)
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--seed", type=int, default=0)

    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=3e-4)

    p.add_argument("--hidden_dims", type=str, default="2048,512")
    p.add_argument("--dropout", type=float, default=0.0)
    p.add_argument("--label_smoothing", type=float, default=0.05)

    p.add_argument("--lr_schedule", type=str, default="cosine", choices=["none", "cosine", "step"])
    p.add_argument("--lr_min", type=float, default=1e-5)
    p.add_argument("--step_size", type=int, default=300)
    p.add_argument("--gamma", type=float, default=0.1)

    # metrics computation
    p.add_argument("--metrics_every", type=int, default=40)
    p.add_argument("--metrics_split", type=str, default="train", choices=["train", "eval"])
    p.add_argument("--metrics_max_batches", type=int, default=200)
    p.add_argument("--g_mode", type=str, default="batch", choices=["batch", "sample"])

    # hz_metrics
    p.add_argument("--k", type=int, default=10)  # smaller k is more meaningful for subspace alignment
    p.add_argument("--eig_tol", type=float, default=1e-12)
    p.add_argument("--alpha_min", type=float, default=0.0)
    p.add_argument("--alpha_max", type=float, default=3.0)
    p.add_argument("--alpha_steps", type=int, default=121)

    p.add_argument("--save_checkpoints_every", type=int, default=200)
    p.add_argument("--save_matrices", action="store_true")
    args = p.parse_args()

    device = utils.get_device()
    set_seed(int(args.seed))

    dataset_name = args.dataset.strip().upper()
    arch = args.arch.strip()
    hidden_dims = parse_int_list(args.hidden_dims)
    dropout = float(args.dropout)
    label_smoothing = float(args.label_smoothing)

    stats = datasets.DATASET_STATS[dataset_name]
    num_classes = int(stats["num_classes"])
    in_channels = int(stats["in_channels"])
    input_shape = (in_channels, *tuple(stats["image_size"]))

    train_full, eval_full, _test_ds = datasets.build_datasets(
        dataset_name, root=args.data_root, download=True, augment_train=False, normalize=True
    )

    train_ds = train_full
    eval_ds = eval_full

    g = torch.Generator().manual_seed(int(args.seed))
    train_loader = DataLoader(
        train_ds,
        batch_size=int(args.batch_size),
        shuffle=True,
        num_workers=int(args.num_workers),
        worker_init_fn=seed_worker if int(args.num_workers) > 0 else None,
        generator=g,
        pin_memory=(device.type == "cuda"),
    )
    eval_loader = DataLoader(
        eval_ds,
        batch_size=int(args.batch_size),
        shuffle=False,
        num_workers=int(args.num_workers),
        pin_memory=(device.type == "cuda"),
    )

    metrics_loader = train_loader if args.metrics_split == "train" else eval_loader

    model = architectures.build_model(
        arch,
        num_classes=num_classes,
        in_channels=in_channels,
        input_shape=input_shape,
        hidden_dims=hidden_dims,
        dropout=dropout,
    ).to(device)

    criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(args.lr), weight_decay=float(args.weight_decay))

    scheduler = None
    if args.lr_schedule == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=int(args.epochs), eta_min=float(args.lr_min)
        )
    elif args.lr_schedule == "step":
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=int(args.step_size), gamma=float(args.gamma)
        )

    run_dir = build_run_dir(
        sweep_root=Path(args.sweep_root),
        arch=arch,
        dataset=dataset_name,
        epochs=int(args.epochs),
        lr=float(args.lr),
        wd=float(args.weight_decay),
        seed=int(args.seed),
        hidden_dims=hidden_dims,
        dropout=dropout,
        label_smoothing=label_smoothing,
    )
    run_dir.mkdir(parents=True, exist_ok=True)

    cfg = {
        "dataset": dataset_name,
        "arch": arch,
        "hidden_dims": list(hidden_dims),
        "dropout": dropout,
        "label_smoothing": label_smoothing,
        "optimizer": {"type": "Adam", "lr": float(args.lr), "weight_decay": float(args.weight_decay)},
        "epochs": int(args.epochs),
        "batch_size": int(args.batch_size),
        "seed": int(args.seed),
        "device": str(device),
        "metrics": {
            "every": int(args.metrics_every),
            "split": str(args.metrics_split),
            "max_batches": int(args.metrics_max_batches),
            "g_mode": str(args.g_mode),
        },
        "hz_metrics": {
            "k": int(args.k),
            "eig_tol": float(args.eig_tol),
            "alpha_min": float(args.alpha_min),
            "alpha_max": float(args.alpha_max),
            "alpha_steps": int(args.alpha_steps),
        },
        "lr_schedule": {
            "type": str(args.lr_schedule),
            "lr_min": float(args.lr_min),
            "step_size": int(args.step_size),
            "gamma": float(args.gamma),
        },
    }
    save_json(run_dir / "config.json", cfg)

    history = {"train_loss": [], "train_accuracy": [], "val_loss": [], "val_accuracy": []}
    metrics_csv_path = run_dir / "crh_pah_metrics.csv"
    wrote_header = False

    print(f"Run dir: {run_dir}")
    t0 = time.time()

    def M(A: torch.Tensor, B: torch.Tensor, allow_indefinite: bool) -> Dict[str, Any]:
        return compute_all_metrics(
            A, B,
            sym=True,
            k=int(args.k),
            eig_tol=float(args.eig_tol),
            alpha_min=float(args.alpha_min),
            alpha_max=float(args.alpha_max),
            alpha_steps=int(args.alpha_steps),
            allow_indefinite=allow_indefinite,
        )

    for epoch in range(1, int(args.epochs) + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, device, criterion)
        if scheduler is not None:
            scheduler.step()

        train_acc = train_loop.get_train_accuracy(model, train_loader, device)
        val_loss, val_acc = train_loop.validate(model, criterion, eval_loader, device)

        history["train_loss"].append(float(train_loss))
        history["train_accuracy"].append(float(train_acc))
        history["val_loss"].append(float(val_loss))
        history["val_accuracy"].append(float(val_acc))

        if epoch % int(args.save_checkpoints_every) == 0:
            torch.save({"epoch": epoch, "state_dict": model.state_dict()}, run_dir / f"ckpt_epoch{epoch}.pth")

        do_metrics = (epoch % int(args.metrics_every) == 0) or (epoch == int(args.epochs))
        if do_metrics:
            max_batches = int(args.metrics_max_batches)
            max_batches = None if max_batches <= 0 else max_batches

            all_layer_results = compute_crh_matrices_all_layers(
                model,
                metrics_loader,
                device,
                max_batches=max_batches,
                g_mode=str(args.g_mode),
            )

            row: Dict[str, Any] = {
                "epoch": int(epoch),
                "lr": float(args.lr),
                "weight_decay": float(args.weight_decay),
                "hidden_dims": str(args.hidden_dims),
                "dropout": float(dropout),
                "label_smoothing": float(label_smoothing),
                "train_loss": float(train_loss),
                "train_acc": float(train_acc),
                "val_loss": float(val_loss),
                "val_acc": float(val_acc),
            }

            # Process metrics for each layer
            for layer_name, mats in all_layer_results.items():
                if layer_name == "__last__":
                    continue  # Handle last layer separately for backward compatibility

                Ha, Hb = mats["Ha"], mats["Hb"]
                HaM, HbM = mats["Ha_means"], mats["Hb_means"]
                WTW, WWT = mats["WTW"], mats["WWT"]
                Ga, Gb = mats["Ga"], mats["Gb"]

                # core pairs (PSD-ish)
                m_Ha_WTW = M(Ha, WTW, allow_indefinite=False)
                m_HaM_WTW = M(HaM, WTW, allow_indefinite=False)
                m_Hb_WWT = M(Hb, WWT, allow_indefinite=False)
                m_HbM_WWT = M(HbM, WWT, allow_indefinite=False)

                # optional theorem chain pairs
                m_Ga_WTW = M(Ga, WTW, allow_indefinite=False)
                m_Gb_WWT = M(Gb, WWT, allow_indefinite=False)

                # trace-removed (indefinite => allow_indefinite=True)
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

                # Add assumption stats for this layer
                for k, v in mats["assumption_stats"].items():
                    if isinstance(v, dict):
                        for kk, vv in v.items():
                            row[f"{layer_name}_assump_{k}_{kk}"] = vv
                    else:
                        row[f"{layer_name}_assump_{k}"] = v

                # norms for trace-removed matrices
                row[f"{layer_name}_||Ha_tr||F"] = fro_norm(Ha_tr)
                row[f"{layer_name}_||WTW_tr||F"] = fro_norm(WTW_tr)
                row[f"{layer_name}_||Hb_tr||F"] = fro_norm(Hb_tr)
                row[f"{layer_name}_||WWT_tr||F"] = fro_norm(WWT_tr)

                row.update({f"{layer_name}_{k}": v for k, v in flatten_pair("HaBetween_WTW", m_HaB_WTW).items()})
                row.update({f"{layer_name}_{k}": v for k, v in flatten_pair("HbBetween_WWT", m_HbB_WWT).items()})

                # and trace-removed versions
                HaB_tr = trace_remove(HaB)
                HbB_tr = trace_remove(HbB)

                row.update({f"{layer_name}_{k}": v for k, v in flatten_pair("HaBetween_WTW_tr", M(HaB_tr, WTW_tr, allow_indefinite=True)).items()})
                row.update({f"{layer_name}_{k}": v for k, v in flatten_pair("HbBetween_WWT_tr", M(HbB_tr, WWT_tr, allow_indefinite=True)).items()})

                row.update({f"{layer_name}_{k}": v for k, v in flatten_pair("Ha_WTW", m_Ha_WTW).items()})
                row.update({f"{layer_name}_{k}": v for k, v in flatten_pair("HaMeans_WTW", m_HaM_WTW).items()})
                row.update({f"{layer_name}_{k}": v for k, v in flatten_pair("Hb_WWT", m_Hb_WWT).items()})
                row.update({f"{layer_name}_{k}": v for k, v in flatten_pair("HbMeans_WWT", m_HbM_WWT).items()})

                row.update({f"{layer_name}_{k}": v for k, v in flatten_pair("Ga_WTW", m_Ga_WTW).items()})
                row.update({f"{layer_name}_{k}": v for k, v in flatten_pair("Gb_WWT", m_Gb_WWT).items()})

                row.update({f"{layer_name}_{k}": v for k, v in flatten_pair("Ha_WTW_tr", m_Ha_WTW_tr).items()})
                row.update({f"{layer_name}_{k}": v for k, v in flatten_pair("HaMeans_WTW_tr", m_HaM_WTW_tr).items()})
                row.update({f"{layer_name}_{k}": v for k, v in flatten_pair("Hb_WWT_tr", m_Hb_WWT_tr).items()})
                row.update({f"{layer_name}_{k}": v for k, v in flatten_pair("HbMeans_WWT_tr", m_HbM_WWT_tr).items()})

            # For backward compatibility: also add last layer metrics without prefix
            if "__last__" in all_layer_results:
                mats = all_layer_results["__last__"]

                Ha, Hb = mats["Ha"], mats["Hb"]
                HaM, HbM = mats["Ha_means"], mats["Hb_means"]
                WTW, WWT = mats["WTW"], mats["WWT"]
                Ga, Gb = mats["Ga"], mats["Gb"]

                # core pairs (PSD-ish)
                m_Ha_WTW = M(Ha, WTW, allow_indefinite=False)
                m_HaM_WTW = M(HaM, WTW, allow_indefinite=False)
                m_Hb_WWT = M(Hb, WWT, allow_indefinite=False)
                m_HbM_WWT = M(HbM, WWT, allow_indefinite=False)

                # optional theorem chain pairs
                m_Ga_WTW = M(Ga, WTW, allow_indefinite=False)
                m_Gb_WWT = M(Gb, WWT, allow_indefinite=False)

                # trace-removed (indefinite => allow_indefinite=True)
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

                # norms for trace-removed matrices (helps interpret stability)
                row["||Ha_tr||F"] = fro_norm(Ha_tr)
                row["||WTW_tr||F"] = fro_norm(WTW_tr)
                row["||Hb_tr||F"] = fro_norm(Hb_tr)
                row["||WWT_tr||F"] = fro_norm(WWT_tr)
                for k, v in mats["assumption_stats"].items():
                    if isinstance(v, dict):
                        for kk, vv in v.items():
                            row[f"assump_{k}_{kk}"] = vv
                    else:
                        row[f"assump_{k}"] = v

                row.update(flatten_pair("HaBetween_WTW", m_HaB_WTW))
                row.update(flatten_pair("HbBetween_WWT", m_HbB_WWT))

                # and trace-removed versions
                HaB_tr = trace_remove(HaB)
                HbB_tr = trace_remove(HbB)

                row.update(flatten_pair("HaBetween_WTW_tr", M(HaB_tr, WTW_tr, allow_indefinite=True)))
                row.update(flatten_pair("HbBetween_WWT_tr", M(HbB_tr, WWT_tr, allow_indefinite=True)))

                row.update(flatten_pair("Ha_WTW", m_Ha_WTW))
                row.update(flatten_pair("HaMeans_WTW", m_HaM_WTW))
                row.update(flatten_pair("Hb_WWT", m_Hb_WWT))
                row.update(flatten_pair("HbMeans_WWT", m_HbM_WWT))

                row.update(flatten_pair("Ga_WTW", m_Ga_WTW))
                row.update(flatten_pair("Gb_WWT", m_Gb_WWT))

                row.update(flatten_pair("Ha_WTW_tr", m_Ha_WTW_tr))
                row.update(flatten_pair("HaMeans_WTW_tr", m_HaM_WTW_tr))
                row.update(flatten_pair("Hb_WWT_tr", m_Hb_WWT_tr))
                row.update(flatten_pair("HbMeans_WWT_tr", m_HbM_WWT_tr))


            with metrics_csv_path.open("a", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=list(row.keys()))
                if not wrote_header:
                    w.writeheader()
                    wrote_header = True
                w.writerow(row)

            save_json(run_dir / f"metrics_epoch{epoch}.json", row)

            # brief console summary focused on the new targets
            print(
                f"[epoch {epoch:4d}] "
                f"HaMeans_WTW eps_lin={row['HaMeans_WTW_eps_lin']:.4f} "
                f"(tr eps_lin={row['HaMeans_WTW_tr_eps_lin']:.4f}) | "
                f"Ha_WTW eps_lin={row['Ha_WTW_eps_lin']:.4f} | "
                f"HaMeans_WTW r2={row['HaMeans_WTW_powerlaw_r2']:.3f}"
            )

        if epoch % 20 == 0 or epoch == 1:
            lr_now = optimizer.param_groups[0]["lr"]
            print(
                f"Epoch {epoch:4d}: train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
                f"val_loss={val_loss:.4f} val_acc={val_acc:.4f} lr={lr_now:g}"
            )

    save_json(run_dir / "history.json", history)
    torch.save({"epoch": int(args.epochs), "state_dict": model.state_dict()}, run_dir / "final.pth")

    t1 = time.time()
    print(f"Done. Wallclock: {t1 - t0:.1f}s")
    print(f"Wrote: {metrics_csv_path}")


if __name__ == "__main__":
    main()