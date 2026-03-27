#!/usr/bin/env python3
"""
analyze_weight_activation_cka.py

Compute CKA between WEIGHTS and ACTIVATIONS by treating UNITS/CHANNELS as "samples".

Given a layer with d units:
  - Weight features:    W_feats in R^{d x P}   (flatten per-unit weights)
  - Activation features A_feats in R^{d x N}   (per-unit response across N images)

Then compute CKA between the two unit-similarity geometries:
  K = W_feats @ W_feats^T   (d x d)
  L = A_feats @ A_feats^T   (d x d)
  CKA = HSIC(K,L) / sqrt(HSIC(K,K)*HSIC(L,L))

This uses your repo's HSIC definitions in metrics_platonic.py :contentReference[oaicite:2]{index=2}
and matches the "kernel/HSIC" path used by AlignmentMetrics.cka :contentReference[oaicite:3]{index=3}.

Example:
  PYTHONPATH=. python analyze_weight_activation_cka.py \
    --ckpt ./runs_resnet20_ln_warmcos/CIFAR10/full/seed_0/resnet20_CIFAR10_full_seed0_best.pth \
    --dataset CIFAR10 --model resnet20 \
    --layer layer3.2.conv2 --hook output --split test \
    --preprocess abs --max-samples 10000 --normalize-rows \
    --out ./weight_activation_cka.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Optional, Tuple, Any

_HERE = Path(__file__).resolve().parent
_SRC  = _HERE.parent.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

import architectures
import datasets as ds_utils
import utils
import metrics_platonic as metrics


# -----------------------
# Checkpoint helpers (same style as your cosine script)
# -----------------------
def load_ckpt_state_dict(path: str) -> Dict[str, torch.Tensor]:
    obj = torch.load(path, map_location="cpu")
    if isinstance(obj, dict) and "state_dict" in obj and isinstance(obj["state_dict"], dict):
        return obj["state_dict"]
    if isinstance(obj, dict):
        return obj
    raise ValueError(f"Unrecognized checkpoint format at: {path}")


def normalize_state_dict_keys(state: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    prefixes = ("module.", "model.", "net.")
    out = dict(state)
    changed = True
    while changed:
        changed = False
        keys = list(out.keys())
        for p in prefixes:
            if keys and all(k.startswith(p) for k in keys):
                out = {k[len(p):]: v for k, v in out.items()}
                changed = True
                break
    return out


# -----------------------
# Best-effort inference
# -----------------------
def guess_model_name_from_state(state: Dict[str, torch.Tensor]) -> Optional[str]:
    keys = set(state.keys())
    if "fc1.weight" in keys:
        return "mlp"
    if "linear.weight" in keys and any(k.startswith("layer1.") for k in keys):
        return "resnet20"
    if "fc.weight" in keys and any(k.startswith("layer4.") for k in keys):
        return "resnet18"
    return None


def infer_num_classes_from_state(state: Dict[str, torch.Tensor]) -> Optional[int]:
    for k in ("linear.weight", "fc.weight"):
        if k in state and state[k].ndim == 2:
            return int(state[k].shape[0])
    fc_keys = [k for k in state.keys() if k.startswith("fc") and k.endswith(".weight")]
    if fc_keys:
        def _idx(x: str) -> int:
            try:
                return int(x.split(".")[0][2:])
            except Exception:
                return -1
        last = max(fc_keys, key=_idx)
        if state[last].ndim == 2:
            return int(state[last].shape[0])
    return None


def infer_in_channels_from_state(state: Dict[str, torch.Tensor]) -> Optional[int]:
    if "conv1.weight" in state and state["conv1.weight"].ndim >= 2:
        return int(state["conv1.weight"].shape[1])
    return None


def infer_resnet20_width_multiplier(state: Dict[str, torch.Tensor]) -> Optional[int]:
    if "conv1.weight" not in state:
        return None
    w = int(state["conv1.weight"].shape[0] // 16)
    return w if w > 0 else None


def infer_resnet20_shortcut_option(state: Dict[str, torch.Tensor]) -> str:
    return "C" if any(k.endswith("shortcut.0.weight") for k in state.keys()) else "A"


def infer_norm_from_state(state: Dict[str, torch.Tensor]) -> str:
    keys = state.keys()
    if any(k.endswith("running_mean") or k.endswith("running_var") for k in keys):
        return "bn"
    if any(".ln.weight" in k or ".ln.bias" in k for k in keys):
        return "flax_ln"
    return "ln"


# -----------------------
# Feature construction
# -----------------------
def reduce_activation(act: torch.Tensor, pool: str) -> torch.Tensor:
    """
    act: [B, C, H, W] -> [B, C] (avg) or [B, C*H*W] (flatten)
    act: [B, D] stays [B, D]
    """
    if act.ndim <= 2:
        return act
    if pool == "avg":
        dims = tuple(range(2, act.ndim))
        return act.mean(dim=dims)
    if pool == "flatten":
        return act.flatten(start_dim=1)
    raise ValueError(f"Unknown pool: {pool}")


def weight_features_for_module(mod: nn.Module, mode: str) -> torch.Tensor:
    """
    Return per-unit weight features (rows = units).
    For Conv2d:
      - mode=output: rows are out_channels, features are in*kH*kW
      - mode=input : rows are in_channels, features are out*kH*kW
    For Linear:
      - mode=output: rows are out_features, features are in_features
      - mode=input : rows are in_features, features are out_features
    """
    if not hasattr(mod, "weight") or mod.weight is None:
        raise ValueError("Selected module has no .weight parameter.")
    W = mod.weight.detach().to(dtype=torch.float32)

    if isinstance(mod, nn.Conv2d):
        # [out, in, kH, kW]
        if mode == "output":
            return W.reshape(W.shape[0], -1)
        if mode == "input":
            return W.permute(1, 0, 2, 3).contiguous().reshape(W.shape[1], -1)
        raise ValueError(f"Unknown mode: {mode}")

    if isinstance(mod, nn.Linear):
        # [out, in]
        if mode == "output":
            return W
        if mode == "input":
            return W.t().contiguous()
        raise ValueError(f"Unknown mode: {mode}")

    # Fallback: treat first dim as "output units"
    if mode == "output":
        return W.reshape(W.shape[0], -1)
    if mode == "input" and W.ndim >= 2:
        return W.transpose(0, 1).contiguous().reshape(W.shape[1], -1)
    raise ValueError(f"Cannot compute weight features for weight shape {tuple(W.shape)} with mode={mode}")


def compute_cka_from_gram(K: torch.Tensor, L: torch.Tensor, unbiased: bool) -> float:
    """
    Compute CKA from Gram matrices using repo HSIC functions.
    """
    hsic_fn = metrics.hsic_unbiased if unbiased else metrics.hsic_biased
    hsic_kk = hsic_fn(K, K)
    hsic_ll = hsic_fn(L, L)
    hsic_kl = hsic_fn(K, L)
    cka = hsic_kl / (torch.sqrt(hsic_kk * hsic_ll) + 1e-6)
    return float(cka.item())


def gram_ip(X: torch.Tensor) -> torch.Tensor:
    return X @ X.t()


def gram_rbf(X: torch.Tensor, sigma: float) -> torch.Tensor:
    # matches metrics_platonic.py rbf construction :contentReference[oaicite:4]{index=4}
    D = torch.cdist(X, X) ** 2
    return torch.exp(-D / (2.0 * sigma * sigma))


# -----------------------
# Main
# -----------------------
def main() -> None:
    p = argparse.ArgumentParser()

    p.add_argument("--ckpt", type=str, required=True)
    p.add_argument("--dataset", type=lambda s: s.strip().upper(), default="CIFAR10",
                   choices=["CIFAR10", "CIFAR100", "MNIST", "FASHIONMNIST"])
    p.add_argument("--data-root", type=str, default="./data")

    p.add_argument("--model", type=str, default=None)
    p.add_argument("--norm", type=str, default=None)
    p.add_argument("--width-multiplier", type=int, default=None)
    p.add_argument("--shortcut-option", type=str, default=None, choices=["A", "B", "C"])
    p.add_argument("--num-classes", type=int, default=None)

    p.add_argument("--layer", type=str, required=True)
    p.add_argument("--hook", type=str, default="output", choices=["output", "input"],
                   help="Defines (i) which activation tensor to use and (ii) whether weights are per-output-unit or per-input-unit.")
    p.add_argument("--pool", type=str, default="avg", choices=["avg", "flatten"],
                   help="How to pool conv activations into vectors per image.")
    p.add_argument("--preprocess", type=str, default="none", choices=["none", "relu", "abs"],
                   help="Apply to activation tensor before pooling.")
    p.add_argument("--split", type=str, default="test",
                   choices=["train_eval", "test", "val", "subset_A_eval", "subset_B_eval", "full_train"])
    p.add_argument("--indices-file", type=str, default=None)

    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--max-samples", type=int, default=10000,
                   help="Max images to build activation response profiles (controls activation feature dim).")

    p.add_argument("--kernel", type=str, default="ip", choices=["ip", "rbf"])
    p.add_argument("--rbf-sigma", type=float, default=1.0)
    p.add_argument("--unbiased", action="store_true",
                   help="Use unbiased HSIC (recommended; also avoids the large-D fast path in AlignmentMetrics.cka).")
    p.add_argument("--normalize-rows", action="store_true",
                   help="L2-normalize each unit's feature vector before building Gram matrices (like your alignment pipeline normalizes features :contentReference[oaicite:5]{index=5}).")

    p.add_argument("--out", type=str, default="./weight_activation_cka.json")
    p.add_argument("--list-layers", action="store_true")

    args = p.parse_args()

    device = utils.get_device()

    # Data
    train_full, eval_full, test_ds = ds_utils.build_datasets(
        args.dataset, root=args.data_root, download=True, augment_train=False, normalize=True
    )

    def get_split_dataset() -> torch.utils.data.Dataset:
        if args.split == "test":
            return test_ds
        if args.split == "full_train":
            return eval_full
        if args.split in ("train_eval", "val", "subset_A_eval", "subset_B_eval"):
            if args.dataset not in ("CIFAR10", "CIFAR100"):
                return eval_full
            if args.indices_file is None:
                raise ValueError(f"--split {args.split} requires --indices-file for CIFAR.")
            idx_obj = torch.load(args.indices_file, map_location="cpu")
            base = eval_full
            key_map = {
                "train_eval": "train_indices",
                "val": "val_indices",
                "subset_A_eval": "subset_a_indices",
                "subset_B_eval": "subset_b_indices",
            }
            k = key_map[args.split]
            if k not in idx_obj:
                raise KeyError(f"Indices file missing key '{k}'. Keys: {list(idx_obj.keys())}")
            return Subset(base, idx_obj[k])
        raise ValueError(f"Unknown split: {args.split}")

    split_ds = get_split_dataset()
    loader = DataLoader(
        split_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )

    # Model
    state = normalize_state_dict_keys(load_ckpt_state_dict(args.ckpt))
    model_name = args.model or guess_model_name_from_state(state)
    if model_name is None:
        raise ValueError("Could not infer --model from checkpoint. Pass --model explicitly (e.g. resnet20).")

    norm = args.norm or infer_norm_from_state(state)
    num_classes = args.num_classes or infer_num_classes_from_state(state) or ds_utils.DATASET_STATS[args.dataset]["num_classes"]

    build_kwargs: Dict[str, Any] = {}
    if model_name == "resnet20":
        build_kwargs["width_multiplier"] = args.width_multiplier or infer_resnet20_width_multiplier(state) or 1
        build_kwargs["shortcut_option"] = args.shortcut_option or infer_resnet20_shortcut_option(state)
        build_kwargs["norm"] = norm
    elif model_name in ("resnet18",):
        build_kwargs["norm"] = norm
    elif model_name == "mlp":
        if "fc1.weight" not in state:
            raise KeyError("Expected fc1.weight in MLP checkpoint.")
        hidden = int(state["fc1.weight"].shape[0])
        build_kwargs["hidden"] = hidden
        build_kwargs["input_shape"] = (
            int(ds_utils.DATASET_STATS[args.dataset]["in_channels"]),
            int(ds_utils.DATASET_STATS[args.dataset]["image_size"][0]),
            int(ds_utils.DATASET_STATS[args.dataset]["image_size"][1]),
        )

    in_channels = infer_in_channels_from_state(state) or int(ds_utils.DATASET_STATS[args.dataset]["in_channels"])
    model = architectures.build_model(
        model_name,
        num_classes=int(num_classes),
        in_channels=int(in_channels),
        **build_kwargs,
    ).to(device)

    # Load weights
    model_keys = set(model.state_dict().keys())
    filtered = {k: v for k, v in state.items() if k in model_keys}
    missing, unexpected = model.load_state_dict(filtered, strict=False)
    if missing:
        raise KeyError(f"Missing keys when loading state_dict (first 20): {missing[:20]}")
    if unexpected:
        print(f"[warn] Unexpected keys (first 20): {unexpected[:20]}")

    if args.list_layers:
        print("Weight-bearing layers (name -> weight shape):")
        for name, mod in model.named_modules():
            if hasattr(mod, "weight") and isinstance(getattr(mod, "weight"), torch.Tensor):
                w = getattr(mod, "weight")
                if w is not None:
                    print(f"  {name:40s} {tuple(w.shape)}  ({mod.__class__.__name__})")
        return

    modules = dict(model.named_modules())
    if args.layer not in modules:
        candidates = [n for n in modules.keys() if args.layer in n]
        hint = "\n  ".join(candidates[:50]) if candidates else "(no partial matches)"
        raise KeyError(f"Layer '{args.layer}' not found. Partial matches:\n  {hint}")
    layer_mod = modules[args.layer]

    # Weight features: [d, P]
    W_feats = weight_features_for_module(layer_mod, mode=args.hook).to(device="cpu", dtype=torch.float32)
    d = int(W_feats.shape[0])

    # Activation accumulation: we want A_all = [N_images, d], then transpose -> [d, N_images]
    act_rows = []

    def hook_fn(_m: nn.Module, inp: Tuple[torch.Tensor, ...], out: torch.Tensor) -> None:
        A = inp[0] if args.hook == "input" else out
        if not torch.is_tensor(A):
            return
        if args.preprocess == "relu":
            A = torch.relu(A)
        elif args.preprocess == "abs":
            A = A.abs()
        A = reduce_activation(A, pool=args.pool)  # -> [B, D]
        A = A.detach().to(device="cpu", dtype=torch.float32)
        act_rows.append(A)

    h = layer_mod.register_forward_hook(hook_fn)

    model.eval()
    n_seen = 0
    with torch.no_grad():
        for xb, _yb in loader:
            xb = xb.to(device)
            _ = model(xb)
            n_seen += xb.size(0)
            if args.max_samples is not None and n_seen >= args.max_samples:
                break

    h.remove()

    A_all = torch.cat(act_rows, dim=0)
    if args.max_samples is not None:
        A_all = A_all[:args.max_samples]

    if A_all.ndim != 2:
        raise RuntimeError(f"Expected pooled activations to be 2D, got {tuple(A_all.shape)}")

    if A_all.shape[1] != d:
        raise RuntimeError(
            f"Dim mismatch: activations have D={A_all.shape[1]} but weights imply d={d}. "
            f"Check --hook/--pool/layer. (For conv layers, use --pool avg.)"
        )

    A_feats = A_all.t().contiguous()  # [d, N_images]

    # Optional row-normalization (per-unit)
    if args.normalize_rows:
        W_feats = F.normalize(W_feats, p=2, dim=1)
        A_feats = F.normalize(A_feats, p=2, dim=1)

    # Build Gram matrices over units (size d x d)
    if args.kernel == "ip":
        K = gram_ip(W_feats)
        L = gram_ip(A_feats)
    else:
        K = gram_rbf(W_feats, sigma=args.rbf_sigma)
        L = gram_rbf(A_feats, sigma=args.rbf_sigma)

    # Compute CKA via repo HSIC
    cka_val = compute_cka_from_gram(K, L, unbiased=bool(args.unbiased))

    out = {
        "ckpt": str(args.ckpt),
        "dataset": args.dataset,
        "split": args.split,
        "layer": args.layer,
        "hook": args.hook,
        "pool": args.pool,
        "preprocess": args.preprocess,
        "normalize_rows": bool(args.normalize_rows),
        "kernel": args.kernel,
        "rbf_sigma": float(args.rbf_sigma),
        "unbiased": bool(args.unbiased),
        "d_units": int(d),
        "n_images": int(A_all.shape[0]),
        "W_feats_shape": list(W_feats.shape),
        "A_feats_shape": list(A_feats.shape),
        "cka_unit_geometry_weights_vs_activations": float(cka_val),
    }

    print(json.dumps(out, indent=2))
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\nSaved: {out_path.resolve()}")


if __name__ == "__main__":
    main()
