#!/usr/bin/env python3
"""
compare_permutations.py

Given:
  - weight-matching permutations (P_wgt)
  - activation-matching permutations (P_act)

This script computes, per permutation key (layer / perm-group), BOTH:

(1) Combinatorial agreement between permutations:
    - Hamming agreement: mean(P_wgt == P_act)
    - Q = inv(P_wgt) o P_act (a permutation over A-indices)
      * fixed-point fraction of Q
      * cycle structure of Q
      * Cayley distance (n - #cycles)

(2) Cross-objective *alignment* (what you asked for):
    - "How well does activation-matching align WEIGHTS?"
        dW under P_wgt (baseline for weights)
        dW under P_act (cross for weights)
        ratio = dW(P_act) / dW(P_wgt)

    - "How well does weight-matching align ACTIVATIONS?"
        dH under P_act (baseline for activations)
        dH under P_wgt (cross for activations)
        ratio = dH(P_wgt) / dH(P_act)

Where:
  dW is a Frobenius norm error on the subset of parameters whose axes are tagged
  by a given perm-key in the PermutationSpec.
  dH is a Frobenius norm error between activations of A and permuted activations of B
  (computed on a real dataset split).

Permutation convention (IMPORTANT):
  p[i] = j means "A unit i matches B unit j" (A -> B assignment).
  To align B into A indexing, we index-select B along unit/channel dim with p.

Supports:
  - MLP (fc1..fcN) as in repo architectures.MLP (fc1..fc4 typical)
  - ResNet20 (CIFAR-style) with LayerNorm2d/Flax-LN naming via resnet20_layernorm_permutation_spec

Typical usage for your case (FashionMNIST MLP):
  python compare_permutations.py \
    --model mlp \
    --dataset FASHIONMNIST \
    --act-perm activation_out/.../permutations.pkl \
    --wgt-perm weight_matching_out.../permutation_seed0.pkl \
    --state-a runs_mlp/FASHIONMNIST/...subsetA...best.pth \
    --state-b runs_mlp/FASHIONMNIST/...subsetB...best.pth \
    --out-json out/report_fashionmnist_mlp.json
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

# Repo imports
import architectures
import datasets as ds_utils

from linear_mode_connectivity.weight_matching_torch import (
    PermutationSpec,
    apply_permutation,
    permutation_spec_from_axes_to_perm,
    resnet20_layernorm_permutation_spec,
)

# Only needed for ResNet20 activation hooks
from model_stitching.resnet20_activation_stitching import (
    infer_width_multiplier_from_state,
    infer_shortcut_option_from_state,
    perm_name_to_hook,
)

# -------------------------
# IO helpers
# -------------------------
def _resolve_rel(base_path: str, maybe_rel: str) -> str:
    if os.path.isabs(maybe_rel):
        return maybe_rel
    return os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(base_path)), maybe_rel))


def _torch_load(path: str) -> Any:
    return torch.load(path, map_location="cpu")


def _load_any(path: str) -> Any:
    ext = os.path.splitext(path)[1].lower()
    if ext in [".pkl", ".pickle"]:
        with open(path, "rb") as f:
            return pickle.load(f)
    if ext in [".pt", ".pth"]:
        return _torch_load(path)
    if ext == ".json":
        with open(path, "r") as f:
            return json.load(f)
    raise ValueError(f"Unsupported file extension: {ext} (path={path})")


def _extract_perm_payload(obj: Any, src_path: str) -> Any:
    """
    Support wrappers:
      - direct dict of permutations
      - dict with key 'permutations'
      - results dict with 'permutations_files' pointing to pickle/pt/json
    """
    if isinstance(obj, dict):
        if "permutations_files" in obj and isinstance(obj["permutations_files"], dict):
            pf = obj["permutations_files"]
            for k in ["pickle", "pt", "json"]:
                if k in pf and pf[k]:
                    perm_path = _resolve_rel(src_path, str(pf[k]))
                    return _extract_perm_payload(_load_any(perm_path), perm_path)
        if "permutations" in obj:
            return obj["permutations"]
    return obj


def _to_long_1d(x: Any) -> torch.Tensor:
    if isinstance(x, torch.Tensor):
        t = x.detach().cpu()
    elif isinstance(x, np.ndarray):
        t = torch.from_numpy(x)
    elif isinstance(x, list):
        t = torch.tensor(x)
    else:
        raise TypeError(f"Unsupported permutation value type: {type(x)}")
    if t.ndim != 1:
        raise ValueError(f"Permutation must be 1D, got shape {tuple(t.shape)}")
    return t.to(dtype=torch.long)


def load_permutations(path: str) -> Dict[str, torch.Tensor]:
    raw = _load_any(path)
    payload = _extract_perm_payload(raw, path)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected dict-like permutations in {path}, got {type(payload)}")

    out: Dict[str, torch.Tensor] = {}
    for k, v in payload.items():
        out[str(k)] = _to_long_1d(v)
    return out


# -------------------------
# State dict loading
# -------------------------
def load_state_dict_any(path: str) -> Dict[str, torch.Tensor]:
    obj = _load_any(path)
    if isinstance(obj, dict) and "state_dict" in obj and isinstance(obj["state_dict"], dict):
        obj = obj["state_dict"]
    if not isinstance(obj, dict):
        raise ValueError(f"Expected dict-like state_dict in {path}, got {type(obj)}")

    state = {str(k): v for k, v in obj.items()}
    for pref in ("module.", "model.", "net."):
        if state and all(k.startswith(pref) for k in state.keys()):
            state = {k[len(pref):]: v for k, v in state.items()}

    out: Dict[str, torch.Tensor] = {}
    for k, v in state.items():
        if not isinstance(v, torch.Tensor):
            v = torch.tensor(v)
        out[k] = v.detach().cpu()
    return out


def global_weight_distance(
    state_a: Dict[str, torch.Tensor],
    state_b: Dict[str, torch.Tensor],
) -> Dict[str, float]:
    ssq = 0.0
    ssqA = 0.0
    num = 0
    for k, Wa in state_a.items():
        if k not in state_b:
            continue
        Wb = state_b[k]
        if Wa.shape != Wb.shape:
            continue
        da = Wa.float()
        db = Wb.float()
        diff = da - db
        ssq += float((diff * diff).sum().item())
        ssqA += float((da * da).sum().item())
        num += 1
    raw = float(ssq ** 0.5)
    rel = raw / ((ssqA ** 0.5) + 1e-12)
    return {"raw_frob": raw, "rel_to_A": rel, "num_tensors": num}


# -------------------------
# Perm-key reconciliation (MLP: "1,2,3" vs "P1,P2,P3")
# -------------------------
_P_INT = re.compile(r"^P(\d+)$")


def reconcile_keys(
    act: Dict[str, torch.Tensor],
    wgt: Dict[str, torch.Tensor],
) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor], str]:
    if set(act).intersection(set(wgt)):
        return act, wgt, "direct"

    act_digits = all(k.isdigit() for k in act.keys())
    wgt_pints = all(_P_INT.match(k) for k in wgt.keys())
    if act_digits and wgt_pints:
        act2 = {f"P{k}": v for k, v in act.items()}
        if set(act2).intersection(set(wgt)):
            return act2, wgt, "act_digit_to_P"

    if wgt_pints:
        wgt2 = {_P_INT.match(k).group(1): v for k, v in wgt.items()}  # type: ignore[union-attr]
        if set(act).intersection(set(wgt2)):
            return act, wgt2, "wgt_P_to_digit"

    return act, wgt, "none"


# -------------------------
# Permutation math
# -------------------------
def is_valid_permutation(p: torch.Tensor) -> bool:
    n = int(p.numel())
    if n <= 0:
        return False
    if p.min().item() < 0 or p.max().item() >= n:
        return False
    s = torch.sort(p).values
    return torch.equal(s, torch.arange(n, dtype=torch.long))


def invert_perm(p_a_to_b: torch.Tensor) -> torch.Tensor:
    n = int(p_a_to_b.numel())
    inv = torch.empty(n, dtype=torch.long)
    inv[p_a_to_b] = torch.arange(n, dtype=torch.long)
    return inv


def compose_q(p_wgt: torch.Tensor, p_act: torch.Tensor) -> torch.Tensor:
    if p_wgt.numel() != p_act.numel():
        raise ValueError(f"Size mismatch: |P_wgt|={p_wgt.numel()} vs |P_act|={p_act.numel()}")
    inv_w = invert_perm(p_wgt)
    return inv_w[p_act]


def cycles_of_perm(q: torch.Tensor) -> List[List[int]]:
    n = int(q.numel())
    visited = [False] * n
    cycles: List[List[int]] = []
    for i in range(n):
        if visited[i]:
            continue
        cur = i
        cyc: List[int] = []
        while not visited[cur]:
            visited[cur] = True
            cyc.append(cur)
            cur = int(q[cur].item())
        cycles.append(cyc)
    return cycles


def cycle_summary(q: torch.Tensor) -> Dict[str, Any]:
    cycs = cycles_of_perm(q)
    lengths = sorted([len(c) for c in cycs], reverse=True)
    hist: Dict[int, int] = {}
    for L in lengths:
        hist[L] = hist.get(L, 0) + 1
    n = int(q.numel())
    return {
        "n": n,
        "num_cycles": len(cycs),
        "cycle_lengths_desc": lengths,
        "cycle_histogram": {str(k): v for k, v in sorted(hist.items())},
        "max_cycle": max(lengths) if lengths else 0,
        "cayley_distance": n - len(cycs),
    }


# -------------------------
# Build a PermutationSpec for MLP (fc1..fcN)
# -------------------------
_FC_W_RE = re.compile(r"^fc(\d+)\.weight$")


def infer_mlp_fc_layer_numbers(state: Dict[str, torch.Tensor]) -> List[int]:
    layers: List[int] = []
    for k in state.keys():
        m = _FC_W_RE.match(k)
        if m:
            layers.append(int(m.group(1)))
    layers = sorted(set(layers))
    if not layers:
        raise KeyError("No fc{n}.weight keys found in checkpoint state_dict.")
    return layers


def mlp_permutation_spec_from_state(state: Dict[str, torch.Tensor]) -> PermutationSpec:
    layer_ids = infer_mlp_fc_layer_numbers(state)
    n_layers = max(layer_ids)
    expected = list(range(1, n_layers + 1))
    if layer_ids != expected:
        raise ValueError(f"Expected contiguous fc layers {expected}, found {layer_ids}")

    axes: Dict[str, Tuple[Optional[str], ...]] = {}

    prev_p: Optional[str] = None
    for i in range(1, n_layers):
        p_out = f"P{i}"
        axes[f"fc{i}.weight"] = (p_out, prev_p)  # (out, in)
        axes[f"fc{i}.bias"] = (p_out,)          # (out,)
        prev_p = p_out

    axes[f"fc{n_layers}.weight"] = (None, prev_p)
    axes[f"fc{n_layers}.bias"] = (None,)
    return permutation_spec_from_axes_to_perm(axes)


# -------------------------
# Alignment metrics you asked for
# -------------------------
def _safe_ratio(a: float, b: float, eps: float = 1e-12) -> float:
    if not (np.isfinite(a) and np.isfinite(b)):
        return float("nan")
    return float(a / (b + eps))


def _frob(x: torch.Tensor) -> float:
    return float(torch.linalg.norm(x).item())


def dW_for_perm_key(
    *,
    perm_key: str,
    ps: PermutationSpec,
    state_a: Dict[str, torch.Tensor],
    state_b_perm: Dict[str, torch.Tensor],
) -> Dict[str, float]:
    if perm_key not in ps.perm_to_axes:
        return {"raw_frob": float("nan"), "rel_to_A": float("nan"), "num_params": 0}

    param_names = sorted({wk for (wk, _axis) in ps.perm_to_axes[perm_key]})
    num = 0
    ssq = 0.0
    ssqA = 0.0
    for name in param_names:
        if name not in state_a or name not in state_b_perm:
            continue
        da = state_a[name].float()
        db = state_b_perm[name].float()
        diff = da - db
        ssq += float((diff * diff).sum().item())
        ssqA += float((da * da).sum().item())
        num += 1

    raw = float(ssq ** 0.5)
    rel = raw / ((ssqA ** 0.5) + 1e-12)
    return {"raw_frob": raw, "rel_to_A": rel, "num_params": num}


def activation_to_2d(out: torch.Tensor, unit_dim: int = 1) -> torch.Tensor:
    if out.ndim < 2:
        raise ValueError(f"Expected activation with ndim>=2, got {tuple(out.shape)}")
    if unit_dim < 0:
        unit_dim = out.ndim + unit_dim
    x = torch.movedim(out, unit_dim, -1)
    return x.reshape(-1, x.shape[-1])


def dH_for_perm_key(
    *,
    perm_key: str,
    feats_a: Dict[str, torch.Tensor],
    feats_b: Dict[str, torch.Tensor],
    p_a_to_b: torch.Tensor,
    unit_dim: int = 1,
) -> Dict[str, float]:
    if perm_key not in feats_a or perm_key not in feats_b:
        return {"raw_frob": float("nan"), "rel_to_A": float("nan")}

    xa = activation_to_2d(feats_a[perm_key], unit_dim=unit_dim).float()
    xb = activation_to_2d(feats_b[perm_key], unit_dim=unit_dim).float()

    d = int(p_a_to_b.numel())
    if xa.shape[1] != d or xb.shape[1] != d:
        return {"raw_frob": float("nan"), "rel_to_A": float("nan")}

    xb_aligned = xb[:, p_a_to_b]
    diff = xa - xb_aligned
    raw = _frob(diff)
    rel = raw / (_frob(xa) + 1e-12)
    return {"raw_frob": raw, "rel_to_A": rel}


# -------------------------
# Make permutations "complete" w.r.t spec (fill missing with identity)
# -------------------------
def complete_perm_dict(
    *,
    ps: PermutationSpec,
    perm: Dict[str, torch.Tensor],
    state_a: Dict[str, torch.Tensor],
) -> Dict[str, torch.Tensor]:
    out = {k: v.clone().to(dtype=torch.long) for k, v in perm.items()}

    for p_name, axes in ps.perm_to_axes.items():
        if p_name in out:
            continue
        # infer size from first (param, axis) pair
        wk, axis = axes[0]
        if wk not in state_a:
            raise KeyError(f"Cannot infer size for missing perm '{p_name}' because '{wk}' not in state_a")
        n = int(state_a[wk].shape[axis])
        out[p_name] = torch.arange(n, dtype=torch.long)

    return out


# -------------------------
# Feature extraction (dataset-based) for dH
# -------------------------
_P_OR_DIGIT = re.compile(r"^(?:P)?(\d+)$")


def _get_submodule_by_name(model: nn.Module, name: str) -> nn.Module:
    cur: nn.Module = model
    for part in name.split("."):
        if part.isdigit():
            idx = int(part)
            if isinstance(cur, (nn.Sequential, nn.ModuleList)):
                cur = cur[idx]
            else:
                cur = getattr(cur, part)
        else:
            cur = getattr(cur, part)
    return cur


@torch.no_grad()
def compute_features_from_state_dicts(
    *,
    model_name: str,
    dataset: str,
    data_root: str,
    split: str,
    samples: int,
    batch_size: int,
    num_workers: int,
    state_a: Dict[str, torch.Tensor],
    state_b: Dict[str, torch.Tensor],
    ps: PermutationSpec,
    norm: str,
    shortcut_option: Optional[str],
    width_multiplier: Optional[int],
    device: torch.device,
    features_dtype: torch.dtype,
) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
    dataset_u = dataset.strip().upper()
    if dataset_u not in ds_utils.DATASET_STATS:
        raise ValueError(f"Unsupported dataset: {dataset_u} (known: {sorted(ds_utils.DATASET_STATS)})")

    train_full, eval_full, test_ds = ds_utils.build_datasets(
        dataset_u,
        root=data_root,
        download=True,
        augment_train=False,
        normalize=True,
    )
    if split == "test":
        ds = test_ds
    elif split == "train_eval":
        ds = eval_full
    else:
        raise ValueError("split must be 'test' or 'train_eval'")

    if samples is not None and int(samples) > 0 and int(samples) < len(ds):
        ds = Subset(ds, list(range(int(samples))))

    loader = DataLoader(
        ds,
        batch_size=int(batch_size),
        shuffle=False,
        num_workers=int(num_workers),
        pin_memory=(device.type == "cuda"),
    )

    perm_names = sorted(ps.perm_to_axes.keys())

    stats = ds_utils.DATASET_STATS[dataset_u]
    in_channels = int(stats["in_channels"])
    img_h, img_w = map(int, stats["image_size"])

    # Build models + load weights
    model_name_l = model_name.strip().lower()
    if model_name_l == "mlp":
        layer_ids = infer_mlp_fc_layer_numbers(state_a)
        n_layers = max(layer_ids)
        hidden = int(state_a["fc1.weight"].shape[0])
        num_classes = int(state_a[f"fc{n_layers}.weight"].shape[0])
        input_shape = (in_channels, img_h, img_w)

        model_a = architectures.build_model(
            "mlp",
            num_classes=num_classes,
            input_shape=input_shape,
            hidden=hidden,
        ).to(device).eval()

        model_b = architectures.build_model(
            "mlp",
            num_classes=num_classes,
            input_shape=input_shape,
            hidden=hidden,
        ).to(device).eval()

        # Hook mapping for MLP perms: P1->fc1, P2->fc2, ...
        def hook_info(pname: str) -> Tuple[str, Optional[Any], int]:
            m = _P_OR_DIGIT.match(pname)
            if not m:
                raise KeyError(f"MLP: don't know how to map perm name to hook layer: {pname}")
            i = int(m.group(1))
            if i < 1 or i >= n_layers:
                raise KeyError(f"MLP: perm {pname} implies fc{i}, but hidden perms are 1..{n_layers-1}")
            return f"fc{i}", F.relu, 1

    elif model_name_l == "resnet20":
        num_classes = int(state_a["linear.weight"].shape[0]) if "linear.weight" in state_a else int(stats["num_classes"])
        if width_multiplier is None:
            width_multiplier = infer_width_multiplier_from_state(state_a)
        if shortcut_option is None:
            shortcut_option = infer_shortcut_option_from_state(state_a)

        model_a = architectures.build_model(
            "resnet20",
            num_classes=num_classes,
            in_channels=in_channels,
            norm=norm,
            width_multiplier=int(width_multiplier),
            shortcut_option=str(shortcut_option),
        ).to(device).eval()

        model_b = architectures.build_model(
            "resnet20",
            num_classes=num_classes,
            in_channels=in_channels,
            norm=norm,
            width_multiplier=int(width_multiplier),
            shortcut_option=str(shortcut_option),
        ).to(device).eval()

        def hook_info(pname: str) -> Tuple[str, Optional[Any], int]:
            return perm_name_to_hook(pname)

    else:
        raise ValueError("--model must be one of: mlp, resnet20 (or use auto detection in main)")

    # Load state dicts (filter to model keys)
    keys = set(model_a.state_dict().keys())
    model_a.load_state_dict({k: v for k, v in state_a.items() if k in keys}, strict=True)
    model_b.load_state_dict({k: v for k, v in state_b.items() if k in keys}, strict=True)

    def extract(model: nn.Module) -> Dict[str, torch.Tensor]:
        store: Dict[str, List[torch.Tensor]] = {p: [] for p in perm_names}
        hooks = []

        for p in perm_names:
            layer_name, preprocess, _unit_dim = hook_info(p)
            mod = _get_submodule_by_name(model, layer_name)

            def make_hook(pname: str, pre):
                def _hook(_m, _inp, out):
                    x = out.detach()
                    if pre is not None:
                        x = pre(x)
                    store[pname].append(x.to(device="cpu", dtype=features_dtype))
                return _hook

            hooks.append(mod.register_forward_hook(make_hook(p, preprocess)))

        seen = 0
        for xb, _yb in loader:
            xb = xb.to(device, non_blocking=True)
            _ = model(xb)
            seen += int(xb.shape[0])
            if samples is not None and int(samples) > 0 and seen >= int(samples):
                break

        for h in hooks:
            h.remove()

        out_feats: Dict[str, torch.Tensor] = {}
        for p in perm_names:
            if len(store[p]) == 0:
                continue
            t = torch.cat(store[p], dim=0)
            if samples is not None and int(samples) > 0 and t.shape[0] > int(samples):
                t = t[: int(samples)]
            out_feats[p] = t
        return out_feats

    return extract(model_a), extract(model_b)


# -------------------------
# Main
# -------------------------
def infer_model_from_state(state: Dict[str, torch.Tensor]) -> str:
    if any(k.startswith("fc1.") for k in state.keys()) or ("fc1.weight" in state):
        return "mlp"
    if ("conv1.weight" in state) or any(k.startswith("layer1.") for k in state.keys()):
        return "resnet20"
    raise ValueError("Could not infer model type from state_dict keys. Use --model explicitly.")


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--act-perm", type=str, required=True,
                    help="Path to activation-matching permutations (.json/.pt/.pkl).")
    ap.add_argument("--wgt-perm", type=str, required=True,
                    help="Path to weight-matching permutations (.pkl/.pt/.json).")
    ap.add_argument("--out-json", type=str, default=None,
                    help="If set, save a JSON report to this path.")

    # What you asked for requires the checkpoints (for dW) and dataset/features (for dH).
    ap.add_argument("--state-a", type=str, default=None, help="Checkpoint/state_dict for model A (.pth/.pt/.pkl).")
    ap.add_argument("--state-b", type=str, default=None, help="Checkpoint/state_dict for model B (.pth/.pt/.pkl).")

    # Model/dataset config (used for correct spec + feature extraction)
    ap.add_argument("--model", type=str, default="auto", choices=["auto", "mlp", "resnet20"],
                    help="Model family. 'auto' infers from state_dict keys.")
    ap.add_argument("--dataset", type=str, default=None,
                    help="Dataset name (e.g. MNIST, FASHIONMNIST, CIFAR10, CIFAR100). "
                         "Needed to compute activations (dH) from checkpoints.")
    ap.add_argument("--data-root", type=str, default="./data", help="Root for dataset downloads.")
    ap.add_argument("--features-split", type=str, default="test", choices=["test", "train_eval"])
    ap.add_argument("--features-samples", type=int, default=512,
                    help="How many samples to use for feature extraction (<=0 => full split).")
    ap.add_argument("--features-batch-size", type=int, default=256)
    ap.add_argument("--features-num-workers", type=int, default=0)
    ap.add_argument("--features-dtype", type=str, default="float16", choices=["float16", "float32"])
    ap.add_argument("--unit-dim", type=int, default=1,
                    help="Unit/channel dimension in activations (default 1; correct for NCHW and [N,D]).")

    # ResNet-specific knobs (safe defaults)
    ap.add_argument("--norm", type=str, default="flax_ln", help="ResNet norm kind (default flax_ln).")
    ap.add_argument("--shortcut-option", type=str, default=None, choices=[None, "A", "B", "C"],
                    help="ResNet shortcut option override. If None, inferred from checkpoint when possible.")
    ap.add_argument("--width-multiplier", type=int, default=None,
                    help="ResNet width multiplier override. If None, inferred from checkpoint.")

    args = ap.parse_args()

    act = load_permutations(args.act_perm)
    wgt = load_permutations(args.wgt_perm)
    act, wgt, strat = reconcile_keys(act, wgt)

    common_keys = sorted(set(act).intersection(set(wgt)))
    if not common_keys:
        raise RuntimeError(
            "No overlapping permutation keys between activation and weight matching.\n"
            f"Reconciliation strategy tried: {strat}\n"
            f"Activation keys (sample): {list(act.keys())[:10]}\n"
            f"Weight keys (sample): {list(wgt.keys())[:10]}"
        )

    do_state = (args.state_a is not None) and (args.state_b is not None)
    state_a: Dict[str, torch.Tensor] = load_state_dict_any(args.state_a) if do_state else {}
    state_b: Dict[str, torch.Tensor] = load_state_dict_any(args.state_b) if do_state else {}

    # Decide model
    model_name = args.model
    if model_name == "auto":
        if not do_state:
            raise ValueError("--model=auto requires --state-a/--state-b (so we can infer).")
        model_name = infer_model_from_state(state_a)

    # Build spec (needed for dW and also to know which perm-names to hook for dH)
    ps: Optional[PermutationSpec] = None
    if do_state:
        if model_name == "resnet20":
            shortcut_opt = args.shortcut_option if args.shortcut_option is not None else None
            # resnet20_layernorm_permutation_spec auto-detects shortcut params if state_dict is passed
            ps = resnet20_layernorm_permutation_spec(
                shortcut_option=str(shortcut_opt) if shortcut_opt is not None else "C",
                state_dict=state_a,
            )
        elif model_name == "mlp":
            ps = mlp_permutation_spec_from_state(state_a)
        else:
            raise ValueError(f"Unsupported model '{model_name}'")

        # Make permutations complete wrt spec (fill missing perms with identity)
        wgt_full = complete_perm_dict(ps=ps, perm=wgt, state_a=state_a)
        act_full = complete_perm_dict(ps=ps, perm=act, state_a=state_a)

        # Permute full B under each method
        b_wgt_perm = apply_permutation(ps, wgt_full, state_b)
        b_act_perm = apply_permutation(ps, act_full, state_b)
    else:
        wgt_full = dict(wgt)
        act_full = dict(act)
        b_wgt_perm = {}
        b_act_perm = {}

    # Compute features for dH (only if we have checkpoints + dataset)
    feats_a: Dict[str, torch.Tensor] = {}
    feats_b: Dict[str, torch.Tensor] = {}
    do_feats = False
    if do_state and args.dataset is not None:
        do_feats = True
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        dtype = torch.float16 if args.features_dtype == "float16" else torch.float32
        n_samp = int(args.features_samples)
        n_samp = n_samp if n_samp > 0 else 0  # 0 => full split

        shortcut_opt = args.shortcut_option if args.shortcut_option is not None else None
        width_mult = int(args.width_multiplier) if args.width_multiplier is not None else None

        if ps is None:
            raise RuntimeError("Internal error: ps is None while do_state=True.")

        feats_a, feats_b = compute_features_from_state_dicts(
            model_name=model_name,
            dataset=str(args.dataset),
            data_root=str(args.data_root),
            split=str(args.features_split),
            samples=n_samp,
            batch_size=int(args.features_batch_size),
            num_workers=int(args.features_num_workers),
            state_a=state_a,
            state_b=state_b,
            ps=ps,
            norm=str(args.norm),
            shortcut_option=shortcut_opt,
            width_multiplier=width_mult,
            device=device,
            features_dtype=dtype,
        )

    report: Dict[str, Any] = {
        "act_perm_path": args.act_perm,
        "wgt_perm_path": args.wgt_perm,
        "key_reconciliation": strat,
        "model": model_name,
        "dataset_for_features": (str(args.dataset).strip().upper() if args.dataset else None),
        "num_common_keys": len(common_keys),
        "per_key": {},
    }

    print(f"[INFO] model={model_name} | common keys={len(common_keys)} | reconcile={strat}")
    if do_state:
        print("[INFO] dW enabled (checkpoints provided)")
    if do_feats:
        print("[INFO] dH enabled (dataset+checkpoints provided)")

    # Per-key
    for k in common_keys:
        p_act = act_full[k] if k in act_full else act[k]
        p_wgt = wgt_full[k] if k in wgt_full else wgt[k]

        if not is_valid_permutation(p_act):
            raise ValueError(f"[{k}] activation permutation is not valid.")
        if not is_valid_permutation(p_wgt):
            raise ValueError(f"[{k}] weight permutation is not valid.")
        if p_act.numel() != p_wgt.numel():
            raise ValueError(f"[{k}] size mismatch: {p_act.numel()} vs {p_wgt.numel()}")

        agreement = float((p_act == p_wgt).float().mean().item())
        q = compose_q(p_wgt, p_act)
        fixed = float((q == torch.arange(q.numel())).float().mean().item())
        cs = cycle_summary(q)

        entry: Dict[str, Any] = {
            "n": int(p_act.numel()),
            "hamming_agreement": agreement,
            "fixed_point_fraction_of_Q": fixed,
            "cycle_summary_of_Q": cs,
        }

        # ---- WEIGHTS: baseline=wgt_perm, cross=act_perm ----
        if do_state and ps is not None:
            dw_wgt = dW_for_perm_key(perm_key=k, ps=ps, state_a=state_a, state_b_perm=b_wgt_perm)
            dw_act = dW_for_perm_key(perm_key=k, ps=ps, state_a=state_a, state_b_perm=b_act_perm)
            entry["weights_alignment"] = {
                "dW_under_wgt_perm_baseline": dw_wgt,
                "dW_under_act_perm_cross": dw_act,
                "cross_over_baseline_ratio_rel": _safe_ratio(dw_act["rel_to_A"], dw_wgt["rel_to_A"]),
                "cross_over_baseline_ratio_raw": _safe_ratio(dw_act["raw_frob"], dw_wgt["raw_frob"]),
                "delta_cross_minus_baseline_rel": float(dw_act["rel_to_A"] - dw_wgt["rel_to_A"]),
                "delta_cross_minus_baseline_raw": float(dw_act["raw_frob"] - dw_wgt["raw_frob"]),
                "cross_vs_baseline_improvement_percent_rel": float(
                    (1.0 - _safe_ratio(dw_act["rel_to_A"], dw_wgt["rel_to_A"])) * 100.0
                ),
            }

        # ---- ACTIVATIONS: baseline=act_perm, cross=wgt_perm ----
        if do_feats:
            dh_act = dH_for_perm_key(
                perm_key=k, feats_a=feats_a, feats_b=feats_b, p_a_to_b=p_act, unit_dim=int(args.unit_dim)
            )
            dh_wgt = dH_for_perm_key(
                perm_key=k, feats_a=feats_a, feats_b=feats_b, p_a_to_b=p_wgt, unit_dim=int(args.unit_dim)
            )
            entry["activations_alignment"] = {
                "dH_under_act_perm_baseline": dh_act,
                "dH_under_wgt_perm_cross": dh_wgt,
                "cross_over_baseline_ratio_rel": _safe_ratio(dh_wgt["rel_to_A"], dh_act["rel_to_A"]),
                "cross_over_baseline_ratio_raw": _safe_ratio(dh_wgt["raw_frob"], dh_act["raw_frob"]),
                "delta_cross_minus_baseline_rel": float(dh_wgt["rel_to_A"] - dh_act["rel_to_A"]),
                "delta_cross_minus_baseline_raw": float(dh_wgt["raw_frob"] - dh_act["raw_frob"]),
                "cross_vs_baseline_improvement_percent_rel": float(
                    (1.0 - _safe_ratio(dh_wgt["rel_to_A"], dh_act["rel_to_A"])) * 100.0
                ),
            }

        report["per_key"][k] = entry

        print(
            f"[{k}] n={entry['n']} agree={agreement:.4f} fixed(Q)={fixed:.4f} "
            f"cycles={cs['num_cycles']} max_cycle={cs['max_cycle']} cayley={cs['cayley_distance']}"
        )
        if "weights_alignment" in entry:
            r = entry["weights_alignment"]["cross_over_baseline_ratio_rel"]
            print(f"      weights:  cross(act)/base(wgt) ratio(rel)={r:.4f}  ( <1 means ACT helps weights )")
        if "activations_alignment" in entry:
            r = entry["activations_alignment"]["cross_over_baseline_ratio_rel"]
            print(f"      acts:     cross(wgt)/base(act) ratio(rel)={r:.4f}  ( <1 means WGT helps activations )")

    # Aggregate summary
    avg_agree = float(np.mean([report["per_key"][k]["hamming_agreement"] for k in common_keys]))
    avg_fixed = float(np.mean([report["per_key"][k]["fixed_point_fraction_of_Q"] for k in common_keys]))
    report["summary"] = {"mean_hamming_agreement": avg_agree, "mean_fixed_point_fraction_Q": avg_fixed}
    print(f"[SUMMARY] mean agreement={avg_agree:.4f} | mean fixed(Q)={avg_fixed:.4f}")

    cross_summary: Dict[str, Any] = {}
    if do_state and ps is not None:
        ratios = []
        for k in common_keys:
            e = report["per_key"][k]
            if "weights_alignment" not in e:
                continue
            ratios.append(float(e["weights_alignment"]["cross_over_baseline_ratio_rel"]))
        if ratios:
            cross_summary["weights_cross_over_wgt_baseline_ratio_rel_mean"] = float(np.mean(ratios))
            print(f"[CROSS][weights] mean ratio(act->weights / wgt->weights)={cross_summary['weights_cross_over_wgt_baseline_ratio_rel_mean']:.4f}")

    if do_feats:
        ratios = []
        for k in common_keys:
            e = report["per_key"][k]
            if "activations_alignment" not in e:
                continue
            ratios.append(float(e["activations_alignment"]["cross_over_baseline_ratio_rel"]))
        if ratios:
            cross_summary["activations_cross_over_act_baseline_ratio_rel_mean"] = float(np.mean(ratios))
            print(f"[CROSS][acts] mean ratio(wgt->acts / act->acts)={cross_summary['activations_cross_over_act_baseline_ratio_rel_mean']:.4f}")

    if cross_summary:
        report["cross_objective_summary"] = cross_summary

    global_none = global_weight_distance(state_a, state_b)
    global_wgt  = global_weight_distance(state_a, b_wgt_perm)
    global_act  = global_weight_distance(state_a, b_act_perm)

    report["global_weights"] = {
        "no_perm": global_none,
        "wgt_perm": global_wgt,
        "act_perm": global_act,
        "wgt_over_no_perm_ratio_rel": _safe_ratio(global_wgt["rel_to_A"], global_none["rel_to_A"]),
        "act_over_no_perm_ratio_rel": _safe_ratio(global_act["rel_to_A"], global_none["rel_to_A"]),
        "wgt_improvement_vs_no_perm_percent_rel": float((1.0 - _safe_ratio(global_wgt["rel_to_A"], global_none["rel_to_A"])) * 100.0),
        "act_improvement_vs_no_perm_percent_rel": float((1.0 - _safe_ratio(global_act["rel_to_A"], global_none["rel_to_A"])) * 100.0),
    }

    if args.out_json:
        os.makedirs(os.path.dirname(os.path.abspath(args.out_json)), exist_ok=True)
        with open(args.out_json, "w") as f:
            json.dump(report, f, indent=2)
        print(f"[INFO] wrote {args.out_json}")


if __name__ == "__main__":
    main()