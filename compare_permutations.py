#!/usr/bin/env python3
"""
compare_permutations.py

Compare permutations found by:
  - weight matching (expects .pkl saved as {perm_name: np.ndarray})
  - activation matching (accepts .json / .pt / .pkl)

Metrics per perm key (layer / perm group):

(A) Combinatorial agreement between permutation sets:
    - Hamming agreement between P_wgt and P_act
    - Fixed-point fraction of Q = inv(P_wgt) o P_act
    - Cycle structure of Q
    - Cayley distance = n - #cycles

(B) Representation alignment quality of each permutation set (optional, needs features):
    - baseline alignment (no perm):      A vs B
    - alignment under weight perm:       A vs permute(B, P_wgt)
    - alignment under activation perm:   A vs permute(B, P_act)
    - deltas vs baseline and vs each other
    - optional: alignment after permuting B's *weights* and re-forwarding

Notes on permutation convention:
  p[i] = j means "A unit i matches B unit j" (A -> B assignment).
  To align B into A indexing, use xb[:, p].
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import re
from typing import Any, Dict, List, Tuple, Optional

import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

from linear_mode_connectivity.weight_matching_torch import (
    resnet20_layernorm_permutation_spec,
    permutation_spec_from_axes_to_perm,
    apply_permutation,
)

import architectures

from model_stitching.resnet20_activation_stitching import (
    infer_width_multiplier_from_state,
    infer_shortcut_option_from_state,
    perm_name_to_hook,
)

# -------------------------
# Optional: import CKA from "elsewhere"
# -------------------------
def _load_cka_fn():
    """
    Try to import a cka(x, y) callable from your repo.
    If you keep it somewhere else, just change this function.
    """
    try:
        from metrics_platonic import AlignmentMetrics  # type: ignore
        return AlignmentMetrics.cka
    except Exception:
        pass

    try:
        from cka import cka  # type: ignore
        return cka
    except Exception as e:
        raise ImportError(
            "Could not import a CKA function. Edit _load_cka_fn() to point to your implementation."
        ) from e


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
    Support a few common wrappers:
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


def _normalize_perm_dict(obj: Any, src_path: str) -> Dict[str, torch.Tensor]:
    if not isinstance(obj, dict):
        raise ValueError(f"Expected dict-like permutations in {src_path}, got {type(obj)}")
    out: Dict[str, torch.Tensor] = {}
    for k, v in obj.items():
        out[str(k)] = _to_long_1d(v)
    return out


def load_permutations(path: str) -> Dict[str, torch.Tensor]:
    raw = _load_any(path)
    payload = _extract_perm_payload(raw, path)
    return _normalize_perm_dict(payload, path)


# -------------------------
# Permutation math
# -------------------------
def is_valid_permutation(p: torch.Tensor) -> bool:
    n = int(p.numel())
    if n == 0:
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
# Key reconciliation (MLP: "1,2,3" vs "P1,P2,P3")
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
# Features helpers
# -------------------------
def activation_to_2d(out: torch.Tensor, unit_dim: int = 1) -> torch.Tensor:
    if out.ndim < 2:
        raise ValueError(f"Expected activation with ndim>=2, got {tuple(out.shape)}")
    if unit_dim < 0:
        unit_dim = out.ndim + unit_dim
    x = torch.movedim(out, unit_dim, -1)
    return x.reshape(-1, x.shape[-1])


def load_features(path: str) -> Dict[str, torch.Tensor]:
    raw = _load_any(path)
    if isinstance(raw, dict) and "features" in raw and isinstance(raw["features"], dict):
        raw = raw["features"]
    if not isinstance(raw, dict):
        raise ValueError(f"Expected a dict of features in {path}, got {type(raw)}")
    out: Dict[str, torch.Tensor] = {}
    for k, v in raw.items():
        if not isinstance(v, torch.Tensor):
            v = torch.tensor(v)
        out[str(k)] = v.detach().cpu()
    return out


_NUM_WEIGHT = re.compile(r"^(?:(?:net|model|module)\.)?(\d+)\.weight$")
_NUM_BIAS   = re.compile(r"^(?:(?:net|model|module)\.)?(\d+)\.bias$")

def canonicalize_sequential_mlp_to_fc(state: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """
    Map Sequential-style Linear layer keys like:
      '0.weight','0.bias','2.weight','2.bias', ...
    into:
      'fc1.weight','fc1.bias','fc2.weight','fc2.bias', ...
    based on sorted numeric layer indices.

    Only triggers if:
      - no 'fc1.weight' is present
      - and we can find >=1 2D weight tensor under numeric keys
    """
    if "fc1.weight" in state:
        return state

    # find candidate numeric "Linear" weights (2D)
    idxs = []
    for k, v in state.items():
        m = _NUM_WEIGHT.match(k)
        if m and isinstance(v, torch.Tensor) and v.ndim == 2:
            idxs.append(int(m.group(1)))
    idxs = sorted(set(idxs))
    if not idxs:
        return state

    idx_to_fc = {idx: (i + 1) for i, idx in enumerate(idxs)}

    out: Dict[str, torch.Tensor] = {}
    for k, v in state.items():
        mw = _NUM_WEIGHT.match(k)
        if mw and int(mw.group(1)) in idx_to_fc and v.ndim == 2:
            j = idx_to_fc[int(mw.group(1))]
            out[f"fc{j}.weight"] = v
            continue

        mb = _NUM_BIAS.match(k)
        if mb and int(mb.group(1)) in idx_to_fc:
            j = idx_to_fc[int(mb.group(1))]
            out[f"fc{j}.bias"] = v
            continue

        out[k] = v

    return out

# -------------------------
# State dict helpers + alignment metrics
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
    out = canonicalize_sequential_mlp_to_fc(out)
    return out


def infer_mlp_fc_layer_numbers(state: Dict[str, torch.Tensor]) -> List[int]:
    pat = re.compile(r"^fc(\d+)\.weight$")
    layers: List[int] = []
    for k in state.keys():
        m = pat.match(k)
        if m:
            layers.append(int(m.group(1)))
    layers = sorted(set(layers))
    if not layers:
        raise KeyError("No fc{n}.weight keys found in checkpoint state_dict.")
    return layers


def mlp_permutation_spec_from_state(state: Dict[str, torch.Tensor]):
    layer_ids = infer_mlp_fc_layer_numbers(state)
    n_layers = max(layer_ids)

    expected = list(range(1, n_layers + 1))
    if layer_ids != expected:
        raise ValueError(f"Expected contiguous fc layers {expected}, found {layer_ids}")

    axes: Dict[str, Tuple[Optional[str], ...]] = {}

    prev_p: Optional[str] = None
    for i in range(1, n_layers):
        p_out = f"P{i}"
        w_key = f"fc{i}.weight"
        b_key = f"fc{i}.bias"
        axes[w_key] = (p_out, prev_p)
        axes[b_key] = (p_out,)
        prev_p = p_out

    last_w = f"fc{n_layers}.weight"
    last_b = f"fc{n_layers}.bias"
    axes[last_w] = (None, prev_p)
    axes[last_b] = (None,)

    axes = {k: v for k, v in axes.items() if k in state}
    return permutation_spec_from_axes_to_perm(axes)


def frob(x: torch.Tensor) -> float:
    return float(torch.linalg.norm(x).item())


def identity_perm(d: int, device: Optional[torch.device] = None) -> torch.Tensor:
    return torch.arange(d, dtype=torch.long, device=device)


def activation_alignment_error(xa2: torch.Tensor, xb2: torch.Tensor, p: torch.Tensor) -> Dict[str, float]:
    diff = xa2 - xb2[:, p]
    raw = frob(diff)
    rel = raw / (frob(xa2) + 1e-12)
    return {"raw_frob": raw, "rel_to_A": rel}


def mean_unit_cosine(xa2: torch.Tensor, xb2_aligned: torch.Tensor) -> float:
    # cosine between corresponding unit columns (vectors of length M)
    num = (xa2 * xb2_aligned).sum(dim=0)
    den = xa2.norm(dim=0) * xb2_aligned.norm(dim=0) + 1e-12
    return float((num / den).mean().item())


def repr_alignment_metrics(
    xa2: torch.Tensor,
    xb2: torch.Tensor,
    *,
    p: Optional[torch.Tensor],
    cka_fn=None,
) -> Dict[str, Any]:
    d = int(xa2.shape[1])
    if xb2.shape[1] != d:
        return {"skipped": True, "reason": f"dim mismatch xa2={tuple(xa2.shape)} xb2={tuple(xb2.shape)}"}

    p_use = identity_perm(d, device=xb2.device) if p is None else p.to(dtype=torch.long, device=xb2.device)
    xb_aligned = xb2[:, p_use]

    out: Dict[str, Any] = {}
    out["dH"] = activation_alignment_error(xa2, xb2, p_use)
    out["mean_unit_cos"] = mean_unit_cosine(xa2, xb_aligned)

    if cka_fn is not None:
        v = cka_fn(xa2, xb_aligned)
        out["cka"] = float(v.item() if isinstance(v, torch.Tensor) else v)

    return out


def dW_for_perm_key(
    *,
    perm_key: str,
    ps,
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


def print_frobenius_norms_model_a(state_a: Dict[str, torch.Tensor]) -> None:
    """
    Compute and print the Frobenius norm of weights of model A at each layer.
    Organizes output by layer for both MLP (fc layers) and ResNet20 architectures.
    """
    if not state_a:
        print("[INFO] state_a is empty, skipping Frobenius norm computation")
        return
    
    print("\n[INFO] Frobenius Norms of Model A Weights by Layer:")
    print("-" * 60)
    
    # Organize weights by layer
    layer_norms: Dict[str, Dict[str, float]] = {}
    
    for key, tensor in sorted(state_a.items()):
        if "weight" not in key:
            continue
        
        t = tensor.float()
        norm = float(torch.linalg.norm(t).item())
        
        # Extract layer identifier
        if "fc" in key:  # MLP layers (fc1, fc2, fc3, etc.)
            match = re.match(r"fc(\d+)\.weight", key)
            if match:
                layer_id = f"FC{match.group(1)}"
            else:
                layer_id = key
        elif "layer" in key:  # ResNet layers (layer1, layer2, layer3)
            match = re.match(r"layer(\d+)\.(\d+)\.", key)
            if match:
                layer_num = match.group(1)
                block_num = match.group(2)
                layer_id = f"Layer{layer_num}_Block{block_num}"
            else:
                layer_id = key
        elif "conv" in key:  # Initial conv layer
            layer_id = "ConvInit"
        elif "linear" in key or "classifier" in key:
            layer_id = "OutputLinear"
        else:
            layer_id = key
        
        if layer_id not in layer_norms:
            layer_norms[layer_id] = {}
        layer_norms[layer_id][key] = norm
    
    # Print organized results
    for layer_id in sorted(layer_norms.keys()):
        print(f"\n{layer_id}:")
        for param_name in sorted(layer_norms[layer_id].keys()):
            norm = layer_norms[layer_id][param_name]
            print(f"  {param_name:50s} : {norm:12.6e}")
    
    # Print overall statistics
    all_norms = [v for layer_dict in layer_norms.values() for v in layer_dict.values()]
    if all_norms:
        print(f"\n{'Overall Statistics':50s}")
        print(f"  {'Mean norm':50s} : {np.mean(all_norms):12.6e}")
        print(f"  {'Max norm':50s} : {np.max(all_norms):12.6e}")
        print(f"  {'Min norm':50s} : {np.min(all_norms):12.6e}")
    print("-" * 60 + "\n")


# -------------------------
# ε-LLFC helpers
# -------------------------
def _parse_lambdas(lams: str) -> List[float]:
    """
    Parse comma-separated lambdas like "0,0.25,0.5,0.75,1".
    """
    out: List[float] = []
    for s in lams.split(","):
        s = s.strip()
        if not s:
            continue
        v = float(s)
        if v < 0.0 or v > 1.0:
            raise ValueError(f"--llfc-lambdas must be in [0,1], got {v}")
        out.append(v)
    if not out:
        raise ValueError("--llfc-lambdas parsed to empty list")
    return out


def interpolate_state_dict(
    a: Dict[str, torch.Tensor],
    b: Dict[str, torch.Tensor],
    lam: float,
) -> Dict[str, torch.Tensor]:
    """
    Weight-space linear interpolation: θ_λ = (1-λ)θ_A + λθ_B.

    For floating tensors -> interpolate.
    For non-floating buffers (ints, counters, etc.) -> keep A's value.
    """
    out: Dict[str, torch.Tensor] = {}
    keys = set(a.keys()).intersection(set(b.keys()))
    # Keep any A-only keys as-is (should be rare if checkpoints match)
    for k in a.keys():
        if k not in keys:
            out[k] = a[k]
    for k in keys:
        va, vb = a[k], b[k]
        if va.shape != vb.shape:
            raise ValueError(f"Shape mismatch in state dict at key '{k}': {tuple(va.shape)} vs {tuple(vb.shape)}")
        if va.dtype.is_floating_point:
            out[k] = (1.0 - lam) * va + lam * vb
        else:
            out[k] = va
    return out


def _llfc_abs_rel(
    *,
    ha2: torch.Tensor,
    hb2: torch.Tensor,
    hl2: torch.Tensor,
    lam: float,
    tau: float,
) -> Tuple[float, float]:
    """
    Absolute ε-LLFC deviation:
      || H_λ - ((1-λ)H_A + λH_B) ||_F

    Relative variant (your later appendix normalization):
      abs / ( (1-λ)||H_A||_F + λ||H_B||_F + tau )
    """
    # Ensure float for stable norms
    ha2 = ha2.float()
    hb2 = hb2.float()
    hl2 = hl2.float()

    target = (1.0 - lam) * ha2 + lam * hb2
    diff = hl2 - target
    abs_err = float(torch.linalg.norm(diff).item())

    scale = (1.0 - lam) * float(torch.linalg.norm(ha2).item()) + lam * float(torch.linalg.norm(hb2).item()) + float(tau)
    rel_err = abs_err / scale
    return abs_err, rel_err


def _summarize_scalar_list(xs: List[float]) -> Dict[str, float]:
    if len(xs) == 0:
        return {"mean": float("nan"), "max": float("nan")}
    return {"mean": float(np.mean(xs)), "max": float(np.max(xs))}


def _format_f(x: float) -> str:
    if x != x:  # NaN
        return "nan"
    return f"{x:.4g}"


# -------------------------
# Generic FC MLP Model
# -------------------------
class GenericFCMLP(nn.Module):
    def __init__(self, layer_sizes: List[int]):
        super().__init__()
        # layer_sizes = [in_dim, h1, h2, ..., out_dim]
        self.n_layers = len(layer_sizes) - 1
        for i in range(1, self.n_layers + 1):
            setattr(self, f"fc{i}", nn.Linear(layer_sizes[i - 1], layer_sizes[i]))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.view(x.size(0), -1)
        for i in range(1, self.n_layers):
            x = F.relu(getattr(self, f"fc{i}")(x))
        x = getattr(self, f"fc{self.n_layers}")(x)
        return x


# -------------------------
# Feature extraction from checkpoints (ResNet20 + MLP)
# -------------------------
_P_INNER_HOOK = re.compile(r"^P_layer(\d+)_(\d+)_inner$")


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
    dataset: str,
    data_root: str,
    split: str,
    samples: int,
    batch_size: int,
    num_workers: int,
    state_a: Dict[str, torch.Tensor],
    state_b: Dict[str, torch.Tensor],
    ps,
    shortcut_option: str,
    width_multiplier: Optional[int],
    device: torch.device,
    features_dtype: torch.dtype,
    return_only_a: bool = False,
) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
    import math
    import re as _re
    import datasets as ds_utils

    dataset_u = dataset.strip().upper()
    if dataset_u not in ds_utils.DATASET_STATS:
        raise ValueError(f"Unsupported dataset: {dataset} (supported: {sorted(ds_utils.DATASET_STATS)})")

    _train_full, eval_full, test_ds = ds_utils.build_datasets(
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
        raise ValueError(f"Unsupported split for features: {split} (use 'test' or 'train_eval')")

    if samples is not None and int(samples) <= 0:
        samples = None
    if samples is not None and samples < len(ds):
        ds = Subset(ds, list(range(int(samples))))

    loader = DataLoader(
        ds,
        batch_size=int(batch_size),
        shuffle=False,
        num_workers=int(num_workers),
        pin_memory=(device.type == "cuda"),
    )

    has_conv = ("conv1.weight" in state_a) or any(k.startswith("layer1.") for k in state_a.keys())
    has_fc1 = ("fc1.weight" in state_a) or any(k.startswith("fc1.") for k in state_a.keys())

    if has_fc1 and not has_conv:
        arch = "mlp"
    elif has_conv:
        arch = "resnet20"
    else:
        raise ValueError(
            "Could not infer architecture from state_dict keys. "
            "Expected either ResNet-like keys (conv1/layer*) or MLP keys (fc1..)."
        )

    perm_names = sorted(ps.perm_to_axes.keys())

    if arch == "mlp":
        pat = _re.compile(r"^fc(\d+)\.weight$")
        layer_ids = sorted({int(pat.match(k).group(1)) for k in state_a.keys() if pat.match(k)})
        if not layer_ids:
            raise KeyError("MLP detected, but no fc{n}.weight keys found in state_dict.")
        n_layers = max(layer_ids)
        expected = list(range(1, n_layers + 1))
        if layer_ids != expected:
            raise ValueError(f"Expected contiguous fc layers {expected}, found {layer_ids}")

        hidden = int(state_a["fc1.weight"].shape[0])
        in_dim = int(state_a["fc1.weight"].shape[1])

        stats = ds_utils.DATASET_STATS[dataset_u]
        c = int(stats["in_channels"])
        h, w = map(int, stats["image_size"])
        if c * h * w != in_dim:
            if in_dim % c != 0:
                raise ValueError(f"Cannot infer input_shape: fc1.weight in_dim={in_dim} not divisible by in_channels={c}")
            side_f = math.sqrt(in_dim / c)
            side = int(round(side_f))
            if side * side * c != in_dim:
                raise ValueError("Cannot infer square input_shape from fc1.weight")
            h = w = side

        # infer layer sizes from checkpoint
        layer_sizes = [in_dim] + [int(state_a[f"fc{i}.weight"].shape[0]) for i in range(1, n_layers + 1)]

        model_a = GenericFCMLP(layer_sizes).to(device).eval()
        if not return_only_a:
            model_b = GenericFCMLP(layer_sizes).to(device).eval()

        model_a.load_state_dict({k: v for k, v in state_a.items() if k in model_a.state_dict()}, strict=True)
        if not return_only_a:
            model_b.load_state_dict({k: v for k, v in state_b.items() if k in model_b.state_dict()}, strict=True)

        _P_OR_DIGIT = _re.compile(r"^(?:P)?(\d+)$")

        def _hook_info(pname: str):
            m = _P_OR_DIGIT.match(pname)
            if not m:
                raise KeyError(f"MLP: can't map perm name to hook: {pname}")
            i = int(m.group(1))
            if i < 1 or i >= n_layers:
                raise KeyError(f"MLP: perm {pname} implies fc{i}, but only hidden perms 1..{n_layers-1} exist.")
            return f"fc{i}", F.relu

    else:
        if "linear.weight" in state_a:
            num_classes = int(state_a["linear.weight"].shape[0])
        else:
            num_classes = int(ds_utils.DATASET_STATS[dataset_u]["num_classes"])

        if width_multiplier is None:
            width_multiplier = infer_width_multiplier_from_state(state_a)

        model_a = architectures.build_model(
            "resnet20",
            num_classes=num_classes,
            norm="flax_ln",
            width_multiplier=int(width_multiplier),
            shortcut_option=str(shortcut_option),
        ).to(device).eval()
        if not return_only_a:
            model_b = architectures.build_model(
                "resnet20",
                num_classes=num_classes,
                norm="flax_ln",
                width_multiplier=int(width_multiplier),
                shortcut_option=str(shortcut_option),
            ).to(device).eval()

        keys = set(model_a.state_dict().keys())
        model_a.load_state_dict({k: v for k, v in state_a.items() if k in keys}, strict=True)
        if not return_only_a:
            model_b.load_state_dict({k: v for k, v in state_b.items() if k in keys}, strict=True)

        def _hook_info(pname: str):
            layer_name, preprocess, _unit_dim = perm_name_to_hook(pname)
            return layer_name, preprocess

    def _extract(model: nn.Module) -> Dict[str, torch.Tensor]:
        store: Dict[str, List[torch.Tensor]] = {p: [] for p in perm_names}
        hooks = []

        for p in perm_names:
            layer_name, preprocess = _hook_info(p)
            mod = _get_submodule_by_name(model, layer_name)

            def _make_hook(pname: str, pre):
                def _hook(_m, _inp, out):
                    x = out.detach()
                    if pre is not None:
                        x = pre(x)
                    store[pname].append(x.to(device="cpu", dtype=features_dtype))
                return _hook

            hooks.append(mod.register_forward_hook(_make_hook(p, preprocess)))

        seen = 0
        for xb, _yb in loader:
            xb = xb.to(device, non_blocking=True)
            _ = model(xb)
            seen += int(xb.shape[0])
            if samples is not None and seen >= samples:
                break

        for h in hooks:
            h.remove()

        out_feats: Dict[str, torch.Tensor] = {}
        for p in perm_names:
            if len(store[p]) == 0:
                continue
            t = torch.cat(store[p], dim=0)
            if samples is not None and t.shape[0] > samples:
                t = t[:samples]
            out_feats[p] = t
        return out_feats

    feats_a = _extract(model_a)
    if return_only_a:
        return feats_a, {}
    feats_b = _extract(model_b)
    return feats_a, feats_b


# -------------------------
# Main
# -------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--act-perm", type=str, default=None)
    ap.add_argument("--wgt-perm", type=str, default=None)
    ap.add_argument("--out-json", type=str, default=None)

    ap.add_argument("--features-a", type=str, default=None)
    ap.add_argument("--features-b", type=str, default=None)
    ap.add_argument("--unit-dim", type=int, default=1)

    ap.add_argument("--state-a", type=str, default=None)
    ap.add_argument("--state-b", type=str, default=None)
    ap.add_argument("--shortcut-option", type=str, default="C", choices=["A", "B", "C"])

    ap.add_argument("--dataset", type=str, default=None, choices=["CIFAR10", "CIFAR100","MNIST", "FASHIONMNIST"])
    ap.add_argument("--data-root", type=str, default="./data")
    ap.add_argument("--features-split", type=str, default="test", choices=["test", "train_eval"])
    ap.add_argument("--features-samples", type=int, default=512)
    ap.add_argument("--features-batch-size", type=int, default=256)
    ap.add_argument("--features-num-workers", type=int, default=0)
    ap.add_argument("--features-dtype", type=str, default="float32", choices=["float16", "float32"])
    ap.add_argument("--width-multiplier", type=int, default=None)

    ap.add_argument(
        "--compare-permuted-weight-model",
        action="store_true",
        help="If set (and dataset+state_a/state_b provided), also forward the permuted-B models (WM/AM) "
             "and report repr alignment without feature-level permuting.",
    )
    ap.add_argument(
        "--compute-llfc",
        action="store_true",
        help="Compute ε-LLFC for raw (A,B) and for A vs permuted-B (wgt/act), over a lambda grid.",
    )
    ap.add_argument(
        "--llfc-lambdas",
        type=str,
        default="0.5",
        help='Comma-separated lambdas in [0,1], e.g. "0,0.1,0.2,...,1".',
    )
    ap.add_argument(
        "--llfc-tau",
        type=float,
        default=1e-8,
        help="Stability constant τ used in relative ε-LLFC normalization.",
    )

    args = ap.parse_args()

    # Load permutations (optional - only if both act and wgt perm paths provided)
    have_perms = (args.act_perm is not None) and (args.wgt_perm is not None)
    act: Dict[str, torch.Tensor] = {}
    wgt: Dict[str, torch.Tensor] = {}
    keys: List[str] = []
    strat = "none"

    if have_perms:
        act = load_permutations(args.act_perm)
        wgt = load_permutations(args.wgt_perm)
        act, wgt, strat = reconcile_keys(act, wgt)
        keys = sorted(set(act).intersection(set(wgt)))
        if not keys:
            raise RuntimeError(
                "No overlapping permutation keys between activation and weight matching.\n"
                f"Reconciliation strategy tried: {strat}\n"
                f"Activation keys (sample): {list(act.keys())[:10]}\n"
                f"Weight keys (sample): {list(wgt.keys())[:10]}"
            )
    else:
        # If no permutations provided, just compute norms and exit
        if (args.state_a is not None) and (args.state_b is not None):
            state_a = load_state_dict_any(args.state_a)
            print_frobenius_norms_model_a(state_a)
            return
        else:
            raise ValueError("Either provide --act-perm and --wgt-perm, or provide --state-a and --state-b for norm computation.")

    report: Dict[str, Any] = {
        "act_perm_path": args.act_perm,
        "wgt_perm_path": args.wgt_perm,
        "key_reconciliation": strat,
        "num_common_keys": len(keys),
        "per_key": {},
    }

    # ---- features / CKA ----
    feats_a: Dict[str, torch.Tensor] = {}
    feats_b: Dict[str, torch.Tensor] = {}
    do_cka = False
    cka_fn = None

    have_feat_files = (args.features_a is not None) and (args.features_b is not None)
    if have_feat_files:
        do_cka = True
        cka_fn = _load_cka_fn()
        feats_a = load_features(args.features_a)
        feats_b = load_features(args.features_b)

    # ---- state dict / ps ----
    do_state = (args.state_a is not None) and (args.state_b is not None)
    state_a = load_state_dict_any(args.state_a) if do_state else {}
    state_b = load_state_dict_any(args.state_b) if do_state else {}

    # Compute and print Frobenius norms of model A's weights
    if do_state:
        print_frobenius_norms_model_a(state_a)

    ps = None
    b_wgt_perm: Dict[str, torch.Tensor] = {}
    b_act_perm: Dict[str, torch.Tensor] = {}

    feats_b_from_wgt_model: Dict[str, torch.Tensor] = {}
    feats_b_from_act_model: Dict[str, torch.Tensor] = {}

    if do_state:
        has_conv = ("conv1.weight" in state_a) or any(k.startswith("layer1.") for k in state_a.keys())
        has_fc1 = ("fc1.weight" in state_a) or any(k.startswith("fc1.") for k in state_a.keys())

        if has_fc1 and not has_conv:
            ps = mlp_permutation_spec_from_state(state_a)
        else:
            ps = resnet20_layernorm_permutation_spec(
                shortcut_option=str(args.shortcut_option),
                state_dict=state_a,
            )

        # Filter keys to only those that exist in the permutation spec
        # (handles case where permutation files have more keys than the network has layers)
        valid_keys = set(ps.perm_to_axes.keys())
        keys_before = set(keys)
        keys = sorted(keys_before.intersection(valid_keys))
        if keys != list(keys_before):
            excluded = keys_before - valid_keys
            print(f"[INFO] Filtered out {len(excluded)} invalid permutation keys: {sorted(excluded)}")
            print(f"[INFO] Processing {len(keys)} valid keys: {keys}")

        b_wgt_perm = apply_permutation(ps, wgt, state_b)
        b_act_perm = apply_permutation(ps, act, state_b)

        # ---- ε-LLFC computation ----
        llfc_per_key: Dict[str, Any] = {}
        llfc_summary: Dict[str, Any] = {}

        if args.compute_llfc:
            if not do_state or ps is None:
                raise ValueError("--compute-llfc requires --state-a and --state-b (so we can interpolate weights).")
            if args.dataset is None:
                raise ValueError("--compute-llfc requires --dataset (so we can forward interpolated models).")

            lambdas = _parse_lambdas(str(args.llfc_lambdas))
            tau = float(args.llfc_tau)

            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            dtype = torch.float16 if args.features_dtype == "float16" else torch.float32
            n_samp = int(args.features_samples)
            n_samp = n_samp if n_samp > 0 else 0

            shortcut_opt = str(args.shortcut_option) if args.shortcut_option else infer_shortcut_option_from_state(state_a)
            width_mult = int(args.width_multiplier) if args.width_multiplier is not None else None

            # Always compute endpoint features from states to ensure consistent dataset/split/samples.
            featsA, featsB_raw = compute_features_from_state_dicts(
                dataset=str(args.dataset),
                data_root=str(args.data_root),
                split=str(args.features_split),
                samples=n_samp,
                batch_size=int(args.features_batch_size),
                num_workers=int(args.features_num_workers),
                state_a=state_a,
                state_b=state_b,
                ps=ps,
                shortcut_option=shortcut_opt,
                width_multiplier=width_mult,
                device=device,
                features_dtype=dtype,
            )

            # B aligned by permuting weights (so weight-space interpolation is meaningful)
            _, featsB_wgt = compute_features_from_state_dicts(
                dataset=str(args.dataset),
                data_root=str(args.data_root),
                split=str(args.features_split),
                samples=n_samp,
                batch_size=int(args.features_batch_size),
                num_workers=int(args.features_num_workers),
                state_a=state_a,
                state_b=b_wgt_perm,
                ps=ps,
                shortcut_option=shortcut_opt,
                width_multiplier=width_mult,
                device=device,
                features_dtype=dtype,
            )
            _, featsB_act = compute_features_from_state_dicts(
                dataset=str(args.dataset),
                data_root=str(args.data_root),
                split=str(args.features_split),
                samples=n_samp,
                batch_size=int(args.features_batch_size),
                num_workers=int(args.features_num_workers),
                state_a=state_a,
                state_b=b_act_perm,
                ps=ps,
                shortcut_option=shortcut_opt,
                width_multiplier=width_mult,
                device=device,
                features_dtype=dtype,
            )

            # Flatten once for endpoints
            common_keys = sorted(set(ps.perm_to_axes.keys()))
            A2 = {k: activation_to_2d(featsA[k], unit_dim=args.unit_dim) for k in common_keys if k in featsA}
            B2_raw = {k: activation_to_2d(featsB_raw[k], unit_dim=args.unit_dim) for k in common_keys if k in featsB_raw}
            B2_wgt = {k: activation_to_2d(featsB_wgt[k], unit_dim=args.unit_dim) for k in common_keys if k in featsB_wgt}
            B2_act = {k: activation_to_2d(featsB_act[k], unit_dim=args.unit_dim) for k in common_keys if k in featsB_act}

            def _run_variant(name: str, sb: Dict[str, torch.Tensor], B2: Dict[str, torch.Tensor]) -> Dict[str, Any]:
                abs_all: List[float] = []
                rel_all: List[float] = []
                per_k_abs: Dict[str, List[float]] = {k: [] for k in common_keys}
                per_k_rel: Dict[str, List[float]] = {k: [] for k in common_keys}

                for lam in lambdas:
                    s_lam = interpolate_state_dict(state_a, sb, lam)
                    feats_lam, _ = compute_features_from_state_dicts(
                        dataset=str(args.dataset),
                        data_root=str(args.data_root),
                        split=str(args.features_split),
                        samples=n_samp,
                        batch_size=int(args.features_batch_size),
                        num_workers=int(args.features_num_workers),
                        state_a=s_lam,
                        state_b=s_lam,
                        ps=ps,
                        shortcut_option=shortcut_opt,
                        width_multiplier=width_mult,
                        device=device,
                        features_dtype=dtype,
                        return_only_a=True,
                    )
                    for k in common_keys:
                        if k not in A2 or k not in B2 or k not in feats_lam:
                            continue
                        hl2 = activation_to_2d(feats_lam[k], unit_dim=args.unit_dim)
                        abs_err, rel_err = _llfc_abs_rel(ha2=A2[k], hb2=B2[k], hl2=hl2, lam=lam, tau=tau)
                        per_k_abs[k].append(abs_err)
                        per_k_rel[k].append(rel_err)
                        abs_all.append(abs_err)
                        rel_all.append(rel_err)

                # per-key summaries
                per_key = {}
                for k in common_keys:
                    per_key[k] = {
                        "abs": _summarize_scalar_list(per_k_abs[k]),
                        "rel": _summarize_scalar_list(per_k_rel[k]),
                    }

                return {
                    "name": name,
                    "lambdas": lambdas,
                    "tau": tau,
                    "overall": {
                        "abs": _summarize_scalar_list(abs_all),
                        "rel": _summarize_scalar_list(rel_all),
                    },
                    "per_key": per_key,
                }

            llfc_raw = _run_variant("raw", state_b, B2_raw)
            llfc_wgt = _run_variant("under_wgt_perm", b_wgt_perm, B2_wgt)
            llfc_act = _run_variant("under_act_perm", b_act_perm, B2_act)

            llfc_summary = {
                "lambdas": lambdas,
                "tau": tau,
                "raw": llfc_raw["overall"],
                "under_wgt_perm": llfc_wgt["overall"],
                "under_act_perm": llfc_act["overall"],
            }
            report["llfc"] = llfc_summary

            # Attach per-key (so it ends up in the output JSON per layer)
            for k in common_keys:
                llfc_per_key[k] = {
                    "raw": llfc_raw["per_key"].get(k, {}),
                    "under_wgt_perm": llfc_wgt["per_key"].get(k, {}),
                    "under_act_perm": llfc_act["per_key"].get(k, {}),
                }

            # Print a compact "final table"
            print("\n[LLFC] ε-LLFC summary (over layers in ps, over λ grid)")
            print(f"  lambdas = {lambdas}")
            print(f"  tau     = {tau:g}\n")

            header = ["metric", "raw", "wgt_perm", "act_perm"]
            rows = [
                ["max_abs", _format_f(llfc_raw["overall"]["abs"]["max"]), _format_f(llfc_wgt["overall"]["abs"]["max"]), _format_f(llfc_act["overall"]["abs"]["max"])],
                ["mean_abs", _format_f(llfc_raw["overall"]["abs"]["mean"]), _format_f(llfc_wgt["overall"]["abs"]["mean"]), _format_f(llfc_act["overall"]["abs"]["mean"])],
                ["max_rel", _format_f(llfc_raw["overall"]["rel"]["max"]), _format_f(llfc_wgt["overall"]["rel"]["max"]), _format_f(llfc_act["overall"]["rel"]["max"])],
                ["mean_rel", _format_f(llfc_raw["overall"]["rel"]["mean"]), _format_f(llfc_wgt["overall"]["rel"]["mean"]), _format_f(llfc_act["overall"]["rel"]["mean"])],
            ]

            colw = [max(len(str(x)) for x in col) for col in zip(header, *rows)]
            def _row(xs): return "  " + " | ".join(str(x).ljust(w) for x, w in zip(xs, colw))
            print(_row(header))
            print("  " + "-+-".join("-" * w for w in colw))
            for r in rows:
                print(_row(r))
            print()

        if (args.dataset is not None) and (not have_feat_files):
            do_cka = True
            cka_fn = _load_cka_fn()

            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            dtype = torch.float16 if args.features_dtype == "float16" else torch.float32
            n_samp = int(args.features_samples)
            n_samp = n_samp if n_samp > 0 else 0

            shortcut_opt = str(args.shortcut_option) if args.shortcut_option else infer_shortcut_option_from_state(state_a)

            feats_a, feats_b = compute_features_from_state_dicts(
                dataset=str(args.dataset),
                data_root=str(args.data_root),
                split=str(args.features_split),
                samples=n_samp,
                batch_size=int(args.features_batch_size),
                num_workers=int(args.features_num_workers),
                state_a=state_a,
                state_b=state_b,
                ps=ps,
                shortcut_option=shortcut_opt,
                width_multiplier=int(args.width_multiplier) if args.width_multiplier is not None else None,
                device=device,
                features_dtype=dtype,
            )

            if args.compare_permuted_weight_model:
                # Forward *permuted-weight* B models and capture features directly in A indexing
                _fa, feats_b_from_wgt_model = compute_features_from_state_dicts(
                    dataset=str(args.dataset),
                    data_root=str(args.data_root),
                    split=str(args.features_split),
                    samples=n_samp,
                    batch_size=int(args.features_batch_size),
                    num_workers=int(args.features_num_workers),
                    state_a=state_a,
                    state_b=b_wgt_perm,
                    ps=ps,
                    shortcut_option=shortcut_opt,
                    width_multiplier=int(args.width_multiplier) if args.width_multiplier is not None else None,
                    device=device,
                    features_dtype=dtype,
                )
                _fa, feats_b_from_act_model = compute_features_from_state_dicts(
                    dataset=str(args.dataset),
                    data_root=str(args.data_root),
                    split=str(args.features_split),
                    samples=n_samp,
                    batch_size=int(args.features_batch_size),
                    num_workers=int(args.features_num_workers),
                    state_a=state_a,
                    state_b=b_act_perm,
                    ps=ps,
                    shortcut_option=shortcut_opt,
                    width_multiplier=int(args.width_multiplier) if args.width_multiplier is not None else None,
                    device=device,
                    features_dtype=dtype,
                )


    print(f"[INFO] common keys = {len(keys)} (reconcile={strat})")

    for k in keys:
        p_act = act[k]
        p_wgt = wgt[k]

        if not is_valid_permutation(p_act):
            raise ValueError(f"[{k}] activation permutation is not a valid permutation.")
        if not is_valid_permutation(p_wgt):
            raise ValueError(f"[{k}] weight permutation is not a valid permutation.")
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

        if do_state and ps is not None:
            entry["dW_weight_alignment_error"] = {
                "under_wgt_perm": dW_for_perm_key(perm_key=k, ps=ps, state_a=state_a, state_b_perm=b_wgt_perm),
                "under_act_perm": dW_for_perm_key(perm_key=k, ps=ps, state_a=state_a, state_b_perm=b_act_perm),
            }

        if do_cka:
            if k not in feats_a or k not in feats_b:
                entry["repr_alignment"] = {"skipped": True, "reason": "missing features for this key"}
            else:
                xa = activation_to_2d(feats_a[k], unit_dim=args.unit_dim)
                xb = activation_to_2d(feats_b[k], unit_dim=args.unit_dim)

                raw_m = repr_alignment_metrics(xa, xb, p=None, cka_fn=cka_fn)
                wgt_m = repr_alignment_metrics(xa, xb, p=p_wgt, cka_fn=cka_fn)
                act_m = repr_alignment_metrics(xa, xb, p=p_act, cka_fn=cka_fn)

                out: Dict[str, Any] = {
                    "raw": raw_m,
                    "under_wgt_perm": wgt_m,
                    "under_act_perm": act_m,
                }

                # deltas (only if not skipped)
                if not raw_m.get("skipped", False) and not wgt_m.get("skipped", False):
                    out["delta_wgt_minus_raw"] = {
                        "dH_raw_frob": wgt_m["dH"]["raw_frob"] - raw_m["dH"]["raw_frob"],
                        "dH_rel_to_A": wgt_m["dH"]["rel_to_A"] - raw_m["dH"]["rel_to_A"],
                        "mean_unit_cos": wgt_m["mean_unit_cos"] - raw_m["mean_unit_cos"],
                        **({"cka": wgt_m["cka"] - raw_m["cka"]} if ("cka" in wgt_m and "cka" in raw_m) else {}),
                    }
                if not raw_m.get("skipped", False) and not act_m.get("skipped", False):
                    out["delta_act_minus_raw"] = {
                        "dH_raw_frob": act_m["dH"]["raw_frob"] - raw_m["dH"]["raw_frob"],
                        "dH_rel_to_A": act_m["dH"]["rel_to_A"] - raw_m["dH"]["rel_to_A"],
                        "mean_unit_cos": act_m["mean_unit_cos"] - raw_m["mean_unit_cos"],
                        **({"cka": act_m["cka"] - raw_m["cka"]} if ("cka" in act_m and "cka" in raw_m) else {}),
                    }
                if not wgt_m.get("skipped", False) and not act_m.get("skipped", False):
                    out["delta_act_minus_wgt"] = {
                        "dH_raw_frob": act_m["dH"]["raw_frob"] - wgt_m["dH"]["raw_frob"],
                        "dH_rel_to_A": act_m["dH"]["rel_to_A"] - wgt_m["dH"]["rel_to_A"],
                        "mean_unit_cos": act_m["mean_unit_cos"] - wgt_m["mean_unit_cos"],
                        **({"cka": act_m["cka"] - wgt_m["cka"]} if ("cka" in act_m and "cka" in wgt_m) else {}),
                    }

                # Optional: compare against features extracted from *permuted-weight* models
                if args.compare_permuted_weight_model and (k in feats_b_from_wgt_model) and (k in feats_b_from_act_model):
                    xb_wgt_model = activation_to_2d(feats_b_from_wgt_model[k], unit_dim=args.unit_dim)
                    xb_act_model = activation_to_2d(feats_b_from_act_model[k], unit_dim=args.unit_dim)
                    out["from_permuted_weight_model"] = {
                        "wgt": repr_alignment_metrics(xa, xb_wgt_model, p=None, cka_fn=cka_fn),
                        "act": repr_alignment_metrics(xa, xb_act_model, p=None, cka_fn=cka_fn),
                    }

                entry["repr_alignment"] = out

        if args.compute_llfc and (k in llfc_per_key):
            entry["llfc"] = llfc_per_key[k]

        report["per_key"][k] = entry

        print(
            f"[{k}] n={entry['n']}  agree={agreement:.4f}  fixed(Q)={fixed:.4f}  "
            f"num_cycles={cs['num_cycles']}  max_cycle={cs['max_cycle']}  cayley={cs['cayley_distance']}"
        )

    if args.out_json:
        os.makedirs(os.path.dirname(os.path.abspath(args.out_json)), exist_ok=True)
        with open(args.out_json, "w") as f:
            json.dump(report, f, indent=2)
        print(f"[INFO] wrote {args.out_json}")

    avg_agree = float(np.mean([report["per_key"][k]["hamming_agreement"] for k in keys]))
    avg_fixed = float(np.mean([report["per_key"][k]["fixed_point_fraction_of_Q"] for k in keys]))
    print(f"[SUMMARY] mean agreement={avg_agree:.4f} | mean fixed(Q)={avg_fixed:.4f}")


if __name__ == "__main__":
    main()