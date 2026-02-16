#!/usr/bin/env python3
"""
compare_permutations.py

Compare permutations found by:
  - weight matching (expects .pkl saved as {perm_name: np.ndarray})
  - activation matching (accepts .json / .pt / .pkl)

Metrics per perm key (layer / perm group):
  (A) Combinatorial agreement:
      - Hamming agreement between P_wgt and P_act
      - Fixed-point fraction of Q = inv(P_wgt) o P_act
      - Cycle structure of Q
      - (optional) Cayley distance = n - #cycles

  (B) Geometric / representation agreement (optional):
      - CKA(A_features, permute(B_features, P_wgt))
      - CKA(A_features, permute(B_features, P_act))

Notes on permutation convention:
  p[i] = j means "A unit i matches B unit j" (A -> B assignment).


HOW TO USE IT:
export PYTHONPATH="$(pwd)"
for x in 1 2 8 16 ; do
    python compare_permutations.py \
    --act-perm activation_out/CIFAR100/activation_stitching_out_cifar100_resnet20_${x}/permutations.json \
    --wgt-perm weight_matching_out/resnet20_${x}/CIFAR100/disjoint/permutation_seed0.pkl \
    --unit-dim 1 \
    --dataset CIFAR100 \
    --state-a runs_resnet20_${x}/CIFAR100/disjoint/seed_0/subset_A/resnet20_CIFAR100_seed0_subsetA_best.pth \
    --state-b runs_resnet20_${x}/CIFAR100/disjoint/seed_0/subset_B/resnet20_CIFAR100_seed0_subsetB_best.pth\
    --shortcut-option C \
    --out-json out/CIFAR100/disjoint/report_${x}.json
    done 
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

from linear_mode_connectivity.weight_matching_torch import (
    resnet20_layernorm_permutation_spec,
    apply_permutation,
)

import torch.nn as nn
from torch.utils.data import DataLoader, Subset

import architectures
import train_resnet

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
    # Example (your repo has AlignmentMetrics.cka)
    try:
        from metrics_platonic import AlignmentMetrics  # type: ignore
        return AlignmentMetrics.cka
    except Exception:
        pass

    # Fallback: if you have a standalone cka.py on PYTHONPATH with `def cka(x, y): ...`
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
            # prefer pickle if available, then pt, then json
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
    """
    Returns dict[str, LongTensor].
    Keys are stringified, but we also attempt to reconcile common key mismatches later.
    """
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

# Other helpers :

def _frob(x: torch.Tensor) -> float:
    return float(torch.linalg.norm(x).item())

def dW_for_perm_key(
    *,
    perm_key: str,
    ps,
    state_a: Dict[str, torch.Tensor],
    state_b_perm: Dict[str, torch.Tensor],
) -> Dict[str, float]:
    # all params whose axes mention perm_key
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

# helpers for shortcut C in resnet20

_RE_STRIDE2 = re.compile(r"^layer([23])\.0\.(conv1|shortcut\.0)\.weight$")

def infer_conv_stride(name: str) -> int:
    return 2 if _RE_STRIDE2.match(name) else 1

def infer_conv_padding(weight: torch.Tensor) -> int:
    # works for 3x3 (pad=1), 1x1 (pad=0), etc.
    kh = int(weight.shape[-2])
    kw = int(weight.shape[-1])
    return kh // 2  # assumes odd kernels used here

_P_INNER = re.compile(r"^P_layer(\d+)_(\d+)_inner$")
def resnet_preprocess_for_perm(perm_name: str, x: torch.Tensor) -> torch.Tensor:
    # matches your activation-stitching mapping: ReLU applied for P_bg0 and inner perms :contentReference[oaicite:9]{index=9}
    if perm_name == "P_bg0" or _P_INNER.match(perm_name):
        return F.relu(x)
    return x

def rcom_for_output_perm(
    *,
    out_perm: str,
    ps,
    state_a: Dict[str, torch.Tensor],
    state_b_perm: Dict[str, torch.Tensor],
    feats_a: Dict[str, torch.Tensor],
    feats_b: Dict[str, torch.Tensor],
    perm_dict: Dict[str, torch.Tensor],  # the SAME dict used to build state_b_perm
) -> Dict[str, float]:
    # sum over conv weights whose output-axis perm == out_perm
    ssq = 0.0
    count = 0

    for wname, axes_perms in ps.axes_to_perm.items():
        if wname not in state_a or wname not in state_b_perm:
            continue
        Wa = state_a[wname]
        if Wa.ndim != 4 or not wname.endswith(".weight"):
            continue  # conv weights only

        p_out, p_in, _, _ = axes_perms  # conv spec is (p_out, p_in, None, None) :contentReference[oaicite:10]{index=10}
        if p_out != out_perm or p_in is None:
            continue

        # Need input activations keyed by the *input* permutation name
        if p_in not in feats_a or p_in not in feats_b or p_in not in perm_dict:
            continue

        Ha = feats_a[p_in].float()
        Hb = feats_b[p_in].float()

        # Optional preprocess to match your permutation computation convention
        Ha = resnet_preprocess_for_perm(p_in, Ha)
        Hb = resnet_preprocess_for_perm(p_in, Hb)

        # Align B channels into A indexing
        Hb_aligned = Hb.index_select(1, perm_dict[p_in])

        dH = Ha - Hb_aligned  # [N,C,H,W]
        dW = state_a[wname].float() - state_b_perm[wname].float()  # [Cout,Cin,kh,kw]

        stride = infer_conv_stride(wname)
        pad = infer_conv_padding(dW)

        out = F.conv2d(dH, dW, bias=None, stride=stride, padding=pad)
        ssq += float((out * out).sum().item())
        count += 1

    raw = float(ssq ** 0.5)
    return {"raw_frob": raw, "num_convs": count}


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
    """
    p maps A-index -> B-index.
    Returns inv mapping B-index -> A-index.
    """
    n = int(p_a_to_b.numel())
    inv = torch.empty(n, dtype=torch.long)
    inv[p_a_to_b] = torch.arange(n, dtype=torch.long)
    return inv


def compose_q(p_wgt: torch.Tensor, p_act: torch.Tensor) -> torch.Tensor:
    """
    Q = inv(P_wgt) o P_act
    where P_wgt, P_act: A -> B (same convention).
    Q: A -> A
    """
    if p_wgt.numel() != p_act.numel():
        raise ValueError(f"Size mismatch: |P_wgt|={p_wgt.numel()} vs |P_act|={p_act.numel()}")
    inv_w = invert_perm(p_wgt)
    return inv_w[p_act]


def cycles_of_perm(q: torch.Tensor) -> List[List[int]]:
    """
    q is a permutation over {0..n-1}. Returns list of cycles (as lists of ints).
    """
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
        "cayley_distance": n - len(cycs),  # minimal #transpositions to reach identity
    }


# -------------------------
# Key reconciliation (MLP: "1,2,3" vs "P1,P2,P3")
# -------------------------
_P_INT = re.compile(r"^P(\d+)$")


def reconcile_keys(
    act: Dict[str, torch.Tensor],
    wgt: Dict[str, torch.Tensor],
) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor], str]:
    """
    Returns possibly-remapped (act, wgt, strategy).
    """
    if set(act).intersection(set(wgt)):
        return act, wgt, "direct"

    # Try mapping activation keys "1" -> "P1" (common for MLP activation matching output)
    act_digits = all(k.isdigit() for k in act.keys())
    wgt_pints = all(_P_INT.match(k) for k in wgt.keys())
    if act_digits and wgt_pints:
        act2 = {f"P{k}": v for k, v in act.items()}
        if set(act2).intersection(set(wgt)):
            return act2, wgt, "act_digit_to_P"

    # Try mapping weight keys "P1" -> "1"
    if wgt_pints:
        wgt2 = {_P_INT.match(k).group(1): v for k, v in wgt.items()}  # type: ignore[union-attr]
        if set(act).intersection(set(wgt2)):
            return act, wgt2, "wgt_P_to_digit"

    return act, wgt, "none"


# -------------------------
# Optional: CKA feature handling
# -------------------------
def activation_to_2d(out: torch.Tensor, unit_dim: int = 1) -> torch.Tensor:
    """
    Same idea as activation_permutation_stitching.activation_to_2d :contentReference[oaicite:3]{index=3}
    """
    if out.ndim < 2:
        raise ValueError(f"Expected activation with ndim>=2, got {tuple(out.shape)}")
    if unit_dim < 0:
        unit_dim = out.ndim + unit_dim
    x = torch.movedim(out, unit_dim, -1)
    return x.reshape(-1, x.shape[-1])


def load_features(path: str) -> Dict[str, torch.Tensor]:
    """
    Expects a dict-like mapping layer_key -> activation tensor.
    Supports .pt/.pkl/.json but typical is torch.save(dict).
    """
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

# -------------------------
# Optional: state-dict + LLFC-style distance metrics
# -------------------------
_PREFIXES = ("module.", "model.", "net.")
_FC_W_RE = re.compile(r"^fc(\d+)\.weight$")

def load_state_dict_any(path: str) -> Dict[str, torch.Tensor]:
    obj = _load_any(path)
    if isinstance(obj, dict) and "state_dict" in obj and isinstance(obj["state_dict"], dict):
        obj = obj["state_dict"]
    if not isinstance(obj, dict):
        raise ValueError(f"Expected dict-like state_dict in {path}, got {type(obj)}")

    # strip common prefixes
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

def parse_layer_index(key: str) -> Optional[int]:
    # supports "1" or "P1" (your reconcile_keys already converts between them)
    if key.isdigit():
        return int(key)
    m = _P_INT.match(key)
    if m:
        return int(m.group(1))
    # optional extra patterns (harmless if unused)
    m = re.match(r"^relu(\d+)$", key)
    if m:
        return int(m.group(1))
    m = re.match(r"^fc(\d+)$", key)
    if m:
        return int(m.group(1))
    return None

def frob(x: torch.Tensor) -> float:
    return float(torch.linalg.norm(x).item())

def activation_alignment_error(xa2: torch.Tensor, xb2: torch.Tensor, p: torch.Tensor) -> Dict[str, float]:
    # xa2, xb2: [M, d], p: [d] with p[i]=j (A->B). Align B into A-order via xb2[:, p].
    diff = xa2 - xb2[:, p]
    raw = frob(diff)
    rel = raw / (frob(xa2) + 1e-12)
    return {"raw_frob": raw, "rel_to_A": rel}

def weight_alignment_error(
    Wa: torch.Tensor,
    Wb: torch.Tensor,
    p_out: Optional[torch.Tensor],
    p_in: Optional[torch.Tensor],
) -> Dict[str, float]:
    # Wa,Wb: [d_out, d_in]. p_out permutes rows, p_in permutes cols (both A->B, so select by p).
    Wb_aligned = Wb
    if p_out is not None:
        Wb_aligned = Wb_aligned.index_select(0, p_out.to(dtype=torch.long))
    if p_in is not None:
        Wb_aligned = Wb_aligned.index_select(1, p_in.to(dtype=torch.long))
    diff = Wa - Wb_aligned
    raw = frob(diff)
    rel = raw / (frob(Wa) + 1e-12)
    return {"raw_frob": raw, "rel_to_A": rel}

def commutativity_residual(
    Wa: torch.Tensor,
    Wb: torch.Tensor,
    Ha_prev_2d: torch.Tensor,
    Hb_prev_2d: torch.Tensor,
    p_out: Optional[torch.Tensor],
    p_in: Optional[torch.Tensor],
    p_prev: Optional[torch.Tensor],
) -> Dict[str, float]:
    # Build aligned Wb (rows by p_out, cols by p_in)
    Wb_aligned = Wb
    if p_out is not None:
        Wb_aligned = Wb_aligned.index_select(0, p_out.to(dtype=torch.long))
    if p_in is not None:
        Wb_aligned = Wb_aligned.index_select(1, p_in.to(dtype=torch.long))
    dW = Wa - Wb_aligned  # [d_out, d_in]

    # Align prev activations into A-order with p_prev (units are columns)
    Hb_aligned = Hb_prev_2d if p_prev is None else Hb_prev_2d[:, p_prev]
    dH = Ha_prev_2d - Hb_aligned  # [M, d_in]

    # Equivalent to || (dW)(dH^T) ||_F, but we compute (dH)(dW^T): [M,d_out]
    prod = dH.matmul(dW.t())
    raw = frob(prod)
    norm = (frob(dW) * frob(dH)) + 1e-12
    return {"raw_frob": raw, "normalized": raw / norm}

# helpers for computing features from state dicts # -------------------------
# ResNet20-LN: infer width/shortcut + map perm-name -> hook layer (copied from resnet20_activation_stitching.py)
# -------------------------

# def infer_width_multiplier_from_state(state: Dict[str, torch.Tensor]) -> int:
#     # conv1 out_channels = 16 * width_multiplier  (same heuristic as stitching script)
#     w = int(state["conv1.weight"].shape[0] // 16)
#     if w <= 0:
#         raise ValueError("Could not infer width_multiplier from conv1.weight")
#     return w

# def infer_shortcut_option_from_state(state: Dict[str, torch.Tensor]) -> str:
#     # If shortcut conv exists, you’re in B/C-style shortcut (your script returns "C" in that case)
#     return "C" if any(k.endswith("shortcut.0.weight") for k in state.keys()) else "A"

_P_INNER_HOOK = re.compile(r"^P_layer(\d+)_(\d+)_inner$")

# def perm_name_to_hook(perm_name: str) -> Tuple[str, Optional[Any], int]:
#     """
#     Returns: (layer_name_to_hook, preprocess_fn, unit_dim)
#     Matches resnet20_activation_stitching.perm_name_to_hook() :contentReference[oaicite:3]{index=3}
#     """
#     if perm_name == "P_bg0":
#         return "n1", torch.relu, 1
#     if perm_name == "P_bg1":
#         return "layer2.0", None, 1
#     if perm_name == "P_bg2":
#         return "layer3.0", None, 1
#     m = _P_INNER_HOOK.match(perm_name)
#     if m:
#         layer = int(m.group(1))
#         block = int(m.group(2))
#         return f"layer{layer}.{block}.n1", torch.relu, 1
#     raise KeyError(f"Don't know how to map perm name to hook layer: {perm_name}")

def _get_submodule_by_name(model: nn.Module, name: str) -> nn.Module:
    """
    Supports dotted paths like:
      - "n1"
      - "layer2.0"
      - "layer3.1.n1"
    """
    cur: nn.Module = model
    for part in name.split("."):
        if part.isdigit():
            idx = int(part)
            # Sequential / ModuleList support
            if isinstance(cur, (nn.Sequential, nn.ModuleList)):
                cur = cur[idx]
            else:
                cur = getattr(cur, part)  # fallback
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
    ps,  # permutation spec
    shortcut_option: str,
    width_multiplier: Optional[int],
    device: torch.device,
    features_dtype: torch.dtype,
) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
    """
    Builds model A/B, loads weights, runs forward passes over the chosen split,
    and records activations for every perm-key in ps.perm_to_axes.

    Supports:
      - ResNet20 (CIFAR-style) via architectures.build_model("resnet20", ...)
      - MLP (fc1..fc4) via architectures.MLP / architectures.build_model("mlp", ...)

    For MLP, since ReLUs are functional (no nn.ReLU modules), we hook the Linear
    layers (fc1/fc2/fc3) and apply ReLU inside the hook so features correspond
    to the post-activation hidden units.
    """
    import math
    import re
    import datasets as ds_utils

    # -------------------------
    # Dataset / loader
    # -------------------------
    dataset_u = dataset.strip().upper()
    if dataset_u not in ds_utils.DATASET_STATS:
        raise ValueError(
            f"Unsupported dataset: {dataset} (supported: {sorted(ds_utils.DATASET_STATS)})"
        )

    # train_full has optional augmentation; eval_full + test have no augmentation
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
        raise ValueError(
            f"Unsupported split for features: {split} (use 'test' or 'train_eval')"
        )

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

    # -------------------------
    # Architecture detection
    # -------------------------
    has_conv = ("conv1.weight" in state_a) or any(k.startswith("layer1.") for k in state_a.keys())
    has_fc1  = ("fc1.weight" in state_a) or any(k.startswith("fc1.") for k in state_a.keys())

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

    # -------------------------
    # Build + load models
    # -------------------------
    if arch == "mlp":
        # Infer contiguous fc1..fcN from checkpoint (like your WM script does)
        pat = re.compile(r"^fc(\d+)\.weight$")
        layer_ids = sorted({int(pat.match(k).group(1)) for k in state_a.keys() if pat.match(k)})
        if not layer_ids:
            raise KeyError("MLP detected, but no fc{n}.weight keys found in state_dict.")
        n_layers = max(layer_ids)
        expected = list(range(1, n_layers + 1))
        if layer_ids != expected:
            raise ValueError(f"Expected contiguous fc layers {expected}, found {layer_ids}")

        # Repo MLP is fc1..fc4 (3 hidden + classifier). Enforce to avoid silent mismatch.
        if n_layers != 4:
            raise ValueError(
                f"This feature-extractor expects the repo MLP with fc1..fc4. "
                f"Found fc1..fc{n_layers} in checkpoint."
            )

        hidden = int(state_a["fc1.weight"].shape[0])
        in_dim = int(state_a["fc1.weight"].shape[1])

        # Prefer dataset stats for input shape, but fall back if needed
        stats = ds_utils.DATASET_STATS[dataset_u]
        c = int(stats["in_channels"])
        h, w = map(int, stats["image_size"])
        if c * h * w != in_dim:
            # Fallback: infer square spatial dims if possible
            if in_dim % c != 0:
                raise ValueError(
                    f"Cannot infer input_shape: fc1.weight has in_dim={in_dim}, "
                    f"but dataset in_channels={c} does not divide it."
                )
            side_f = math.sqrt(in_dim / c)
            side = int(round(side_f))
            if side * side * c != in_dim:
                raise ValueError(
                    f"Cannot infer input_shape: fc1.weight in_dim={in_dim} not compatible "
                    f"with square H=W given in_channels={c}."
                )
            h = w = side

        input_shape = (c, h, w)
        num_classes = int(state_a[f"fc{n_layers}.weight"].shape[0])

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

        keys = set(model_a.state_dict().keys())
        state_a_f = {k: v for k, v in state_a.items() if k in keys}
        state_b_f = {k: v for k, v in state_b.items() if k in keys}
        model_a.load_state_dict(state_a_f, strict=True)
        model_b.load_state_dict(state_b_f, strict=True)

        # Map perm name -> (module_name, preprocess_fn, unit_dim)
        # Perm keys are typically "P1","P2","P3" (hidden layers).
        _P_OR_DIGIT = re.compile(r"^(?:P)?(\d+)$")

        def _hook_info(pname: str) -> Tuple[str, Optional[Any], int]:
            m = _P_OR_DIGIT.match(pname)
            if not m:
                raise KeyError(f"MLP: don't know how to map perm name to hook layer: {pname}")
            i = int(m.group(1))
            # hidden layers correspond to fc1..fc(n_layers-1)
            if i < 1 or i >= n_layers:
                raise KeyError(f"MLP: perm {pname} implies fc{i}, but only hidden perms 1..{n_layers-1} exist.")
            return f"fc{i}", F.relu, 1

    else:
        # ResNet20
        # Prefer inferring num_classes from checkpoint if present; otherwise dataset stats.
        if "linear.weight" in state_a:
            num_classes = int(state_a["linear.weight"].shape[0])
        else:
            num_classes = int(ds_utils.DATASET_STATS[dataset_u]["num_classes"])

        if width_multiplier is None:
            width_multiplier = infer_width_multiplier_from_state(state_a)

        # Allow caller to pass shortcut_option; if you want full auto, uncomment:
        # shortcut_option = infer_shortcut_option_from_state(state_a)

        model_a = architectures.build_model(
            "resnet20",
            num_classes=num_classes,
            norm="flax_ln",
            width_multiplier=int(width_multiplier),
            shortcut_option=str(shortcut_option),
        ).to(device).eval()

        model_b = architectures.build_model(
            "resnet20",
            num_classes=num_classes,
            norm="flax_ln",
            width_multiplier=int(width_multiplier),
            shortcut_option=str(shortcut_option),
        ).to(device).eval()

        keys = set(model_a.state_dict().keys())
        state_a_f = {k: v for k, v in state_a.items() if k in keys}
        state_b_f = {k: v for k, v in state_b.items() if k in keys}
        model_a.load_state_dict(state_a_f, strict=True)
        model_b.load_state_dict(state_b_f, strict=True)

        def _hook_info(pname: str) -> Tuple[str, Optional[Any], int]:
            return perm_name_to_hook(pname)

    # -------------------------
    # Forward-hook extraction
    # -------------------------
    def _extract(model: nn.Module) -> Dict[str, torch.Tensor]:
        store: Dict[str, List[torch.Tensor]] = {p: [] for p in perm_names}
        hooks = []

        for p in perm_names:
            layer_name, preprocess, _unit_dim = _hook_info(p)
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
    feats_b = _extract(model_b)
    return feats_a, feats_b

# -------------------------
# Main
# -------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--act-perm", type=str, required=True,
                    help="Path to activation-matching permutations (.json/.pt/.pkl).")
    ap.add_argument("--wgt-perm", type=str, required=True,
                    help="Path to weight-matching permutations (.pkl).")
    ap.add_argument("--out-json", type=str, default=None,
                    help="If set, save a JSON report to this path.")

    # Optional CKA inputs
    ap.add_argument("--features-a", type=str, default=None,
                    help="Optional: path to saved A features dict[layer_key -> tensor].")
    ap.add_argument("--features-b", type=str, default=None,
                    help="Optional: path to saved B features dict[layer_key -> tensor].")
    ap.add_argument("--unit-dim", type=int, default=1,
                    help="Feature dimension corresponding to units/channels (default 1).")
    ap.add_argument("--state-a", type=str, default=None,
                    help="Optional: checkpoint/state_dict for model A (.pth/.pt/.pkl). Needed for dW and r_com.")
    ap.add_argument("--state-b", type=str, default=None,
                    help="Optional: checkpoint/state_dict for model B (.pth/.pt/.pkl). Needed for dW and r_com.")
    ap.add_argument("--shortcut-option", type=str, default="C", choices=["A", "B", "C"],
                    help="ResNet shortcut option. (Spec auto-detects from state_dict, but keep for safety.)")
    # If provided, compute features internally (so you can omit --features-a/--features-b)
    ap.add_argument("--dataset", type=str, default=None, choices=["CIFAR10", "CIFAR100"],
                    help="If set (and --state-a/--state-b are set), compute features on the fly.")
    ap.add_argument("--data-root", type=str, default="./data",
                    help="Root for torchvision CIFAR downloads (same as train_resnet).")
    ap.add_argument("--features-split", type=str, default="test", choices=["test", "train_eval"],
                    help="Which split to use for feature extraction.")
    ap.add_argument("--features-samples", type=int, default=512,
                    help="How many samples to use for features (<=0 means full split).")
    ap.add_argument("--features-batch-size", type=int, default=256)
    ap.add_argument("--features-num-workers", type=int, default=0)
    ap.add_argument("--features-dtype", type=str, default="float16", choices=["float16", "float32"])
    ap.add_argument("--width-multiplier", type=int, default=None,
                    help="Override inferred width multiplier (otherwise inferred from conv1.weight).")

    args = ap.parse_args()

    act = load_permutations(args.act_perm)
    wgt = load_permutations(args.wgt_perm)
    # if it is inverted 
    # wgt = {k: invert_perm(v) for k, v in wgt.items()}
    # act = {k: invert_perm(v) for k, v in act.items()}

    act, wgt, strat = reconcile_keys(act, wgt)

    keys = sorted(set(act).intersection(set(wgt)))
    if not keys:
        raise RuntimeError(
            "No overlapping permutation keys between activation and weight matching.\n"
            f"Reconciliation strategy tried: {strat}\n"
            f"Activation keys (sample): {list(act.keys())[:10]}\n"
            f"Weight keys (sample): {list(wgt.keys())[:10]}"
        )

    report: Dict[str, Any] = {
        "act_perm_path": args.act_perm,
        "wgt_perm_path": args.wgt_perm,
        "key_reconciliation": strat,
        "num_common_keys": len(keys),
        "per_key": {},
    }

    # ---- features / CKA source ----
    feats_a: Dict[str, torch.Tensor] = {}
    feats_b: Dict[str, torch.Tensor] = {}

    have_feat_files = (args.features_a is not None) and (args.features_b is not None)
    do_cka = False
    cka_fn = None

    if have_feat_files:
        do_cka = True
        cka_fn = _load_cka_fn()
        feats_a = load_features(args.features_a)
        feats_b = load_features(args.features_b)

    do_state = (args.state_a is not None) and (args.state_b is not None)
    state_a = load_state_dict_any(args.state_a) if do_state else {}
    state_b = load_state_dict_any(args.state_b) if do_state else {}

    ps = None
    b_wgt_perm = {}
    b_act_perm = {}
    if do_state:
        # Build spec (handles LN param layout and shortcut params; detects presence of shortcut convs) :contentReference[oaicite:6]{index=6}
        ps = resnet20_layernorm_permutation_spec(
            shortcut_option=str(args.shortcut_option),
            state_dict=state_a,
        )

        # Permute ALL B params into A-indexing under each permutation-set
        b_wgt_perm = apply_permutation(ps, wgt, state_b)
        b_act_perm = apply_permutation(ps, act, state_b)
                # If user asked for dataset-based features, compute them now (no need for --features-a/--features-b)
        if (args.dataset is not None) and (not have_feat_files):
            do_cka = True
            cka_fn = _load_cka_fn()

            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            dtype = torch.float16 if args.features_dtype == "float16" else torch.float32
            n_samp = int(args.features_samples)
            n_samp = n_samp if n_samp > 0 else 0  # 0 => full

            # shortcut option: use CLI if you want, but you can also infer from the checkpoint
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

    # Map each perm-key to a layer index (so we can fetch prev layer)
    key_to_idx = {k: parse_layer_index(k) for k in set(act.keys()).union(set(wgt.keys()))}
    idx_to_key: Dict[int, str] = {}
    for kk, ii in key_to_idx.items():
        if ii is not None and ii not in idx_to_key:
            idx_to_key[ii] = kk


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

        # agreement / fixed points
        agreement = float((p_act == p_wgt).float().mean().item())

        # Q = inv(P_wgt) o P_act
        q = compose_q(p_wgt, p_act)
        fixed = float((q == torch.arange(q.numel())).float().mean().item())

        # cycle structure
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

        # Optional CKA: compare A vs permuted-B representations under each permutation
        if do_cka:
            if k not in feats_a or k not in feats_b:
                entry["cka"] = {"skipped": True, "reason": "missing features for this key"}
            else:
                xa = activation_to_2d(feats_a[k], unit_dim=args.unit_dim)
                xb = activation_to_2d(feats_b[k], unit_dim=args.unit_dim)

                d = int(p_act.numel())
                if xa.shape[1] != d or xb.shape[1] != d:
                    entry["cka"] = {
                        "skipped": True,
                        "reason": f"feature dim mismatch: xa={tuple(xa.shape)}, xb={tuple(xb.shape)}, perm_dim={d}",
                    }
                else:
                    xb_wgt = xb[:, p_wgt]
                    xb_act = xb[:, p_act]
                                        # d_H under each permutation (WM vs AM)
                    dH_wgt = {
                        "raw_frob": float(torch.linalg.norm(xa - xb_wgt).item()),
                        "rel_to_A": float(torch.linalg.norm(xa - xb_wgt).item()) / (float(torch.linalg.norm(xa).item()) + 1e-12),
                    }
                    dH_act = {
                        "raw_frob": float(torch.linalg.norm(xa - xb_act).item()),
                        "rel_to_A": float(torch.linalg.norm(xa - xb_act).item()) / (float(torch.linalg.norm(xa).item()) + 1e-12),
                    }
                    entry["dH_activation_alignment_error"] = {
                        "under_wgt_perm": dH_wgt,
                        "under_act_perm": dH_act,
                        "delta_act_minus_wgt_raw": dH_act["raw_frob"] - dH_wgt["raw_frob"],
                        "delta_act_minus_wgt_rel": dH_act["rel_to_A"] - dH_wgt["rel_to_A"],
                    }
                    def _to_float(v):
                        return float(v.item() if isinstance(v, torch.Tensor) else v)

                    cka_wgt = _to_float(cka_fn(xa, xb_wgt))
                    cka_act = _to_float(cka_fn(xa, xb_act))
                    entry["cka"] = {
                        "cka_A_vs_Bperm_wgt": cka_wgt,
                        "cka_A_vs_Bperm_act": cka_act,
                        "delta_act_minus_wgt": cka_act - cka_wgt,
                    }

        report["per_key"][k] = entry

        print(
            f"[{k}] n={entry['n']}  agree={agreement:.4f}  fixed(Q)={fixed:.4f}  "
            f"num_cycles={cs['num_cycles']}  max_cycle={cs['max_cycle']}  cayley={cs['cayley_distance']}"
        )

        if do_state and ps is not None and do_cka:
            # r_com under each permutation-set (WM vs AM), grouping by output perm key k
            entry["r_com_commutativity_residual_conv"] = {
                "under_wgt_perm": rcom_for_output_perm(
                    out_perm=k, ps=ps,
                    state_a=state_a, state_b_perm=b_wgt_perm,
                    feats_a=feats_a, feats_b=feats_b,
                    perm_dict=wgt,
                ),
                "under_act_perm": rcom_for_output_perm(
                    out_perm=k, ps=ps,
                    state_a=state_a, state_b_perm=b_act_perm,
                    feats_a=feats_a, feats_b=feats_b,
                    perm_dict=act,
                ),
            }

    if args.out_json:
        os.makedirs(os.path.dirname(os.path.abspath(args.out_json)), exist_ok=True)
        with open(args.out_json, "w") as f:
            json.dump(report, f, indent=2)
        print(f"[INFO] wrote {args.out_json}")

    # Also print a quick overall summary
    avg_agree = float(np.mean([report["per_key"][k]["hamming_agreement"] for k in keys]))
    avg_fixed = float(np.mean([report["per_key"][k]["fixed_point_fraction_of_Q"] for k in keys]))
    print(f"[SUMMARY] mean agreement={avg_agree:.4f} | mean fixed(Q)={avg_fixed:.4f}")


if __name__ == "__main__":
    main()