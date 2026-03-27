#!/usr/bin/env python3
"""
activation_permutation_stitching.py

Activation-based (representation-based) permutation finding + stitching utilities
for MLPs like architectures.MLP in this repo.

Instead of *weight matching*, we derive permutations by aligning hidden units using
their activations on a shared dataset (typically the training set, as in model stitching).

Core idea (per hidden layer ℓ):
  - Run both models A and B on the SAME input examples.
  - Collect activations h_A^ℓ ∈ R^{N×d}, h_B^ℓ ∈ R^{N×d}.
  - Build a similarity matrix S_{ij} = corr(h_A[:,i], h_B[:,j]) (or cosine).
  - Use Hungarian (linear_sum_assignment) to find a permutation π maximizing Σ_i S_{i,π(i)}.
  - Apply π to reparameterize B (function-preserving) and/or to adapt B's downstream layers
    to consume A's representation order at a stitch boundary.

This module avoids importing your repo's weight_matching implementation on purpose.
We only reuse the *concept* (Hungarian assignment), but driven by activations.

Dependencies:
  - torch
  - scipy (linear_sum_assignment)
"""

from __future__ import annotations

import pickle
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Callable, Any

import torch
import torch.nn as nn

try:
    from scipy.optimize import linear_sum_assignment
except Exception as e:  # pragma: no cover
    raise ImportError("scipy is required for Hungarian matching (linear_sum_assignment).") from e


TensorDict = Dict[str, torch.Tensor]

# functions added for compatibiltiy with ResNEt

def _as_tensor(out: Any) -> torch.Tensor:
    if isinstance(out, torch.Tensor):
        return out
    if isinstance(out, (tuple, list)) and len(out) > 0 and isinstance(out[0], torch.Tensor):
        return out[0]
    raise TypeError(f"Unsupported hook output type: {type(out)}") 


def activation_to_2d(out: torch.Tensor, unit_dim: int = 1) -> torch.Tensor:
    """
    Convert an activation tensor to shape [M, d] where d indexes the permutable 'units'
    (e.g., hidden units for MLP, channels for CNN), and M flattens everything else.

    Examples:
      - MLP hidden: [N, d] -> [N, d]
      - Conv feature map: [N, C, H, W] with unit_dim=1 -> [N*H*W, C]
    """
    if out.ndim < 2:
        raise ValueError(f"Expected activation with ndim>=2, got {out.shape}")
    if unit_dim < 0:
        unit_dim = out.ndim + unit_dim
    if not (0 <= unit_dim < out.ndim):
        raise ValueError(f"unit_dim out of range: {unit_dim} for out.ndim={out.ndim}")
    x = torch.movedim(out, unit_dim, -1)
    return x.reshape(-1, x.shape[-1])

# -------------------------
# IO helpers
# -------------------------
def load_ckpt_state_dict(path: str) -> TensorDict:
    """
    Loads checkpoints saved either as:
      - {"state_dict": ...}  (train_loop.py style), or
      - raw state_dict dict[str, Tensor].
    """
    obj = torch.load(path, map_location="cpu")
    if isinstance(obj, dict) and "state_dict" in obj:
        return obj["state_dict"]
    if isinstance(obj, dict):
        return obj
    raise ValueError(f"Unrecognized checkpoint format at: {path}")


def normalize_state_dict_keys(state: TensorDict) -> TensorDict:
    """
    Strip common wrappers (DataParallel/Lightning/custom).
    """
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


# -------------------------
# MLP discovery helpers
# -------------------------
_FC_WEIGHT_RE = re.compile(r"^fc(\d+)\.weight$")


def infer_fc_layer_numbers_from_state(state: TensorDict) -> List[int]:
    """
    Looks for keys like fc1.weight, fc2.weight, ... and returns sorted indices.
    """
    layers: List[int] = []
    for k in state.keys():
        m = _FC_WEIGHT_RE.match(k)
        if m:
            layers.append(int(m.group(1)))
    layers = sorted(set(layers))
    if not layers:
        raise KeyError("No fc{n}.weight keys found in state_dict.")
    return layers


def infer_relu_module_names(model: nn.Module, n_fc_layers: Optional[int] = None) -> List[str]:
    """
    For architectures.MLP, hidden activations are produced by relu1..relu{n-1}.
    We detect these and return them in order.

    If relu{i} doesn't exist, we fall back to hooking fc{i} (pre-activation).
    """
    if n_fc_layers is None:
        n_fc_layers = 0
        for name, _ in model.named_modules():
            m = re.match(r"^fc(\d+)$", name)
            if m:
                n_fc_layers = max(n_fc_layers, int(m.group(1)))
        if n_fc_layers == 0:
            raise ValueError("Could not infer number of fc layers from model modules. Provide n_fc_layers.")

    relus: List[str] = []
    for i in range(1, n_fc_layers):
        nm = f"relu{i}"
        relus.append(nm if hasattr(model, nm) else f"fc{i}")
    return relus


# -------------------------
# Activation stats + Hungarian matching
# -------------------------
@dataclass
class CorrStats:
    n: int
    sum_a: torch.Tensor
    sum_a2: torch.Tensor
    sum_b: torch.Tensor
    sum_b2: torch.Tensor
    sum_ab: torch.Tensor

    @classmethod
    def init(cls, d: int, device: torch.device) -> "CorrStats":
        # MPS doesn't support float64
        dtype = torch.float32 if device.type == "mps" else torch.float64
        z = torch.zeros(d, dtype=dtype, device=device)
        m = torch.zeros((d, d), dtype=dtype, device=device)
        return cls(n=0, sum_a=z.clone(), sum_a2=z.clone(), sum_b=z.clone(), sum_b2=z.clone(), sum_ab=m)

    def update(self, a: torch.Tensor, b: torch.Tensor) -> None:
        if a.ndim != 2 or b.ndim != 2:
            raise ValueError(f"Expected 2D activations, got {tuple(a.shape)} and {tuple(b.shape)}")
        if a.shape != b.shape:
            raise ValueError(f"Activation shapes differ: {tuple(a.shape)} vs {tuple(b.shape)}")

        dtype = self.sum_a.dtype  # follow accumulator dtype
        a = a.to(dtype=dtype)
        b = b.to(dtype=dtype)

        self.n += int(a.shape[0])
        self.sum_a += a.sum(dim=0)
        self.sum_a2 += (a * a).sum(dim=0)
        self.sum_b += b.sum(dim=0)
        self.sum_b2 += (b * b).sum(dim=0)
        self.sum_ab += a.t().matmul(b)


    def correlation(self, eps: float = 1e-12) -> torch.Tensor:
        if self.n <= 0:
            raise ValueError("No samples accumulated.")
        n = float(self.n)
        ea = self.sum_a / n
        eb = self.sum_b / n
        ea2 = self.sum_a2 / n
        eb2 = self.sum_b2 / n
        eab = self.sum_ab / n

        var_a = torch.clamp(ea2 - ea * ea, min=0.0)
        var_b = torch.clamp(eb2 - eb * eb, min=0.0)
        std_a = torch.sqrt(var_a + eps)
        std_b = torch.sqrt(var_b + eps)

        cov = eab - ea[:, None] * eb[None, :]
        denom = (std_a[:, None] * std_b[None, :]) + eps
        return cov / denom


@dataclass(frozen=True)
class LayerPermutation:
    """
    A->B assignment at some boundary layer.
    perm_a_to_b[i] = j means A unit i matches B unit j.
    """
    layer_name: str
    perm_a_to_b: torch.Tensor  # [d] long on CPU
    corr_matrix: Optional[torch.Tensor] = None  # [d, d] float64 on CPU (optional)


def hungarian_maximize(similarity: torch.Tensor) -> torch.Tensor:
    """
    similarity: [d, d] (higher is better). Returns perm_a_to_b [d] long on CPU.
    """
    if similarity.ndim != 2 or similarity.shape[0] != similarity.shape[1]:
        raise ValueError(f"Expected square similarity matrix, got {tuple(similarity.shape)}")
    sim = similarity.detach().cpu().numpy()
    row_ind, col_ind = linear_sum_assignment(-sim)  # maximize
    perm = torch.empty(similarity.shape[0], dtype=torch.long)
    perm[row_ind] = torch.from_numpy(col_ind).to(dtype=torch.long)
    return perm


# -------------------------
# Activation collection
# -------------------------
@torch.no_grad()
def compute_layer_permutation_from_activations(
    *,
    model_a: nn.Module,
    model_b: nn.Module,
    loader: torch.utils.data.DataLoader,
    layer_name: str,
    device: torch.device,
    max_batches: Optional[int] = None,
    eps: float = 1e-12,
    unit_dim: int = 1,
    preprocess: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
) -> LayerPermutation:
    """
    Compute Hungarian permutation based on correlation of activations at `layer_name`.
    """
    model_a.eval()
    model_b.eval()
    model_a.to(device)
    model_b.to(device)

    mod_a = dict(model_a.named_modules()).get(layer_name, None)
    mod_b = dict(model_b.named_modules()).get(layer_name, None)
    if mod_a is None or mod_b is None:
        raise KeyError(f"Layer '{layer_name}' not found in both models.")

    acts: Dict[str, torch.Tensor] = {}

    def hook_a(_m, _inp, out):
        acts["a"] = _as_tensor(out).detach()

    def hook_b(_m, _inp, out):  
        acts["b"] = _as_tensor(out).detach()

    ha = mod_a.register_forward_hook(hook_a)
    hb = mod_b.register_forward_hook(hook_b)

    stats: Optional[CorrStats] = None
    try:
        for b_ix, (x, _y) in enumerate(loader):
            if max_batches is not None and b_ix >= max_batches:
                break
            x = x.to(device, non_blocking=True)

            _ = model_a(x)
            a = acts.get("a", None)
            if a is None:
                raise RuntimeError(f"Hook for model_a layer '{layer_name}' did not fire.")

            _ = model_b(x)
            b = acts.get("b", None)
            if b is None:
                raise RuntimeError(f"Hook for model_b layer '{layer_name}' did not fire.")
            
            if preprocess is not None:
                a = preprocess(a)
                b = preprocess(b)
            
            a2 = activation_to_2d(a, unit_dim=unit_dim)
            b2 = activation_to_2d(b, unit_dim=unit_dim)

            # a2 = a.view(a.shape[0], -1)
            # b2 = b.view(b.shape[0], -1)
            # a2 = a.view(a.shape[0], -1).detach().cpu().double()
            # b2 = b.view(b.shape[0], -1).detach().cpu().double()

            if stats is None:
                d = int(a2.shape[1])
                if b2.shape[1] != d:
                    raise ValueError(f"Activation dims differ at {layer_name}: {a2.shape[1]} vs {b2.shape[1]}")
                stats = CorrStats.init(d=d, device=device)

            stats.update(a2, b2)

        if stats is None:
            raise ValueError("No batches processed for activation matching.")

        corr = stats.correlation(eps=eps)  # float64 on device
        perm = hungarian_maximize(corr)    # long on CPU
        return LayerPermutation(layer_name=layer_name, perm_a_to_b=perm, corr_matrix=corr.detach().cpu())
    finally:
        ha.remove()
        hb.remove()


# -------------------------
# Parameter permutation application (function-preserving reparameterization)
# -------------------------
def _perm_rows(W: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
    return W.index_select(0, p.to(device=W.device))


def _perm_cols(W: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
    return W.index_select(1, p.to(device=W.device))


def apply_mlp_hidden_permutations_to_state_dict(
    *,
    state_b: TensorDict,
    perms: Dict[int, torch.Tensor],
    n_layers: Optional[int] = None,
) -> TensorDict:
    """
    Apply a set of hidden-layer permutations to an MLP state dict (function-preserving).

    perms maps hidden layer index -> perm_a_to_b (A->B assignment), with indices 1..N-1.
    We reorder B's hidden units into A order:
      B_perm unit i := B unit p[i]

    For each hidden layer k:
      - permute rows of fc{k}.weight and fc{k}.bias by p_k
      - permute columns of fc{k+1}.weight by p_k
    """
    out = {k: v.clone() for k, v in state_b.items()}
    fc_layers = infer_fc_layer_numbers_from_state(state_b)
    if n_layers is None:
        n_layers = max(fc_layers)
    if fc_layers != list(range(1, n_layers + 1)):
        raise ValueError(f"Expected contiguous fc1..fc{n_layers}, found {fc_layers}")

    for k, p in perms.items():
        if not (1 <= k <= n_layers - 1):
            raise ValueError(f"Permutation index k must be in [1, {n_layers-1}], got {k}")
        p = p.to(dtype=torch.long)

        w_k = f"fc{k}.weight"
        b_k = f"fc{k}.bias"
        w_next = f"fc{k+1}.weight"

        out[w_k] = _perm_rows(out[w_k], p)
        out[b_k] = out[b_k].index_select(0, p.to(device=out[b_k].device))
        out[w_next] = _perm_cols(out[w_next], p)

    return out


def stitch_state_dict_mlp(
    *,
    state_a: TensorDict,
    state_b: TensorDict,
    cut_layer: int,
    n_layers: Optional[int] = None,
) -> TensorDict:
    """
    Stitch MLP params at the parameter level (fc1..fcN):
      cut_layer = k:
        - take fc1..fck from A
        - take fc(k+1)..fcN from B
    """
    fc_layers = infer_fc_layer_numbers_from_state(state_a)
    if n_layers is None:
        n_layers = max(fc_layers)

    expected = list(range(1, n_layers + 1))
    if fc_layers != expected:
        raise ValueError(f"Expected contiguous fc layers {expected}, found {fc_layers}")
    if not (0 <= cut_layer <= n_layers):
        raise ValueError(f"cut_layer must be in [0, {n_layers}], got {cut_layer}")

    out: TensorDict = {}
    for i in range(1, n_layers + 1):
        src = state_a if i <= cut_layer else state_b
        out[f"fc{i}.weight"] = src[f"fc{i}.weight"]
        out[f"fc{i}.bias"] = src[f"fc{i}.bias"]
    return out


def interpolate_state_dict(a: TensorDict, b: TensorDict, lam: float) -> TensorDict:
    """
    Linear interpolation between two state dicts (floating-point params only).
    """
    if a.keys() != b.keys():
        missing_in_b = sorted(set(a.keys()) - set(b.keys()))
        missing_in_a = sorted(set(b.keys()) - set(a.keys()))
        raise KeyError(
            "State dict keysets differ; cannot interpolate safely.\n"
            f"Missing in B: {missing_in_b[:20]}{' ...' if len(missing_in_b) > 20 else ''}\n"
            f"Missing in A: {missing_in_a[:20]}{' ...' if len(missing_in_a) > 20 else ''}\n"
        )
    out: TensorDict = {}
    for k in a.keys():
        va, vb = a[k], b[k]
        if va.dtype.is_floating_point:
            out[k] = (1.0 - lam) * va + lam * vb
        else:
            out[k] = va
    return out


def save_permutations_pickle(perms: Dict[int, torch.Tensor], path: str) -> None:
    payload = {int(k): v.detach().cpu().numpy() for k, v in perms.items()}
    with open(path, "wb") as f:
        pickle.dump(payload, f)


def to_device(params: TensorDict, device: torch.device) -> TensorDict:
    return {k: v.to(device) for k, v in params.items()}
