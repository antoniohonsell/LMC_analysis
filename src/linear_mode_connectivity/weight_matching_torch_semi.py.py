# linear_mode_connectivity/weight_matching_torch_semi.py
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment


# 
# Same core data structures of weight_matching_torch.py

@dataclass(frozen=True)
class PermutationSpec:
    # perm_name -> list of (state_dict_key, axis_index)
    perm_to_axes: Dict[str, List[Tuple[str, int]]]
    # state_dict_key -> tuple[perm_name or None] of length = tensor.ndim
    axes_to_perm: Dict[str, Tuple[Optional[str], ...]]


def permutation_spec_from_axes_to_perm(
    axes_to_perm: Dict[str, Tuple[Optional[str], ...]]
) -> PermutationSpec:
    perm_to_axes = defaultdict(list)
    for k, axis_perms in axes_to_perm.items():
        for axis, p in enumerate(axis_perms):
            if p is not None:
                perm_to_axes[p].append((k, axis))
    return PermutationSpec(perm_to_axes=dict(perm_to_axes), axes_to_perm=axes_to_perm)


def resnet20_layernorm_permutation_spec(
    *,
    shortcut_option: str = "C",
    state_dict: Optional[Dict[str, torch.Tensor]] = None,
) -> PermutationSpec:
    """
    Same as linear_mode_connectivity/weight_matching_torch.py, copied verbatim so this file
    is self-contained.
    """
    conv_w = lambda k, p_in, p_out: {k: (p_out, p_in, None, None)}
    linear_wb = lambda k, p_in: {f"{k}.weight": (None, p_in), f"{k}.bias": (None,)}

    def norm_wb(base: str, p: str) -> Dict[str, Tuple[Optional[str], ...]]:
        d: Dict[str, Tuple[Optional[str], ...]] = {}
        # Default: assume LayerNorm2d wrapper layout
        if state_dict is None:
            d[f"{base}.ln.weight"] = (p,)
            d[f"{base}.ln.bias"] = (p,)
            return d

        # LayerNorm2d wrapper
        if f"{base}.ln.weight" in state_dict:
            d[f"{base}.ln.weight"] = (p,)
            if f"{base}.ln.bias" in state_dict:
                d[f"{base}.ln.bias"] = (p,)
            return d

        # BN / GN / other affine norms
        if f"{base}.weight" in state_dict:
            d[f"{base}.weight"] = (p,)
        if f"{base}.bias" in state_dict:
            d[f"{base}.bias"] = (p,)
        if f"{base}.running_mean" in state_dict:
            d[f"{base}.running_mean"] = (p,)
        if f"{base}.running_var" in state_dict:
            d[f"{base}.running_var"] = (p,)
        return d

    # Detect whether your checkpoint uses n1/n2 (current code) or norm1/norm2 (old code)
    stem_norm = "n1" if (state_dict and any(k.startswith("n1.") for k in state_dict)) else "norm1"
    block_n1 = "n1" if (state_dict and any(".n1." in k for k in state_dict)) else "norm1"
    block_n2 = "n2" if (state_dict and any(".n2." in k for k in state_dict)) else "norm2"

    # Detect shortcut params from the checkpoint if possible (more reliable than the CLI flag)
    use_shortcut = str(shortcut_option).upper() in ("B", "C")
    if state_dict is not None:
        use_shortcut = any(k.endswith("shortcut.0.weight") for k in state_dict.keys())

    axes: Dict[str, Tuple[Optional[str], ...]] = {}

    # Stem
    axes.update(conv_w("conv1.weight", None, "P_bg0"))
    axes.update(norm_wb(stem_norm, "P_bg0"))

    def easyblock(layer: int, block: int, p: str) -> Dict[str, Tuple[Optional[str], ...]]:
        inner = f"P_layer{layer}_{block}_inner"
        prefix = f"layer{layer}.{block}"
        d: Dict[str, Tuple[Optional[str], ...]] = {}
        d.update(conv_w(f"{prefix}.conv1.weight", p, inner))
        d.update(norm_wb(f"{prefix}.{block_n1}", inner))
        d.update(conv_w(f"{prefix}.conv2.weight", inner, p))
        d.update(norm_wb(f"{prefix}.{block_n2}", p))
        return d

    def transitionblock(layer: int, block: int, p_in: str, p_out: str) -> Dict[str, Tuple[Optional[str], ...]]:
        inner = f"P_layer{layer}_{block}_inner"
        prefix = f"layer{layer}.{block}"
        d: Dict[str, Tuple[Optional[str], ...]] = {}
        d.update(conv_w(f"{prefix}.conv1.weight", p_in, inner))
        d.update(norm_wb(f"{prefix}.{block_n1}", inner))
        d.update(conv_w(f"{prefix}.conv2.weight", inner, p_out))
        d.update(norm_wb(f"{prefix}.{block_n2}", p_out))
        if use_shortcut:
            d.update(conv_w(f"{prefix}.shortcut.0.weight", p_in, p_out))
            d.update(norm_wb(f"{prefix}.shortcut.1", p_out))
        return d

    # layer1
    axes.update(easyblock(1, 0, "P_bg0"))
    axes.update(easyblock(1, 1, "P_bg0"))
    axes.update(easyblock(1, 2, "P_bg0"))

    # layer2
    axes.update(transitionblock(2, 0, "P_bg0", "P_bg1"))
    axes.update(easyblock(2, 1, "P_bg1"))
    axes.update(easyblock(2, 2, "P_bg1"))

    # layer3
    axes.update(transitionblock(3, 0, "P_bg1", "P_bg2"))
    axes.update(easyblock(3, 1, "P_bg2"))
    axes.update(easyblock(3, 2, "P_bg2"))

    # classifier
    axes.update(linear_wb("linear", "P_bg2"))

    # IMPORTANT: prune spec to exactly what exists in the checkpoint, to avoid KeyErrors later
    if state_dict is not None:
        axes = {k: v for k, v in axes.items() if k in state_dict}

    return permutation_spec_from_axes_to_perm(axes)


# -------------------------
# Application helpers
# -------------------------
def _index_select(x: torch.Tensor, dim: int, index: torch.Tensor) -> torch.Tensor:
    # index must be 1D long on same device
    return torch.index_select(x, dim, index)


def get_permuted_param(
    ps: PermutationSpec,
    perm: Dict[str, torch.Tensor],
    k: str,
    params: Dict[str, torch.Tensor],
    except_axis: Optional[int] = None,
) -> torch.Tensor:
    """
    Works for:
      - permutations (index is a reordering of [0..n-1])
      - semi-permutations (index is a subset of [0..nB-1], length nA < nB)
    """
    w = params[k]
    axis_perms = ps.axes_to_perm[k]
    for axis, p in enumerate(axis_perms):
        if axis == except_axis:
            continue
        if p is not None:
            w = _index_select(w, axis, perm[p].to(device=w.device))
    return w


def apply_permutation(
    ps: PermutationSpec,
    perm: Dict[str, torch.Tensor],
    params: Dict[str, torch.Tensor],
) -> Dict[str, torch.Tensor]:
    out: Dict[str, torch.Tensor] = {}
    for k in params.keys():
        if k not in ps.axes_to_perm:
            out[k] = params[k]
        else:
            out[k] = get_permuted_param(ps, perm, k, params)
    return out


# -------------------------
# Classic (square) weight matching (copied behavior)
# -------------------------
def weight_matching(
    seed: int,
    ps: PermutationSpec,
    params_a: Dict[str, torch.Tensor],
    params_b: Dict[str, torch.Tensor],
    max_iter: int = 100,
    init_perm: Optional[Dict[str, torch.Tensor]] = None,
    silent: bool = False,
) -> Dict[str, torch.Tensor]:
    """
    Original behavior: assumes every perm acts on axes with the SAME size in A and B.
    """
    perm_sizes = {
        p: params_a[axes[0][0]].shape[axes[0][1]] for p, axes in ps.perm_to_axes.items()
    }

    if init_perm is None:
        perm = {p: torch.arange(n, dtype=torch.long) for p, n in perm_sizes.items()}
    else:
        perm = {p: v.clone().to(dtype=torch.long) for p, v in init_perm.items()}

    perm_names = list(perm.keys())

    params_a_cpu = {k: v.detach().cpu() for k, v in params_a.items()}
    params_b_cpu = {k: v.detach().cpu() for k, v in params_b.items()}
    perm = {k: v.detach().cpu() for k, v in perm.items()}

    for it in range(max_iter):
        progress = False
        rng = np.random.default_rng(seed + it)
        for p_ix in rng.permutation(len(perm_names)):
            p = perm_names[p_ix]
            n = perm_sizes[p]

            A = torch.zeros((n, n), dtype=torch.float64)
            for wk, axis in ps.perm_to_axes[p]:
                w_a = params_a_cpu[wk]
                w_b = get_permuted_param(ps, perm, wk, params_b_cpu, except_axis=axis)

                w_a = torch.movedim(w_a, axis, 0).reshape(n, -1).to(torch.float64)
                w_b = torch.movedim(w_b, axis, 0).reshape(n, -1).to(torch.float64)
                A += w_a @ w_b.T

            A_np = A.numpy()

            try:
                ri, ci = linear_sum_assignment(A_np, maximize=True)
            except TypeError:
                ri, ci = linear_sum_assignment(-A_np)

            assert np.all(ri == np.arange(len(ri)))

            oldL = float(A_np[np.arange(n), perm[p].numpy()].sum())
            newL = float(A_np[np.arange(n), ci].sum())

            if not silent:
                print(f"{it}/{p}: {newL - oldL:.6e}")

            if newL > oldL + 1e-12:
                progress = True
                perm[p] = torch.tensor(ci, dtype=torch.long)

        if not progress:
            break

    return perm


# -------------------------
# Semi-permutation weight matching (rectangular Hungarian)
# -------------------------
def _infer_perm_sizes_ab(
    ps: PermutationSpec,
    params_a: Dict[str, torch.Tensor],
    params_b: Dict[str, torch.Tensor],
) -> Dict[str, Tuple[int, int]]:
    """
    For each perm name p, infer (nA, nB) from all tensors/axes that reference p.
    Enforces consistency within A and within B.
    """
    out: Dict[str, Tuple[int, int]] = {}
    for p, axes in ps.perm_to_axes.items():
        a_sizes = {int(params_a[wk].shape[axis]) for (wk, axis) in axes}
        b_sizes = {int(params_b[wk].shape[axis]) for (wk, axis) in axes}
        if len(a_sizes) != 1 or len(b_sizes) != 1:
            raise ValueError(
                f"Inconsistent sizes for perm '{p}'. "
                f"A sizes={sorted(a_sizes)}, B sizes={sorted(b_sizes)}. "
                f"Spec may not match these checkpoints."
            )
        out[p] = (next(iter(a_sizes)), next(iter(b_sizes)))
    return out


def weight_matching_semi(
    seed: int,
    ps: PermutationSpec,
    params_a: Dict[str, torch.Tensor],
    params_b: Dict[str, torch.Tensor],
    max_iter: int = 100,
    init_perm: Optional[Dict[str, torch.Tensor]] = None,
    silent: bool = False,
) -> Dict[str, torch.Tensor]:
    """
    Semi-permutation extension of weight matching.

    For each perm p:
      - nA = size of the permuted axis in params_a
      - nB = size of the permuted axis in params_b

    Cases:
      - nA == nB: standard permutation (same behavior as weight_matching)
      - nA <  nB: semi-permutation (injective): returns an index vector of length nA with
                 DISTINCT entries in [0..nB-1]. This effectively *selects* and reorders a subset
                 of B channels to match A.
      - nA >  nB: not supported here (cannot be injective). Swap A/B if you want to crop the other model.

    Return format:
      perm[p] is a 1D LongTensor of length nA, to be used with apply_permutation().
    """
    perm_sizes = _infer_perm_sizes_ab(ps, params_a, params_b)

    # Initialize
    if init_perm is None:
        perm: Dict[str, torch.Tensor] = {}
        rng0 = np.random.default_rng(seed)
        for p, (nA, nB) in perm_sizes.items():
            if nA == nB:
                perm[p] = torch.arange(nA, dtype=torch.long)
            elif nA < nB:
                # random distinct subset (deterministic via seed)
                subset = rng0.permutation(nB)[:nA]
                perm[p] = torch.from_numpy(subset).to(dtype=torch.long)
            else:
                raise ValueError(
                    f"Semi-matching requires nA <= nB for every perm. For '{p}' got nA={nA}, nB={nB}. "
                    f"Swap (params_a, params_b) if you want to crop the other model."
                )
    else:
        perm = {p: v.clone().to(dtype=torch.long) for p, v in init_perm.items()}

    # Validate init
    for p, (nA, nB) in perm_sizes.items():
        if p not in perm:
            raise KeyError(f"init_perm missing key '{p}'")
        idx = perm[p]
        if idx.ndim != 1 or int(idx.numel()) != nA:
            raise ValueError(f"init_perm['{p}'] must be 1D of length nA={nA}, got shape {tuple(idx.shape)}")
        if idx.min().item() < 0 or idx.max().item() >= nB:
            raise ValueError(f"init_perm['{p}'] has indices out of range [0, {nB-1}]")
        if nA <= nB:
            # enforce injective selection
            if len(torch.unique(idx.cpu())) != nA:
                raise ValueError(f"init_perm['{p}'] must have distinct indices for semi-permutation (nA={nA} <= nB={nB})")

    perm_names = list(perm.keys())

    # CPU tensors for matching (like your original)
    params_a_cpu = {k: v.detach().cpu() for k, v in params_a.items()}
    params_b_cpu = {k: v.detach().cpu() for k, v in params_b.items()}
    perm = {k: v.detach().cpu() for k, v in perm.items()}

    for it in range(max_iter):
        progress = False
        rng = np.random.default_rng(seed + it)

        for p_ix in rng.permutation(len(perm_names)):
            p = perm_names[p_ix]
            nA, nB = perm_sizes[p]

            # Build rectangular similarity matrix: [nA, nB]
            S = torch.zeros((nA, nB), dtype=torch.float64)
            for wk, axis in ps.perm_to_axes[p]:
                w_a = params_a_cpu[wk]
                w_b = get_permuted_param(ps, perm, wk, params_b_cpu, except_axis=axis)

                # axis being optimized: keep full nB on B side, full nA on A side
                w_a2 = torch.movedim(w_a, axis, 0).reshape(nA, -1).to(torch.float64)
                w_b2 = torch.movedim(w_b, axis, 0).reshape(nB, -1).to(torch.float64)
                S += w_a2 @ w_b2.T

            S_np = S.numpy()

            # Current objective
            oldL = float(S_np[np.arange(nA), perm[p].numpy()].sum())

            # Rectangular Hungarian maximize:
            try:
                row_ind, col_ind = linear_sum_assignment(S_np, maximize=True)
            except TypeError:
                row_ind, col_ind = linear_sum_assignment(-S_np)

            # For nA <= nB, every row is matched exactly once (but row_ind may be unsorted)
            new_perm = np.empty((nA,), dtype=np.int64)
            new_perm[row_ind] = col_ind

            newL = float(S_np[np.arange(nA), new_perm].sum())

            if not silent:
                print(f"{it}/{p}: {newL - oldL:.6e} (nA={nA}, nB={nB})")

            if newL > oldL + 1e-12:
                progress = True
                perm[p] = torch.from_numpy(new_perm).to(dtype=torch.long)

        if not progress:
            break

    return perm
