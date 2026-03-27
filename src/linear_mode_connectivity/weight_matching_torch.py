# weight_matching_torch.py
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment


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
    Permutation spec for CIFAR ResNet-20.

    This is made robust to:
      - naming differences: (n1/n2) vs (norm1/norm2)
      - norm param layouts: LayerNorm2d wrapper uses `.ln.weight/.ln.bias`,
        BN/GN use `.weight/.bias` (+ optional running stats)
      - shortcut option: A has no shortcut params; B/C have shortcut conv+norm params
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



def resnet50_permutation_spec(
    *,
    state_dict: Optional[Dict[str, torch.Tensor]] = None,
) -> PermutationSpec:
    """
    Permutation spec for ResNet50CIFAR (TVBottleneck, BN).

    Background perms (stage output channels):
      P_bg0: stem, 64 ch    P_bg1: layer1, 256 ch
      P_bg2: layer2, 512 ch   P_bg3: layer3, 1024 ch   P_bg4: layer4, 2048 ch

    Per-block inner perms (bottleneck internals):
      P_layer{l}_{b}_inner1: after conv1+bn1 (reduction)
      P_layer{l}_{b}_inner2: after conv2+bn2 (spatial)

    Block counts are inferred from state_dict when provided.
    """
    import re as _re

    def bn_wb(base: str, p: str) -> Dict[str, Tuple[Optional[str], ...]]:
        d: Dict[str, Tuple[Optional[str], ...]] = {}
        for suffix in ("weight", "bias", "running_mean", "running_var"):
            key = f"{base}.{suffix}"
            if state_dict is None or key in state_dict:
                d[key] = (p,)
        return d

    axes: Dict[str, Tuple[Optional[str], ...]] = {}

    # Stem
    axes["conv1.weight"] = ("P_bg0", None, None, None)
    axes.update(bn_wb("bn1", "P_bg0"))

    # layer → (default_n_blocks, p_in, p_out)
    layer_cfgs = [
        (1, 3, "P_bg0", "P_bg1"),
        (2, 4, "P_bg1", "P_bg2"),
        (3, 6, "P_bg2", "P_bg3"),
        (4, 3, "P_bg3", "P_bg4"),
    ]

    for layer_idx, default_n_blocks, p_in_stage, p_out_stage in layer_cfgs:
        # Infer actual block count from checkpoint.
        n_blocks = default_n_blocks
        if state_dict is not None:
            found = {int(m.group(1))
                     for k in state_dict
                     for m in [_re.match(rf"layer{layer_idx}\.(\d+)\.", k)] if m}
            if found:
                n_blocks = max(found) + 1

        for b in range(n_blocks):
            prefix = f"layer{layer_idx}.{b}"
            inner1 = f"P_layer{layer_idx}_{b}_inner1"
            inner2 = f"P_layer{layer_idx}_{b}_inner2"
            p_in  = p_in_stage if b == 0 else p_out_stage
            p_out = p_out_stage

            # Bottleneck: conv1 (1×1) → conv2 (3×3) → conv3 (1×1)
            axes[f"{prefix}.conv1.weight"] = (inner1, p_in,  None, None)
            axes.update(bn_wb(f"{prefix}.bn1", inner1))
            axes[f"{prefix}.conv2.weight"] = (inner2, inner1, None, None)
            axes.update(bn_wb(f"{prefix}.bn2", inner2))
            axes[f"{prefix}.conv3.weight"] = (p_out,  inner2, None, None)
            axes.update(bn_wb(f"{prefix}.bn3", p_out))

            # Downsample (present when in/out channels differ — always block 0).
            ds_conv = f"{prefix}.downsample.0.weight"
            ds_bn   = f"{prefix}.downsample.1"
            if state_dict is None or ds_conv in state_dict:
                axes[ds_conv] = (p_out, p_in, None, None)
                axes.update(bn_wb(ds_bn, p_out))

    # Classifier
    axes["fc.weight"] = (None, "P_bg4")
    axes["fc.bias"]   = (None,)

    # Prune to keys that exist in the checkpoint.
    if state_dict is not None:
        axes = {k: v for k, v in axes.items() if k in state_dict}

    return permutation_spec_from_axes_to_perm(axes)


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
    w = params[k]
    axis_perms = ps.axes_to_perm[k]
    for axis, p in enumerate(axis_perms):
        if axis == except_axis:
            continue
        if p is not None:
            w = _index_select(w, axis, perm[p])
    return w


def apply_permutation(
    ps: PermutationSpec,
    perm: Dict[str, torch.Tensor],
    params: Dict[str, torch.Tensor],
) -> Dict[str, torch.Tensor]:
    out = {}
    for k in params.keys():
        if k not in ps.axes_to_perm:
            # leave untouched if not part of the spec
            out[k] = params[k]
        else:
            out[k] = get_permuted_param(ps, perm, k, params)
    return out


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
    Find a permutation of params_b to best match params_a (channel-wise) using Hungarian updates.
    """
    # sizes inferred from params_a shapes
    perm_sizes = {
        p: params_a[axes[0][0]].shape[axes[0][1]] for p, axes in ps.perm_to_axes.items()
    }

    if init_perm is None:
        perm = {p: torch.arange(n, dtype=torch.long) for p, n in perm_sizes.items()}
    else:
        perm = {p: v.clone().to(dtype=torch.long) for p, v in init_perm.items()}

    perm_names = list(perm.keys())

    # Ensure CPU tensors for matching
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

            # SciPy maximize fallback if needed
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
