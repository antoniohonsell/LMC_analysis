#!/usr/bin/env python3
"""
compute_llfc_mlp.py

Compute LLFC cosine similarity between:
  h_lambda(x)   = activations of the interpolated-weight model at lambda
and
  h_lin(x)      = (1-lambda) * h_A(x) + lambda * h_B(x)

Supports:
- MNIST / FashionMNIST
- CIFAR10 / CIFAR100
- MLP + ResNet20 (incl. LayerNorm2d / "flax_ln") + different widths (width_multiplier)

TO RUN MLP:
export PYTHONPATH="$(pwd)"

python LLFC/compute_llfc.py \
--dataset MNIST \
--arch mlp \
--ckpt_a runs_mlp/MNIST/disjoint/seed_0/subset_A/MLP_MNIST_subsetA_seed0_best.pth \
--ckpt_b runs_mlp/MNIST/disjoint/seed_0/subset_B/MLP_MNIST_subsetB_seed0_best.pth \
--out runs/llfc_mlp_MNIST



TO RUN ResNet:
export PYTHONPATH="$(pwd)"
for w in 32; do
    python LLFC/compute_llfc.py \
    --dataset CIFAR100 \
    --arch resnet20 \
    --norm flax_ln \
    --width_multiplier $w \
    --shortcut_option C \
    --device mps \
    --ckpt_a runs_resnet20_$w/CIFAR100/disjoint/seed_0/subset_A/resnet20_CIFAR100_seed0_subsetA_best.pth \
    --ckpt_b runs_resnet20_$w/CIFAR100/disjoint/seed_0/subset_B/resnet20_CIFAR100_seed0_subsetB_best.pth \
    --out runs/llfc_resnet20_ln_cifar100_w$w
    --max_batches 10
done

Notes:
- architectures.build_model(name, num_classes=..., ...) is expected to exist.
- For ResNet20, default hook points are the 9 residual blocks (layer{1,2,3}.{0,1,2}),
  optionally plus the head linear if --include_head is set.
"""

from __future__ import annotations

import argparse
import os
import pickle
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms

# Make imports robust whether this file is in repo root or in a subfolder like LLFC/
_THIS = Path(__file__).resolve()
for _p in (_THIS.parent, _THIS.parent.parent):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from metrics_platonic import cosine_similarity_over_samples, best_scalar_coef  # repo-local

try:
    import architectures  # repo-local
except Exception as e:
    raise ImportError("Could not import repo module `architectures.py`. "
                      "Run from repo root or ensure PYTHONPATH includes the repo.") from e

# Optional: reuse CIFAR stats from your training script if available
try:
    import train_resnet  # repo-local
    _CIFAR_STATS = getattr(train_resnet, "DATASET_STATS", None)
except Exception:
    _CIFAR_STATS = None


TensorDict = Dict[str, torch.Tensor]


def load_wm_perm_pkl(path: str) -> Dict[str, torch.Tensor]:
    """
    Load a weight-matching permutation saved as a pickle:
      {perm_name: np.ndarray | list | torch.Tensor}
    Returns:
      {perm_name: 1D LongTensor}
    """
    with open(path, "rb") as f:
        obj = pickle.load(f)
    if not isinstance(obj, dict):
        raise ValueError(f"--wm_perm must be a pickled dict, got {type(obj)} at {path}")
    perm: Dict[str, torch.Tensor] = {}
    for k, v in obj.items():
        t = torch.as_tensor(v, dtype=torch.long)
        if t.ndim != 1:
            raise ValueError(f"Permutation '{k}' must be 1D, got shape={tuple(t.shape)}")
        perm[str(k)] = t.cpu()
    return perm

def mlp_axes_to_perm_from_state(state: TensorDict) -> Dict[str, Tuple[Optional[str], ...]]:
    """
    Same logic as linear_mode_connectivity/mlp_weight_matching_interp.py:
    build axes_to_perm for an MLP with fc1..fcN. (Hidden layers permuted, classifier output not permuted.)
    """
    pat = re.compile(r"^fc(\d+)\.weight$")
    layer_ids: List[int] = []
    for k in state.keys():
        m = pat.match(k)
        if m:
            layer_ids.append(int(m.group(1)))
    layer_ids = sorted(set(layer_ids))
    if not layer_ids:
        raise KeyError("No fc{n}.weight keys found (needed to build MLP permutation spec).")
    n_layers = max(layer_ids)
    expected = list(range(1, n_layers + 1))
    if layer_ids != expected:
        raise ValueError(f"Expected contiguous fc layers {expected}, found {layer_ids}")

    axes: Dict[str, Tuple[Optional[str], ...]] = {}
    prev_p: Optional[str] = None
    for i in range(1, n_layers):
        p_out = f"P{i}"
        axes[f"fc{i}.weight"] = (p_out, prev_p)  # (out, in)
        axes[f"fc{i}.bias"]   = (p_out,)         # (out,)
        prev_p = p_out
    axes[f"fc{n_layers}.weight"] = (None, prev_p)
    axes[f"fc{n_layers}.bias"]   = (None,)
    return axes


# ---------------------------
# Checkpoint helpers
# ---------------------------
def load_ckpt_state_dict(path: str) -> TensorDict:
    """
    Loads checkpoints saved either as:
      - {"state_dict": ...}  (train_loop.py style), or
      - raw state_dict dict[str, Tensor].
    """
    obj = torch.load(path, map_location="cpu")
    if isinstance(obj, dict) and "state_dict" in obj and isinstance(obj["state_dict"], dict):
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


def filter_to_model_keys(state: TensorDict, model: nn.Module) -> TensorDict:
    """
    Keep only keys that exist in model.state_dict() (helps when ckpt contains extras).
    """
    mk = set(model.state_dict().keys())
    return {k: v for k, v in state.items() if k in mk}


def interpolate_state_dict(a: TensorDict, b: TensorDict, lam: float) -> TensorDict:
    """
    Linear interpolation between two state dicts (floating-point params only):
        (1-lam)*a + lam*b
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
        if torch.is_tensor(va) and torch.is_tensor(vb) and va.dtype.is_floating_point:
            if va.shape != vb.shape:
                raise ValueError(f"Shape mismatch at key={k}: {tuple(va.shape)} vs {tuple(vb.shape)}")
            out[k] = (1.0 - lam) * va + lam * vb
        else:
            out[k] = va
    return out

def interpolate_model_inplace_(model_l: torch.nn.Module,
                              sd_a_dev: dict[str, torch.Tensor],
                              sd_b_dev: dict[str, torch.Tensor],
                              lam: float) -> None:
    sd_l = model_l.state_dict()  # tensors reference module storage
    one_minus = 1.0 - lam
    with torch.no_grad():
        for k, dst in sd_l.items():
            a = sd_a_dev[k]
            b = sd_b_dev[k]
            if dst.dtype.is_floating_point:
                dst.copy_(a)
                dst.mul_(one_minus)
                dst.add_(b, alpha=lam)
            else:
                dst.copy_(a)


# ---------------------------
# Dataset helpers
# ---------------------------
def dataset_info(name: str) -> Tuple[str, int, int, Tuple[int, int, int]]:
    """
    Returns (canonical_name, num_classes, in_channels, input_shape).
    canonical_name uses the conventions used elsewhere in the repo.
    """
    k = name.strip().lower()
    if k in ("mnist",):
        return ("MNIST", 10, 1, (1, 28, 28))
    if k in ("fmnist", "fashionmnist", "fashion-mnist"):
        return ("FMNIST", 10, 1, (1, 28, 28))
    if k in ("cifar10", "cifar-10"):
        return ("CIFAR10", 10, 3, (3, 32, 32))
    if k in ("cifar100", "cifar-100"):
        return ("CIFAR100", 100, 3, (3, 32, 32))
    raise ValueError("Unsupported --dataset. Choose from: MNIST, FMNIST, CIFAR10, CIFAR100")


def make_eval_transform(canonical_dataset: str) -> transforms.Compose:
    """
    Eval transform (no augmentation): ToTensor + Normalize.
    CIFAR stats are taken from train_resnet.DATASET_STATS when available.
    """
    if canonical_dataset == "MNIST":
        mean, std = (0.1307,), (0.3081,)
        return transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean, std)])
    if canonical_dataset == "FMNIST":
        mean, std = (0.2860,), (0.3530,)
        return transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean, std)])

    # CIFAR
    if _CIFAR_STATS is not None and canonical_dataset in _CIFAR_STATS:
        mean = tuple(_CIFAR_STATS[canonical_dataset]["mean"])
        std = tuple(_CIFAR_STATS[canonical_dataset]["std"])
    else:
        # Fallback (common defaults) if train_resnet isn't importable
        if canonical_dataset == "CIFAR10":
            mean, std = (0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)
        else:  # CIFAR100
            mean, std = (0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2761)

    return transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean, std)])


def make_loader(canonical_dataset: str, data_root: str, batch_size: int, num_workers: int, device: torch.device) -> DataLoader:
    tfm = make_eval_transform(canonical_dataset)

    if canonical_dataset == "MNIST":
        ds = torchvision.datasets.MNIST(root=data_root, train=False, download=True, transform=tfm)
    elif canonical_dataset == "FMNIST":
        ds = torchvision.datasets.FashionMNIST(root=data_root, train=False, download=True, transform=tfm)
    elif canonical_dataset == "CIFAR10":
        ds = torchvision.datasets.CIFAR10(root=data_root, train=False, download=True, transform=tfm)
    elif canonical_dataset == "CIFAR100":
        ds = torchvision.datasets.CIFAR100(root=data_root, train=False, download=True, transform=tfm)
    else:
        raise ValueError(f"Unsupported dataset: {canonical_dataset}")

    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
    )


# ---------------------------
# ResNet20-specific inference
# ---------------------------
def infer_width_multiplier_from_state(state: TensorDict) -> Optional[int]:
    """
    For CIFAR-ResNet family in architectures.py:
      conv1.out_channels = 16 * width_multiplier
    """
    w = state.get("conv1.weight", None)
    if w is None or not torch.is_tensor(w) or w.ndim != 4:
        return None
    out_ch = int(w.shape[0])
    if out_ch % 16 != 0:
        return None
    return out_ch // 16


def infer_shortcut_option_from_state(state: TensorDict) -> Optional[str]:
    """
    Detect shortcut option based on presence/shape of shortcut conv weights.

    - Option A: no parameters in shortcut (LambdaLayer)
    - Option B: 1x1 conv in shortcut
    - Option C: 3x3 conv in shortcut (Flax-like in this repo)
    """
    for k, v in state.items():
        if k.endswith("shortcut.0.weight") and torch.is_tensor(v) and v.ndim == 4:
            kh, kw = int(v.shape[2]), int(v.shape[3])
            if (kh, kw) == (1, 1):
                return "B"
            if (kh, kw) == (3, 3):
                return "C"
    # If no shortcut conv weights found, likely option A
    return "A"


# ---------------------------
# Hooking utilities
# ---------------------------
@dataclass
class HookCollector:
    layer_names: List[str]
    activations: Dict[str, torch.Tensor]
    handles: List[torch.utils.hooks.RemovableHandle]

    @staticmethod
    def for_module_names(model: nn.Module, module_names: List[str]) -> "HookCollector":
        name_to_mod = dict(model.named_modules())
        missing = [n for n in module_names if n not in name_to_mod]
        if missing:
            raise KeyError(f"Some requested hook modules were not found: {missing}")

        activations: Dict[str, torch.Tensor] = {}
        handles: List[torch.utils.hooks.RemovableHandle] = []

        def make_hook(nm: str):
            def _hook(_module, _inp, out):
                if isinstance(out, (tuple, list)) and len(out) > 0:
                    out = out[0]
                if not torch.is_tensor(out):
                    raise TypeError(f"Hook output at '{nm}' is not a Tensor (got {type(out)})")
                activations[nm] = out.detach()
            return _hook

        for nm in module_names:
            m = name_to_mod[nm]
            handles.append(m.register_forward_hook(make_hook(nm)))

        return HookCollector(layer_names=list(module_names), activations=activations, handles=handles)

    @staticmethod
    def for_linears(model: nn.Module, include_last: bool) -> "HookCollector":
        linear_names = [name for name, m in model.named_modules() if isinstance(m, nn.Linear)]
        if not include_last and len(linear_names) > 0:
            linear_names = linear_names[:-1]
        return HookCollector.for_module_names(model, linear_names)

    def clear(self) -> None:
        self.activations.clear()

    def remove(self) -> None:
        for h in self.handles:
            h.remove()
        self.handles.clear()


def discover_hook_layers(model: nn.Module, arch: str, include_head: bool) -> List[str]:
    """
    Choose a stable set of hook points based on architecture.
    - resnet20: hook residual blocks layer{1,2,3}.{0,1,2} (+ optional head)
    - otherwise: default to all Linear layers (excluding last unless include_head)
    """
    a = arch.strip().lower()

    if a == "resnet20":
        block_pat = re.compile(r"^layer([123])\.(\d+)$")
        blocks: List[Tuple[int, int, str]] = []
        for name, _m in model.named_modules():
            m = block_pat.match(name)
            if m:
                stage = int(m.group(1))
                idx = int(m.group(2))
                blocks.append((stage, idx, name))
        blocks.sort()
        names = [n for (_s, _i, n) in blocks]
        if include_head:
            # CIFARResNet uses `linear` as classifier head
            if "linear" in dict(model.named_modules()):
                names.append("linear")
        return names

    # Fallback: linears
    linear_names = [name for name, m in model.named_modules() if isinstance(m, nn.Linear)]
    if not include_head and len(linear_names) > 0:
        linear_names = linear_names[:-1]
    return linear_names


def forward_collect(model: nn.Module, hooks: HookCollector, x: torch.Tensor, flatten_input: bool) -> Dict[str, torch.Tensor]:
    hooks.clear()
    if flatten_input:
        x = x.view(x.size(0), -1)
    _ = model(x)
    return dict(hooks.activations)


# ---------------------------
# LLFC computation
# ---------------------------
#@torch.no_grad()
def compute_llfc_over_lambdas(
    *,
    model_a: nn.Module,
    model_b: nn.Module,
    model_l: nn.Module,
    sd_a: TensorDict,
    sd_b: TensorDict,
    hook_names: List[str],
    loader: DataLoader,
    lambdas: List[float],
    device: torch.device,
    flatten_input: bool,
    max_batches: int,
    eps: float,
) -> Dict[str, torch.Tensor]:
    with torch.inference_mode():
        hooks_a = HookCollector.for_module_names(model_a, hook_names)
        hooks_b = HookCollector.for_module_names(model_b, hook_names)
        hooks_l = HookCollector.for_module_names(model_l, hook_names)

        n_layers = len(hook_names)
        n_lams = len(lambdas)

        cos_mean = torch.zeros(n_layers, n_lams, dtype=torch.float64)
        cos_std = torch.zeros(n_layers, n_lams, dtype=torch.float64)
        coef_mean = torch.zeros(n_layers, n_lams, dtype=torch.float64)

        sd_a_dev = {k: v.detach() for k, v in model_a.state_dict().items()}
        sd_b_dev = {k: v.detach() for k, v in model_b.state_dict().items()}
        try:
            model_l.eval()
            for j, lam in enumerate(lambdas):
                interpolate_model_inplace_(model_l, sd_a_dev, sd_b_dev, lam)
                

                sum_cos = torch.zeros(n_layers, device=device, dtype=torch.float32)
                sum_cos2 = torch.zeros(n_layers, device=device, dtype=torch.float32)
                sum_coef = torch.zeros(n_layers, device=device, dtype=torch.float32)
                n_total = 0

                for bi, (x, _y) in enumerate(loader):
                    if max_batches > 0 and bi >= max_batches:
                        break
                    x = x.to(device, non_blocking=True)
                    bs = int(x.size(0))
                    n_total += bs

                    feats_a = forward_collect(model_a, hooks_a, x, flatten_input=flatten_input)
                    feats_b = forward_collect(model_b, hooks_b, x, flatten_input=flatten_input)
                    feats_l = forward_collect(model_l, hooks_l, x, flatten_input=flatten_input)

                    for li, lname in enumerate(hook_names):
                        ha = feats_a[lname]
                        hb = feats_b[lname]
                        hl = feats_l[lname]

                        hint = (1.0 - lam) * ha + lam * hb

                        hl_f   = hl.flatten(start_dim=1)
                        hint_f = hint.flatten(start_dim=1)

                        dot  = (hl_f * hint_f).sum(dim=1)
                        hl2  = (hl_f * hl_f).sum(dim=1).clamp_min(eps)
                        hi2  = (hint_f * hint_f).sum(dim=1).clamp_min(eps)

                        cos  = dot / (torch.sqrt(hl2 * hi2).clamp_min(eps))
                        coef = dot / hi2              # [B]

                        sum_cos[li] += cos.sum()
                        sum_cos2[li] += (cos * cos).sum()
                        sum_coef[li] += coef.sum()

                denom = max(n_total, 1)
                mean = sum_cos / denom
                var = (sum_cos2 / denom) - mean * mean
                std = torch.sqrt(torch.clamp(var, min=0.0))

                # move once per lambda
                cos_mean[:, j]  = mean.detach().cpu().double()
                cos_std[:, j]   = std.detach().cpu().double()
                coef_mean[:, j] = (sum_coef / denom).detach().cpu().double()

                print(f"[LLFC] lambda={lam:.3f} done ({n_total} samples)")

        finally:
            hooks_a.remove()
            hooks_b.remove()
            hooks_l.remove()

        cos_mean_layeravg = cos_mean.mean(dim=0)
        cos_std_layeravg = torch.sqrt(torch.clamp((cos_std ** 2).mean(dim=0), min=0.0))

        return {
            "layers": hook_names,
            "lambdas": torch.tensor(lambdas, dtype=torch.float64),
            "cos_mean": cos_mean,
            "cos_std": cos_std,
            "coef_mean": coef_mean,
            "cos_mean_layeravg": cos_mean_layeravg,
            "cos_std_layeravg": cos_std_layeravg,
        }


# ---------------------------
# Main
# ---------------------------
def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", type=str, required=True, help="MNIST | FMNIST | CIFAR10 | CIFAR100")
    p.add_argument("--data_root", type=str, default="./data")
    p.add_argument("--ckpt_a", type=str, required=True)
    p.add_argument("--ckpt_b", type=str, required=True)
    p.add_argument("--wm_perm", type=str, default=None,
                   help="Optional: path to weight-matching permutation_seed*.pkl. If set, B is permuted before LLFC.")
    p.add_argument("--wm_perm_invert", action="store_true",
                   help="Invert the permutation (useful if your saved mapping is B->A instead of A->B).")
    p.add_argument("--out", type=str, required=True)

    p.add_argument("--arch", type=str, required=True, help="e.g. mlp, resnet20, resnet18, lightnet, ...")

    # ResNet-ish knobs (harmless for non-resnet models)
    p.add_argument("--norm", type=str, default=None, help="bn | ln | flax_ln | none (default depends on arch)")
    p.add_argument("--width_multiplier", type=int, default=None, help="CIFAR ResNet width multiplier (auto-infer if omitted)")
    p.add_argument("--shortcut_option", type=str, default=None, choices=["A", "B", "C"], help="ResNet shortcut option (auto-infer if omitted)")

    # MLP/lightnet knobs
    p.add_argument("--mlp_hidden", type=int, default=512, help="Used for arch=mlp/lightnet where applicable")

    # Runtime knobs
    p.add_argument("--batch_size", type=int, default=512)
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")

    p.add_argument("--lambdas", type=int, default=9, help="Number of lambda points in [0,1]")
    p.add_argument("--max_batches", type=int, default=0, help="0 = full test set; else limit number of batches")
    p.add_argument("--eps", type=float, default=1e-12)

    # Hooking / feature details
    p.add_argument("--include_head", action="store_true", help="Include final classifier layer among hook points")

    # Flatten control (two flags so we can have an auto-default)
    p.add_argument("--flatten_input", action="store_true", help="Force flatten input before forward")
    p.add_argument("--no_flatten_input", action="store_true", help="Force NOT flatten input before forward")

    args = p.parse_args()

    os.makedirs(args.out, exist_ok=True)
    device = torch.device(args.device)

    ds_name, num_classes, in_channels, input_shape = dataset_info(args.dataset)

    # Load + normalize state dicts
    sd_a_raw = normalize_state_dict_keys(load_ckpt_state_dict(args.ckpt_a))
    sd_b_raw = normalize_state_dict_keys(load_ckpt_state_dict(args.ckpt_b))

    arch = args.arch.strip()

    # Choose defaults for ResNet20 LayerNorm runs
    norm = args.norm
    if norm is None:
        norm = "flax_ln" if arch.lower() == "resnet20" else "bn"

    width_mult = args.width_multiplier
    shortcut_opt = args.shortcut_option

    if arch.lower() == "resnet20":
        if width_mult is None:
            width_mult = infer_width_multiplier_from_state(sd_a_raw)
        if shortcut_opt is None:
            shortcut_opt = infer_shortcut_option_from_state(sd_a_raw)
        if width_mult is None:
            raise ValueError("Could not infer --width_multiplier from checkpoint. Please pass --width_multiplier explicitly.")
        if shortcut_opt is None:
            shortcut_opt = "A"

    # Auto flatten default:
    # - mlp/lightnet: flatten
    # - resnets/convnets: don't flatten
    if args.flatten_input and args.no_flatten_input:
        raise ValueError("Cannot set both --flatten_input and --no_flatten_input")
    if args.flatten_input:
        flatten_input = True
    elif args.no_flatten_input:
        flatten_input = False
    else:
        flatten_input = arch.lower() in ("mlp", "lightnet")  # safe default

    # Build models
    build_kwargs: Dict[str, Any] = {
        "num_classes": num_classes,
        "in_channels": in_channels,
        "norm": norm,
    }
    # Pass input_shape/hidden for MLP-like models in this repo
    if arch.lower() == "mlp":
        build_kwargs["input_shape"] = input_shape
        build_kwargs["hidden"] = int(args.mlp_hidden)
    if arch.lower() == "lightnet":
        build_kwargs["input_shape"] = input_shape
        build_kwargs["hidden"] = int(args.mlp_hidden)
    # CIFAR ResNet family extras
    if arch.lower().startswith("resnet"):
        if width_mult is not None:
            build_kwargs["width_multiplier"] = int(width_mult)
        if shortcut_opt is not None:
            build_kwargs["shortcut_option"] = str(shortcut_opt)

    model_a = architectures.build_model(arch, **build_kwargs).to(device).eval()
    model_b = architectures.build_model(arch, **build_kwargs).to(device).eval()
    model_l = architectures.build_model(arch, **build_kwargs).to(device).eval()

    # Filter sd to exactly model keys (avoid strict-load issues from extra ckpt keys)
    sd_a = filter_to_model_keys(sd_a_raw, model_a)
    sd_b = filter_to_model_keys(sd_b_raw, model_b)

    # Ensure both are aligned key-wise for interpolation
    # Ensure both are aligned key-wise for interpolation (fail fast)
    if sd_a.keys() != sd_b.keys():
        missing_in_b = sorted(set(sd_a.keys()) - set(sd_b.keys()))
        missing_in_a = sorted(set(sd_b.keys()) - set(sd_a.keys()))
        raise KeyError(
            "State dict keysets differ after filtering to model keys; cannot safely compare/interpolate.\n"
            f"Missing in B: {missing_in_b[:20]}{' ...' if len(missing_in_b) > 20 else ''}\n"
            f"Missing in A: {missing_in_a[:20]}{' ...' if len(missing_in_a) > 20 else ''}\n"
        )
    # -------------------------`
    # OPTIONAL: permute B using a saved weight-matching permutation
    # -------------------------
    if args.wm_perm is not None:
        try:
            from linear_mode_connectivity.weight_matching_torch import (
                apply_permutation,
                permutation_spec_from_axes_to_perm,
                resnet20_layernorm_permutation_spec,
            )
        except Exception as e:
            raise ImportError(
                "Failed to import linear_mode_connectivity.weight_matching_torch (needed for --wm_perm). "
                "Make sure dependencies (notably SciPy) are installed."
            ) from e

        perm = load_wm_perm_pkl(args.wm_perm)
        if args.wm_perm_invert:
            perm = {k: torch.argsort(v) for k, v in perm.items()}

        if arch.lower() == "resnet20":
            ps = resnet20_layernorm_permutation_spec(
                shortcut_option=str(shortcut_opt or "C"),
                state_dict=sd_b,   # prune spec to exactly existing keys
            )
        elif arch.lower() in ("mlp", "lightnet"):
            axes_to_perm = mlp_axes_to_perm_from_state(sd_a)
            ps = permutation_spec_from_axes_to_perm(axes_to_perm)
        else:
            raise ValueError(f"--wm_perm currently supported only for arch in {{resnet20, mlp, lightnet}}; got {arch}")

        needed = set(ps.perm_to_axes.keys())
        missing = sorted(needed - set(perm.keys()))
        if missing:
            raise KeyError(f"--wm_perm is missing permutation keys required by spec: {missing}")
        perm = {k: v for k, v in perm.items() if k in needed}

        # Replace B with P(B)
        sd_b = apply_permutation(ps, perm, sd_b)

    model_a.load_state_dict(sd_a, strict=True)
    model_b.load_state_dict(sd_b, strict=True)

    # Loader
    loader = make_loader(ds_name, args.data_root, args.batch_size, args.num_workers, device)

    # Hook points
    hook_names = discover_hook_layers(model_a, arch, include_head=args.include_head)
    if not hook_names:
        raise RuntimeError("No hook layers discovered. Try --include_head or verify the architecture name.")

    # Lambdas
    lambdas = torch.linspace(0.0, 1.0, steps=int(args.lambdas)).tolist()

    # Compute
    results = compute_llfc_over_lambdas(
        model_a=model_a,
        model_b=model_b,
        model_l=model_l,
        sd_a=sd_a,
        sd_b=sd_b,
        hook_names=hook_names,
        loader=loader,
        lambdas=lambdas,
        device=device,
        flatten_input=flatten_input,
        max_batches=int(args.max_batches),
        eps=float(args.eps),
    )

    # Save
    tag_parts = [ds_name.lower(), arch.lower()]
    if arch.lower() == "resnet20":
        tag_parts += [f"norm-{norm}", f"w{width_mult}", f"sc{shortcut_opt}"]
    tag = "_".join(tag_parts)

    save_path = os.path.join(args.out, f"llfc_cos_{tag}.pt")
    payload = {
        "dataset": ds_name,
        "arch": arch,
        "norm": norm,
        "width_multiplier": width_mult,
        "shortcut_option": shortcut_opt,
        "num_classes": num_classes,
        "in_channels": in_channels,
        "ckpt_a": args.ckpt_a,
        "ckpt_b": args.ckpt_b,
        "wm_perm": args.wm_perm,
        "wm_perm_invert": bool(args.wm_perm_invert),
        "include_head": bool(args.include_head),
        "flatten_input": bool(flatten_input),
        **results,
    }
    torch.save(payload, save_path)
    print(f"Saved: {save_path}")
    print(f"Hook layers ({len(hook_names)}): {hook_names}")


if __name__ == "__main__":
    main()