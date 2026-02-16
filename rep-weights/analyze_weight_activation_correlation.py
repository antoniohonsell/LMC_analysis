#!/usr/bin/env python3
"""
analyze_weight_activation_correlation.py

Compute correlations between a layer's weights and the representations (activations)
at that layer, using models/datasets defined in this repo.

Typical usage (run from repo root):
  PYTHONPATH=. python analyze_weight_activation_correlation.py \
    --ckpt ./runs_resnet20_ln_warmcos/CIFAR10/full/seed_0/..._best.pth \
    --dataset CIFAR10 --model resnet20 --layer layer3.2.conv2 --split test

Layer naming is via model.named_modules() (e.g., conv1, layer1.0.conv1, linear, fc2, ...).

HOW TO USE:

PYTHONPATH=. python analyze_weight_activation_correlation.py \
  --ckpt ./runs_resnet20_ln_warmcos/CIFAR10/full/seed_0/resnet20_CIFAR10_full_seed0_best.pth \
  --dataset CIFAR10 --model resnet20 \
  --layer layer3.2.conv2 --hook output --split test \
  --samplewise
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Optional, Tuple, Any, List

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

import architectures
import datasets as ds_utils
import utils


# -----------------------
# Checkpoint helpers
# -----------------------
def load_ckpt_state_dict(path: str) -> Dict[str, torch.Tensor]:
    obj = torch.load(path, map_location="cpu")
    if isinstance(obj, dict) and "state_dict" in obj and isinstance(obj["state_dict"], dict):
        return obj["state_dict"]
    if isinstance(obj, dict):
        # allow raw state_dict saved directly
        return obj
    raise ValueError(f"Unrecognized checkpoint format at: {path}")


def normalize_state_dict_keys(state: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """
    Strip common wrappers (DataParallel/Lightning/etc.) by removing leading prefixes
    like 'module.' / 'model.' / 'net.' if they appear on *all* keys.
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


# -----------------------
# Simple inference (best-effort)
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
    # MLP: last fcN.weight has shape [num_classes, hidden]
    fc_keys = [k for k in state.keys() if k.startswith("fc") and k.endswith(".weight")]
    if fc_keys:
        # choose max layer index
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
    # conv1 out_channels = 16 * width_multiplier
    if "conv1.weight" not in state:
        return None
    w = int(state["conv1.weight"].shape[0] // 16)
    return w if w > 0 else None


def infer_resnet20_shortcut_option(state: Dict[str, torch.Tensor]) -> str:
    # matches your heuristic: option "C" if any shortcut conv exists, else "A"
    return "C" if any(k.endswith("shortcut.0.weight") for k in state.keys()) else "A"


def infer_norm_from_state(state: Dict[str, torch.Tensor]) -> str:
    # Best-effort:
    # - BN has running_mean/running_var
    # - flax_ln uses LayerNorm2d wrapper storing params under ".ln."
    keys = state.keys()
    if any(k.endswith("running_mean") or k.endswith("running_var") for k in keys):
        return "bn"
    if any(".ln.weight" in k or ".ln.bias" in k for k in keys):
        return "flax_ln"
    # GroupNorm(1, C) used for "ln" (LN-like) typically has only weight/bias
    return "ln"


# -----------------------
# Stats / correlation
# -----------------------
def _pearson_corr(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    if x.size != y.size or x.size == 0:
        return float("nan")
    x = x - x.mean()
    y = y - y.mean()
    denom = np.sqrt(np.sum(x * x) * np.sum(y * y))
    if denom <= 0:
        return float("nan")
    return float(np.sum(x * y) / denom)


def _rankdata(a: np.ndarray) -> np.ndarray:
    """
    Average ranks for ties (Spearman).
    """
    a = np.asarray(a, dtype=np.float64).reshape(-1)
    order = np.argsort(a, kind="mergesort")
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(a.size, dtype=np.float64)

    # tie handling
    sorted_a = a[order]
    i = 0
    while i < a.size:
        j = i + 1
        while j < a.size and sorted_a[j] == sorted_a[i]:
            j += 1
        if j - i > 1:
            avg = (i + (j - 1)) / 2.0
            ranks[order[i:j]] = avg
        i = j
    return ranks


def _spearman_corr(x: np.ndarray, y: np.ndarray) -> float:
    return _pearson_corr(_rankdata(x), _rankdata(y))


def _reduce_activation(act: torch.Tensor) -> torch.Tensor:
    """
    Reduce [B, C, H, W] -> [B, C] by spatial mean; keep [B, D] as-is.
    """
    if act.ndim <= 2:
        return act
    # mean over all dims except batch and channel/feature dim=1
    dims = tuple(range(2, act.ndim))
    return act.mean(dim=dims)


def _weight_vector_for_module(mod: nn.Module, mode: str, weight_stat: str) -> torch.Tensor:
    """
    Produce a 1D vector (length = #units) from mod.weight.

    mode:
      - "output": per-output-unit stats (rows / out_channels)
      - "input" : per-input-unit stats (cols / in_channels)
    weight_stat: "l2" | "l1" | "absmean"
    """
    if not hasattr(mod, "weight") or mod.weight is None:
        raise ValueError("Selected module has no .weight parameter.")
    W = mod.weight.detach().to(dtype=torch.float32)

    if W.ndim == 1:
        # e.g., (GroupNorm/LayerNorm gamma): already per-channel
        v = W.abs() if weight_stat in ("l1", "absmean") else W
        return v.clone()

    if isinstance(mod, nn.Linear):
        # W: [out, in]
        if mode == "output":
            X = W
        elif mode == "input":
            X = W.transpose(0, 1)  # [in, out]
        else:
            raise ValueError(f"Unknown mode: {mode}")
        X = X.reshape(X.shape[0], -1)
    elif isinstance(mod, nn.Conv2d):
        # W: [out, in, kH, kW]
        if mode == "output":
            X = W.reshape(W.shape[0], -1)  # [out, in*kH*kW]
        elif mode == "input":
            X = W.permute(1, 0, 2, 3).contiguous().reshape(W.shape[1], -1)  # [in, out*kH*kW]
        else:
            raise ValueError(f"Unknown mode: {mode}")
    else:
        # Generic tensor: treat dim0 as "output units", dim1 as "input units" if requested.
        if mode == "output":
            X = W.reshape(W.shape[0], -1)
        elif mode == "input" and W.ndim >= 2:
            X = W.transpose(0, 1).contiguous().reshape(W.shape[1], -1)
        else:
            raise ValueError(f"Cannot compute mode={mode} for weight with shape {tuple(W.shape)}")

    if weight_stat == "l2":
        return torch.linalg.vector_norm(X, ord=2, dim=1)
    if weight_stat == "l1":
        return X.abs().sum(dim=1)
    if weight_stat == "absmean":
        return X.abs().mean(dim=1)
    raise ValueError(f"Unknown weight_stat: {weight_stat}")


class ActivationAccumulator:
    def __init__(self, d: int, samplewise: bool, samplewise_max: int):
        self.d = int(d)
        self.n = 0
        self.sum = torch.zeros(self.d, dtype=torch.float64)
        self.sum_abs = torch.zeros(self.d, dtype=torch.float64)
        self.sumsq = torch.zeros(self.d, dtype=torch.float64)
        self.samplewise = bool(samplewise)
        self.samplewise_max = int(samplewise_max)
        self.sample_corrs: List[float] = []

    def update(self, A: torch.Tensor, w_vec: np.ndarray) -> None:
        """
        A: [B, d] reduced activations on CPU/float32 or float64
        """
        if A.ndim != 2 or A.shape[1] != self.d:
            raise ValueError(f"Activation shape mismatch: expected [B,{self.d}], got {tuple(A.shape)}")
        B = int(A.shape[0])
        Af = A.to(dtype=torch.float64)
        self.n += B
        self.sum += Af.sum(dim=0)
        self.sum_abs += Af.abs().sum(dim=0)
        self.sumsq += (Af * Af).sum(dim=0)

        if self.samplewise and len(self.sample_corrs) < self.samplewise_max:
            A_np = Af.to(dtype=torch.float64).cpu().numpy()
            remaining = self.samplewise_max - len(self.sample_corrs)
            take = min(remaining, B)
            for i in range(take):
                self.sample_corrs.append(_pearson_corr(w_vec, A_np[i]))

    def finalize(self) -> Dict[str, np.ndarray]:
        n = max(1, self.n)
        mean = (self.sum / n).cpu().numpy()
        mean_abs = (self.sum_abs / n).cpu().numpy()
        ex2 = (self.sumsq / n).cpu().numpy()
        var = np.maximum(0.0, ex2 - mean * mean)
        std = np.sqrt(var)
        return {"mean": mean, "mean_abs": mean_abs, "std": std}


# -----------------------
# Main
# -----------------------
def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=str, required=True, help="Path to .pth checkpoint (or a directory containing .pth files).")
    p.add_argument("--dataset", type=lambda s: s.strip().upper(), default="CIFAR10",
                   choices=["CIFAR10", "CIFAR100", "MNIST", "FASHIONMNIST"])
    p.add_argument("--data-root", type=str, default="./data")

    p.add_argument("--model", type=str, default=None,
                   help="Model name for architectures.build_model (e.g. resnet20, resnet18, mlp). If omitted, best-effort inferred.")
    p.add_argument("--norm", type=str, default=None, help="bn | ln | flax_ln | none. If omitted, best-effort inferred.")
    p.add_argument("--width-multiplier", type=int, default=None, help="ResNet20 only; inferred if omitted.")
    p.add_argument("--shortcut-option", type=str, default=None, choices=["A", "B", "C"], help="ResNet20 only; inferred if omitted.")
    p.add_argument("--num-classes", type=int, default=None, help="If omitted, inferred from final layer weights.")

    p.add_argument("--layer", type=str, required=True, help="Layer name from model.named_modules() (e.g. layer3.2.conv2, linear, fc2).")
    p.add_argument("--hook", type=str, default="output", choices=["output", "input"],
                   help="Hook output reps (default) or input reps. Input mode correlates against per-input-unit weight stats.")
    p.add_argument("--preprocess", type=str, default="none", choices=["none", "relu", "abs"],
                   help="Apply to activations before stats/correlation.")
    p.add_argument("--weight-stat", type=str, default="l2", choices=["l2", "l1", "absmean"],
                   help="How to reduce weights to one scalar per unit/channel.")
    p.add_argument("--split", type=str, default="test",
                   choices=["train_eval", "test", "val", "subset_A_eval", "subset_B_eval", "full_train"],
                   help="Which dataset split to use. For CIFAR, val/subset_* require --indices-file.")
    p.add_argument("--indices-file", type=str, default=None,
                   help="Optional indices_*.pt produced by your CIFAR training script (needed for val/subset splits).")

    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--max-batches", type=int, default=0, help="<=0 means full split")
    p.add_argument("--list-layers", action="store_true", help="Print weight-bearing layer names and exit.")

    p.add_argument("--samplewise", action="store_true",
                   help="Also compute per-sample correlation across units (one corr per sample).")
    p.add_argument("--samplewise-max-samples", type=int, default=5000)
    p.add_argument("--out", type=str, default="./weight_activation_corr.json")

    args = p.parse_args()

    device = utils.get_device()
    ckpt_path = Path(args.ckpt)

    # Resolve checkpoint(s)
    ckpt_files: List[Path]
    if ckpt_path.is_dir():
        ckpt_files = sorted(ckpt_path.glob("*.pth"))
        if not ckpt_files:
            raise FileNotFoundError(f"No .pth files found in directory: {ckpt_path}")
    else:
        ckpt_files = [ckpt_path]

    results_all: Dict[str, Any] = {
        "dataset": args.dataset,
        "split": args.split,
        "layer": args.layer,
        "hook": args.hook,
        "preprocess": args.preprocess,
        "weight_stat": args.weight_stat,
        "checkpoints": {},
    }

    # Build datasets once
    train_full, eval_full, test_ds = ds_utils.build_datasets(
        args.dataset, root=args.data_root, download=True, augment_train=False, normalize=True
    )

    def get_split_dataset() -> torch.utils.data.Dataset:
        if args.split == "test":
            return test_ds
        if args.split == "full_train":
            return eval_full  # deterministic transform
        if args.split in ("train_eval", "val", "subset_A_eval", "subset_B_eval"):
            if args.dataset not in ("CIFAR10", "CIFAR100"):
                # For MNIST-like, fall back: train_eval == full_train
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

    for ckpt in ckpt_files:
        state = normalize_state_dict_keys(load_ckpt_state_dict(str(ckpt)))

        model_name = args.model or guess_model_name_from_state(state)
        if model_name is None:
            raise ValueError(
                "Could not infer --model from checkpoint. Please pass --model explicitly "
                "(e.g. resnet20, resnet18, mlp)."
            )

        norm = args.norm or infer_norm_from_state(state)
        num_classes = args.num_classes or infer_num_classes_from_state(state) or ds_utils.DATASET_STATS[args.dataset]["num_classes"]

        # Determine model-specific params
        build_kwargs: Dict[str, Any] = {}
        if model_name == "resnet20":
            build_kwargs["width_multiplier"] = args.width_multiplier or infer_resnet20_width_multiplier(state) or 1
            build_kwargs["shortcut_option"] = args.shortcut_option or infer_resnet20_shortcut_option(state)
            build_kwargs["norm"] = norm
        elif model_name in ("resnet18",):
            build_kwargs["norm"] = norm
        elif model_name == "mlp":
            # infer hidden from fc1.weight
            if "fc1.weight" not in state:
                raise KeyError("Expected fc1.weight in MLP checkpoint.")
            hidden = int(state["fc1.weight"].shape[0])
            build_kwargs["hidden"] = hidden
            build_kwargs["input_shape"] = (
                int(ds_utils.DATASET_STATS[args.dataset]["in_channels"]),
                int(ds_utils.DATASET_STATS[args.dataset]["image_size"][0]),
                int(ds_utils.DATASET_STATS[args.dataset]["image_size"][1]),
            )
        # in_channels for conv nets
        in_channels = infer_in_channels_from_state(state) or int(ds_utils.DATASET_STATS[args.dataset]["in_channels"])

        model = architectures.build_model(
            model_name,
            num_classes=int(num_classes),
            in_channels=int(in_channels),
            **build_kwargs,
        ).to(device)

        if args.list_layers:
            print("Weight-bearing layers (name -> weight shape):")
            for name, mod in model.named_modules():
                if hasattr(mod, "weight") and isinstance(getattr(mod, "weight"), torch.Tensor):
                    w = getattr(mod, "weight")
                    if w is not None:
                        print(f"  {name:40s} {tuple(w.shape)}  ({mod.__class__.__name__})")
            return

        # Load only matching keys (robust to extra ckpt entries)
        model_keys = set(model.state_dict().keys())
        filtered = {k: v for k, v in state.items() if k in model_keys}
        missing, unexpected = model.load_state_dict(filtered, strict=False)
        if missing:
            raise KeyError(f"Missing keys when loading state_dict (first 20): {missing[:20]}")
        if unexpected:
            # unexpected can happen if ckpt has extra; filtered should prevent it, but keep check
            print(f"[warn] Unexpected keys (first 20): {unexpected[:20]}")

        # Find layer module
        modules = dict(model.named_modules())
        if args.layer not in modules:
            # helpful suggestions
            candidates = [n for n in modules.keys() if args.layer in n]
            hint = "\n  ".join(candidates[:50]) if candidates else "(no partial matches)"
            raise KeyError(f"Layer '{args.layer}' not found. Partial matches:\n  {hint}")
        layer_mod = modules[args.layer]

        # Compute weight vector (length d)
        w_vec_t = _weight_vector_for_module(layer_mod, mode=args.hook, weight_stat=args.weight_stat)
        w_vec = w_vec_t.cpu().numpy().astype(np.float64)
        d = int(w_vec.shape[0])

        acc = ActivationAccumulator(d=d, samplewise=args.samplewise, samplewise_max=args.samplewise_max_samples)

        # Hook
        def hook_fn(_m: nn.Module, inp: Tuple[torch.Tensor, ...], out: torch.Tensor) -> None:
            A = inp[0] if args.hook == "input" else out
            if not torch.is_tensor(A):
                return
            if args.preprocess == "relu":
                A = torch.relu(A)
            elif args.preprocess == "abs":
                A = A.abs()
            A = _reduce_activation(A)
            A = A.detach().to(device="cpu", dtype=torch.float32)
            acc.update(A, w_vec=w_vec)

        h = layer_mod.register_forward_hook(hook_fn)

        # Forward over data
        model.eval()
        with torch.no_grad():
            for b_ix, (x, _y) in enumerate(loader):
                if args.max_batches and args.max_batches > 0 and b_ix >= args.max_batches:
                    break
                _ = model(x.to(device))

        h.remove()

        stats = acc.finalize()
        mean_abs = stats["mean_abs"]
        std = stats["std"]

        out_entry: Dict[str, Any] = {
            "d": d,
            "n_samples": int(acc.n),
            "weight_vec": {
                "stat": args.weight_stat,
                "mode": args.hook,
                "min": float(np.min(w_vec)),
                "max": float(np.max(w_vec)),
                "mean": float(np.mean(w_vec)),
                "std": float(np.std(w_vec)),
            },
            "activation": {
                "preprocess": args.preprocess,
                "mean_abs": {
                    "min": float(np.min(mean_abs)),
                    "max": float(np.max(mean_abs)),
                    "mean": float(np.mean(mean_abs)),
                    "std": float(np.std(mean_abs)),
                },
                "std": {
                    "min": float(np.min(std)),
                    "max": float(np.max(std)),
                    "mean": float(np.mean(std)),
                    "std": float(np.std(std)),
                },
            },
            "correlation_across_units": {
                "pearson(weight, mean_abs_act)": _pearson_corr(w_vec, mean_abs),
                "spearman(weight, mean_abs_act)": _spearman_corr(w_vec, mean_abs),
                "pearson(weight, std_act)": _pearson_corr(w_vec, std),
                "spearman(weight, std_act)": _spearman_corr(w_vec, std),
            },
        }

        if args.samplewise:
            sc = np.asarray(acc.sample_corrs, dtype=np.float64)
            out_entry["samplewise_corr_across_units"] = {
                "n": int(sc.size),
                "mean": float(np.mean(sc)) if sc.size else float("nan"),
                "median": float(np.median(sc)) if sc.size else float("nan"),
                "std": float(np.std(sc)) if sc.size else float("nan"),
                "min": float(np.min(sc)) if sc.size else float("nan"),
                "max": float(np.max(sc)) if sc.size else float("nan"),
            }

        results_all["checkpoints"][str(ckpt)] = out_entry
        print(f"\n=== {ckpt.name} ===")
        print(json.dumps(out_entry, indent=2))

    # Save aggregated output
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results_all, indent=2), encoding="utf-8")
    print(f"\nSaved: {out_path.resolve()}")


if __name__ == "__main__":
    main()
