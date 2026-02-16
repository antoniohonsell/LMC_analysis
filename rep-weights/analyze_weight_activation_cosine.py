#!/usr/bin/env python3
"""
analyze_weight_activation_cosine.py

Compute cosine similarity in "channel/unit space" between:
  w      = per-unit/channel weight statistic vector (e.g., filter L2 norms), shape [d]
  a(x)   = per-image activation vector at a chosen layer (spatial-mean per channel), shape [d]

Optional: z-score across channels per image before cosine (reduces dominance of pure scale effects):
  a_z(x) = (a(x) - mean_channels(a(x))) / std_channels(a(x))

Optional: z-score w across channels too:
  w_z    = (w - mean(w)) / std(w)

Example (vanilla, like before):
  PYTHONPATH=. python analyze_weight_activation_cosine.py \
    --ckpt ./runs_resnet20_ln_warmcos/CIFAR10/full/seed_0/resnet20_CIFAR10_full_seed0_best.pth \
    --dataset CIFAR10 --model resnet20 \
    --layer layer3.2.conv2 --hook output --split test \
    --preprocess abs --samplewise

Example (z-score activations per image before cosine):
  PYTHONPATH=. python analyze_weight_activation_cosine.py \
    --ckpt ./runs_resnet20_ln_warmcos/CIFAR10/full/seed_0/resnet20_CIFAR10_full_seed0_best.pth \
    --dataset CIFAR10 --model resnet20 \
    --layer layer3.2.conv2 --hook output --split test \
    --preprocess abs --samplewise \
    --act-channel-norm zscore

Example (z-score both activations and w):
  ... --act-channel-norm zscore --w-norm zscore
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
# Math helpers
# -----------------------
def _cosine_sim_np(x: np.ndarray, y: np.ndarray, eps: float = 1e-12) -> float:
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    if x.size != y.size or x.size == 0:
        return float("nan")
    nx = np.linalg.norm(x)
    ny = np.linalg.norm(y)
    denom = max(eps, nx * ny)
    return float(np.dot(x, y) / denom)


def _reduce_activation(act: torch.Tensor) -> torch.Tensor:
    # [B, C, H, W] -> [B, C] by spatial mean; keep [B, D] as-is.
    if act.ndim <= 2:
        return act
    dims = tuple(range(2, act.ndim))
    return act.mean(dim=dims)


def _zscore_channels(A: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """
    Z-score across channels per sample.
    A: [B, d] -> [B, d]
    """
    mu = A.mean(dim=1, keepdim=True)
    sigma = A.std(dim=1, keepdim=True, unbiased=False)
    return (A - mu) / (sigma + eps)


def _zscore_vector(v: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """
    Z-score a 1D vector across its entries.
    v: [d] -> [d]
    """
    mu = v.mean()
    sigma = v.std(unbiased=False)
    return (v - mu) / (sigma + eps)


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
        v = W.abs() if weight_stat in ("l1", "absmean") else W
        return v.clone()

    if isinstance(mod, nn.Linear):
        # W: [out, in]
        X = W if mode == "output" else W.transpose(0, 1)
        X = X.reshape(X.shape[0], -1)
    elif isinstance(mod, nn.Conv2d):
        # W: [out, in, kH, kW]
        if mode == "output":
            X = W.reshape(W.shape[0], -1)
        elif mode == "input":
            X = W.permute(1, 0, 2, 3).contiguous().reshape(W.shape[1], -1)
        else:
            raise ValueError(f"Unknown mode: {mode}")
    else:
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


def _cosine_sim_batch_torch(w: torch.Tensor, A: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """
    w: [d]
    A: [B, d]
    returns: [B]
    """
    w = w.to(dtype=torch.float64)
    A = A.to(dtype=torch.float64)
    w_norm = torch.linalg.vector_norm(w, ord=2)
    a_norm = torch.linalg.vector_norm(A, ord=2, dim=1)
    denom = torch.clamp(w_norm * a_norm, min=eps)
    return (A @ w) / denom


class ActivationAccumulator:
    """
    Accumulates dataset-level stats AND samplewise cosine similarity between w and a(x).
    """
    def __init__(
        self,
        d: int,
        w_vec: torch.Tensor,
        samplewise: bool,
        samplewise_max: int,
        act_channel_norm: str = "none",
        w_norm: str = "none",
    ):
        self.d = int(d)
        self.n = 0
        self.sum = torch.zeros(self.d, dtype=torch.float64)
        self.sum_abs = torch.zeros(self.d, dtype=torch.float64)
        self.sumsq = torch.zeros(self.d, dtype=torch.float64)

        self.samplewise = bool(samplewise)
        self.samplewise_max = int(samplewise_max)
        self.sample_cos: List[float] = []

        self.act_channel_norm = act_channel_norm
        self.w_norm = w_norm

        w = w_vec.detach().cpu().to(dtype=torch.float64)
        if self.w_norm == "zscore":
            w = _zscore_vector(w)
        self.w_for_cos = w

    def update(self, A: torch.Tensor) -> None:
        """
        A: [B, d] activations on CPU (float32/float64).
        """
        if A.ndim != 2 or A.shape[1] != self.d:
            raise ValueError(f"Activation shape mismatch: expected [B,{self.d}], got {tuple(A.shape)}")
        B = int(A.shape[0])
        Af = A.to(dtype=torch.float64)

        # Dataset-level stats (always on raw Af after your chosen preprocess/reduce)
        self.n += B
        self.sum += Af.sum(dim=0)
        self.sum_abs += Af.abs().sum(dim=0)
        self.sumsq += (Af * Af).sum(dim=0)

        # Samplewise cosine
        if self.samplewise and len(self.sample_cos) < self.samplewise_max:
            remaining = self.samplewise_max - len(self.sample_cos)
            take = min(remaining, B)

            A_for_cos = Af[:take]
            if self.act_channel_norm == "zscore":
                A_for_cos = _zscore_channels(A_for_cos)

            cos_vals = _cosine_sim_batch_torch(self.w_for_cos, A_for_cos).cpu().numpy().astype(np.float64)
            self.sample_cos.extend([float(x) for x in cos_vals])

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
    p.add_argument("--ckpt", type=str, required=True, help="Path to .pth checkpoint (or directory containing .pth files).")
    p.add_argument("--dataset", type=lambda s: s.strip().upper(), default="CIFAR10",
                   choices=["CIFAR10", "CIFAR100", "MNIST", "FASHIONMNIST"])
    p.add_argument("--data-root", type=str, default="./data")

    p.add_argument("--model", type=str, default=None,
                   help="Model name for architectures.build_model (e.g. resnet20, resnet18, mlp). If omitted, inferred.")
    p.add_argument("--norm", type=str, default=None, help="bn | ln | flax_ln | none. If omitted, inferred.")
    p.add_argument("--width-multiplier", type=int, default=None, help="ResNet20 only; inferred if omitted.")
    p.add_argument("--shortcut-option", type=str, default=None, choices=["A", "B", "C"], help="ResNet20 only; inferred if omitted.")
    p.add_argument("--num-classes", type=int, default=None, help="If omitted, inferred from final layer weights.")

    p.add_argument("--layer", type=str, required=True, help="Layer name from model.named_modules()")
    p.add_argument("--hook", type=str, default="output", choices=["output", "input"],
                   help="Use output reps (default) or input reps (input correlates against per-input-unit weight stats).")
    p.add_argument("--preprocess", type=str, default="none", choices=["none", "relu", "abs"],
                   help="Apply to activations before cosine/statistics.")
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
                   help="Compute per-image cosine similarity cos(w, a(x)) and summarize.")
    p.add_argument("--samplewise-max-samples", type=int, default=5000)
    p.add_argument("--out", type=str, default="./weight_activation_cosine.json")

    # NEW: normalization controls
    p.add_argument("--act-channel-norm", type=str, default="none", choices=["none", "zscore"],
                   help="Normalize activations across channels per image before cosine.")
    p.add_argument("--w-norm", type=str, default="none", choices=["none", "zscore"],
                   help="Normalize weight vector across channels before cosine.")

    args = p.parse_args()

    device = utils.get_device()
    ckpt_path = Path(args.ckpt)

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
        "act_channel_norm": args.act_channel_norm,
        "w_norm": args.w_norm,
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

        if args.list_layers:
            print("Weight-bearing layers (name -> weight shape):")
            for name, mod in model.named_modules():
                if hasattr(mod, "weight") and isinstance(getattr(mod, "weight"), torch.Tensor):
                    w = getattr(mod, "weight")
                    if w is not None:
                        print(f"  {name:40s} {tuple(w.shape)}  ({mod.__class__.__name__})")
            return

        model_keys = set(model.state_dict().keys())
        filtered = {k: v for k, v in state.items() if k in model_keys}
        missing, unexpected = model.load_state_dict(filtered, strict=False)
        if missing:
            raise KeyError(f"Missing keys when loading state_dict (first 20): {missing[:20]}")
        if unexpected:
            print(f"[warn] Unexpected keys (first 20): {unexpected[:20]}")

        modules = dict(model.named_modules())
        if args.layer not in modules:
            candidates = [n for n in modules.keys() if args.layer in n]
            hint = "\n  ".join(candidates[:50]) if candidates else "(no partial matches)"
            raise KeyError(f"Layer '{args.layer}' not found. Partial matches:\n  {hint}")
        layer_mod = modules[args.layer]

        # Weight vector w (length d)
        w_vec_t = _weight_vector_for_module(layer_mod, mode=args.hook, weight_stat=args.weight_stat)
        w_vec_t = w_vec_t.detach().cpu().to(dtype=torch.float64)
        w_vec = w_vec_t.numpy().astype(np.float64)
        d = int(w_vec.shape[0])

        acc = ActivationAccumulator(
            d=d,
            w_vec=w_vec_t,
            samplewise=args.samplewise,
            samplewise_max=args.samplewise_max_samples,
            act_channel_norm=args.act_channel_norm,
            w_norm=args.w_norm,
        )

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
            acc.update(A)

        h = layer_mod.register_forward_hook(hook_fn)

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
            # Cosine in the same "channel space" (dataset-level vectors)
            "cosine_across_units": {
                "cos(w, mean_abs_act)": _cosine_sim_np(w_vec, mean_abs),
                "cos(w, std_act)": _cosine_sim_np(w_vec, std),
            },
        }

        if args.samplewise:
            sc = np.asarray(acc.sample_cos, dtype=np.float64)
            out_entry["samplewise_cosine_similarity"] = {
                "n": int(sc.size),
                "mean": float(np.mean(sc)) if sc.size else float("nan"),
                "median": float(np.median(sc)) if sc.size else float("nan"),
                "std": float(np.std(sc)) if sc.size else float("nan"),
                "min": float(np.min(sc)) if sc.size else float("nan"),
                "max": float(np.max(sc)) if sc.size else float("nan"),
                "act_channel_norm": args.act_channel_norm,
                "w_norm": args.w_norm,
            }

        results_all["checkpoints"][str(ckpt)] = out_entry
        print(f"\n=== {ckpt.name} ===")
        print(json.dumps(out_entry, indent=2))

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results_all, indent=2), encoding="utf-8")
    print(f"\nSaved: {out_path.resolve()}")


if __name__ == "__main__":
    main()
