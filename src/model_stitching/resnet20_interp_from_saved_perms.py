#!/usr/bin/env python3
"""
resnet20_interp_from_saved_perms.py

Compute Linear Mode Connectivity (linear interpolation) curves *after the fact*,
using permutations already produced by activation stitching.

This is the same "interp" block you would get from:
  model_stitching/resnet20_activation_stitching.py --do-interp
but without recomputing permutations/stitching.

Typical usage (single run dir):
  PYTHONPATH="$(pwd)" python model_stitching/resnet20_interp_from_saved_perms.py \
    --out-dir ./activation_stitching_out_cifar10_resnet20_32 \
    --data-root ./data \
    --eval-samples 10000 \
    --num-lambdas 11

If --out-dir does NOT directly contain permutations.json, the script will treat it as a parent
folder and process all immediate subfolders that contain permutations.json.

Outputs (per run dir):
  - interp_acc.png
  - interp_loss.png
  - interp_table.csv
  - results.json / results.pt updated with an "interpolation" field (same schema as original script)
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

_HERE = Path(__file__).resolve().parent
_SRC  = _HERE.parent
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

import architectures
import train_resnet
from linear_mode_connectivity.weight_matching_torch import (
    resnet20_layernorm_permutation_spec, apply_permutation
)

from model_stitching.activation_permutation_stitching import (
    interpolate_state_dict,
    load_ckpt_state_dict,
    normalize_state_dict_keys,
    to_device,
)

TensorDict = Dict[str, torch.Tensor]

# plotting style
import utils
utils.apply_stitching_trend_style()
palette = utils.get_deep_palette()


@torch.no_grad()
def eval_loss_acc(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    max_batches: Optional[int] = None,
) -> Tuple[float, float]:
    model.eval()
    crit = nn.CrossEntropyLoss(reduction="sum")
    loss_sum = 0.0
    correct = 0
    seen = 0
    for b_ix, (x, y) in enumerate(loader):
        if max_batches is not None and b_ix >= max_batches:
            break
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss_sum += float(crit(logits, y).item())
        correct += int((logits.argmax(dim=1) == y).sum().item())
        seen += int(y.numel())
    denom = max(1, seen)
    return loss_sum / denom, correct / denom


def _device_from_utils_fallback() -> torch.device:
    try:
        import utils  # type: ignore
        return utils.get_device()  # type: ignore[attr-defined]
    except Exception:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _save_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, sort_keys=True)


def _plot_interpolation(
    *,
    title: str,
    lambdas: List[float],
    metrics: Dict[str, Dict[str, List[float]]],
    out_dir: Path,
) -> None:
    fig = plt.figure()
    ax = fig.add_subplot(111)
    for split, d in metrics.items():
        ax.plot(lambdas, d["acc_naive"], linestyle="dashed", alpha=0.5, linewidth=2, label=f"{split} naive")
        ax.plot(lambdas, d["acc_perm"], linestyle="solid", linewidth=2, label=f"{split} perm")
    ax.set_xlabel(r"$\lambda$  (0 = A, 1 = B)")
    ax.set_ylabel("accuracy")
    ax.set_title(title)
    ax.legend(framealpha=0.6)
    fig.tight_layout()
    fig.savefig(out_dir / "interp_acc.png", dpi=300)
    plt.close(fig)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    for split, d in metrics.items():
        ax.plot(lambdas, d["loss_naive"], linestyle="dashed", alpha=0.5, linewidth=2, label=f"{split} naive")
        ax.plot(lambdas, d["loss_perm"], linestyle="solid", linewidth=2, label=f"{split} perm")
    ax.set_xlabel(r"$\lambda$  (0 = A, 1 = B)")
    ax.set_ylabel("cross-entropy loss")
    ax.set_title(title)
    ax.legend(framealpha=0.6)
    fig.tight_layout()
    fig.savefig(out_dir / "interp_loss.png", dpi=300)
    plt.close(fig)


def infer_width_multiplier_from_state(state: TensorDict) -> int:
    w = int(state["conv1.weight"].shape[0] // 16)
    if w <= 0:
        raise ValueError("Could not infer width_multiplier from conv1.weight")
    return w


def infer_shortcut_option_from_state(state: TensorDict) -> str:
    return "C" if any(k.endswith("shortcut.0.weight") for k in state.keys()) else "A"


def _resolve_existing_path(p: str, base_dirs: List[Path]) -> Path:
    cand = Path(p)
    if cand.exists():
        return cand
    for base in base_dirs:
        cand2 = base / p
        if cand2.exists():
            return cand2
    raise FileNotFoundError(f"Path not found: {p} (also tried relative to {base_dirs})")


def _load_permutations(run_dir: Path) -> Dict[str, torch.Tensor]:
    pj = run_dir / "permutations.json"
    ppt = run_dir / "permutations.pt"
    if pj.exists():
        obj = json.loads(pj.read_text(encoding="utf-8"))
        return {k: torch.tensor(v, dtype=torch.long) for k, v in obj.items()}
    if ppt.exists():
        obj = torch.load(ppt, map_location="cpu")
        return {k: v.to(dtype=torch.long) for k, v in obj.items()}
    raise FileNotFoundError(f"Could not find permutations.json or permutations.pt in {run_dir}")


def _build_eval_loaders(
    *,
    dataset: str,
    data_root: str,
    indices_file: Path,
    batch_size: int,
    num_workers: int,
    device: torch.device,
    eval_samples: int,
) -> Tuple[Dict[str, DataLoader], Dict[str, Optional[int]]]:
    idx_obj = torch.load(indices_file, map_location="cpu")
    train_indices = idx_obj["train_indices"]
    val_indices = idx_obj["val_indices"]
    subset_a_idx = idx_obj["subset_a_indices"]
    subset_b_idx = idx_obj["subset_b_indices"]

    _train_full, eval_full, test_ds, _targets = train_resnet.load_cifar_datasets(dataset, data_root)

    train_eval = Subset(eval_full, train_indices)
    subset_a_eval = Subset(eval_full, subset_a_idx)
    subset_b_eval = Subset(eval_full, subset_b_idx)
    val_ds = Subset(eval_full, val_indices)

    loaders: Dict[str, DataLoader] = {
        #"subset_A_eval": DataLoader(subset_a_eval, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=(device.type == "cuda")),
        #"subset_B_eval": DataLoader(subset_b_eval, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=(device.type == "cuda")),
        #"val": DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=(device.type == "cuda")),
        "test": DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=False),
        "train_eval": DataLoader(train_eval, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=False),
    }

    def max_batches_for(ds_len: int) -> Optional[int]:
        if eval_samples is None or eval_samples <= 0:
            return None
        return int((eval_samples + batch_size - 1) // batch_size)

    max_batches = {k: max_batches_for(len(dl.dataset)) for k, dl in loaders.items()}
    return loaders, max_batches


def _run_one_dir(
    run_dir: Path,
    *,
    data_root: str,
    num_lambdas: int,
    batch_size: int,
    num_workers: int,
    eval_samples: int,
    overwrite: bool,
) -> None:
    run_dir = run_dir.resolve()
    run_dir.mkdir(parents=True, exist_ok=True)

    results_json = run_dir / "results.json"
    results_pt = run_dir / "results.pt"

    results: Dict[str, Any] = {}
    if results_json.exists():
        results = json.loads(results_json.read_text(encoding="utf-8"))

    if (not overwrite) and isinstance(results.get("interpolation", None), dict):
        print(f"[skip] {run_dir} already has interpolation in results.json (use --overwrite to recompute)")
        return

    dataset = str(results.get("dataset", "CIFAR10"))
    ckpt_a_s = results.get("ckpt_a", None)
    ckpt_b_s = results.get("ckpt_b", None)
    if ckpt_a_s is None or ckpt_b_s is None:
        raise ValueError(f"{run_dir} has no ckpt_a/ckpt_b in results.json.")

    idx_s = results.get("indices_file", None)
    if idx_s is None:
        raise ValueError(f"{run_dir} has no indices_file in results.json.")

    device = _device_from_utils_fallback()

    ckpt_a = _resolve_existing_path(str(ckpt_a_s), [run_dir, Path.cwd()])
    ckpt_b = _resolve_existing_path(str(ckpt_b_s), [run_dir, Path.cwd()])
    idx_file = _resolve_existing_path(str(idx_s), [run_dir, Path.cwd()])

    perm = _load_permutations(run_dir)

    state_a_full = normalize_state_dict_keys(load_ckpt_state_dict(str(ckpt_a)))
    state_b_full = normalize_state_dict_keys(load_ckpt_state_dict(str(ckpt_b)))

    width_multiplier = int(results.get("width_multiplier", 0)) or infer_width_multiplier_from_state(state_a_full)
    shortcut_option = str(results.get("shortcut_option", "")) or infer_shortcut_option_from_state(state_a_full)

    num_classes = int(train_resnet.DATASET_STATS[dataset]["num_classes"])

    model_tmp = architectures.build_model(
        "resnet20",
        num_classes=num_classes,
        norm="flax_ln",
        width_multiplier=width_multiplier,
        shortcut_option=shortcut_option,
    )
    keys = set(model_tmp.state_dict().keys())
    del model_tmp

    state_a: TensorDict = {k: v for k, v in state_a_full.items() if k in keys}
    state_b: TensorDict = {k: v for k, v in state_b_full.items() if k in keys}

    ps = resnet20_layernorm_permutation_spec(shortcut_option=shortcut_option, state_dict=state_a)

    missing = sorted(set(ps.perm_to_axes.keys()) - set(perm.keys()))
    if missing:
        raise KeyError(f"{run_dir}: permutations.json is missing keys required by the permutation spec: {missing}.")

    state_b_perm = apply_permutation(ps, perm, state_b)

    a_dev = to_device(state_a, device)
    b_dev = to_device(state_b, device)
    bperm_dev = to_device(state_b_perm, device)

    loaders, max_batches = _build_eval_loaders(
        dataset=dataset,
        data_root=data_root,
        indices_file=idx_file,
        batch_size=batch_size,
        num_workers=num_workers,
        device=device,
        eval_samples=eval_samples,
    )

    model_eval = architectures.build_model(
        "resnet20",
        num_classes=num_classes,
        norm="flax_ln",
        width_multiplier=width_multiplier,
        shortcut_option=shortcut_option,
    ).to(device)

    lambdas = torch.linspace(0.0, 1.0, steps=int(num_lambdas)).tolist()
    interp_metrics = {split: {"acc_naive": [], "acc_perm": [], "loss_naive": [], "loss_perm": []} for split in loaders.keys()}

    for lam in lambdas:
        sd = interpolate_state_dict(a_dev, b_dev, float(lam))
        model_eval.load_state_dict(sd, strict=True)
        for split, loader in loaders.items():
            loss, acc = eval_loss_acc(model_eval, loader, device, max_batches=max_batches[split])
            interp_metrics[split]["loss_naive"].append(loss)
            interp_metrics[split]["acc_naive"].append(acc)

        sd = interpolate_state_dict(a_dev, bperm_dev, float(lam))
        model_eval.load_state_dict(sd, strict=True)
        for split, loader in loaders.items():
            loss, acc = eval_loss_acc(model_eval, loader, device, max_batches=max_batches[split])
            interp_metrics[split]["loss_perm"].append(loss)
            interp_metrics[split]["acc_perm"].append(acc)

    title = (
        f"{dataset} ResNet20-LN interpolation from saved perms "
        f"(width={width_multiplier}, shortcut={shortcut_option})"
    )

    _plot_interpolation(title=title, lambdas=lambdas, metrics=interp_metrics, out_dir=run_dir)

    interp_csv = run_dir / "interp_table.csv"
    with interp_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["split", "lambda", "acc_naive", "acc_perm", "loss_naive", "loss_perm"])
        for split, d in interp_metrics.items():
            for i, lam in enumerate(lambdas):
                w.writerow([split, lam, d["acc_naive"][i], d["acc_perm"][i], d["loss_naive"][i], d["loss_perm"][i]])

    interp = {
        "lambdas": lambdas,
        "metrics": interp_metrics,
        "plots": {"acc": str(run_dir / "interp_acc.png"), "loss": str(run_dir / "interp_loss.png")},
        "csv": str(interp_csv),
    }
    results["interpolation"] = interp
    results.setdefault("dataset", dataset)
    results.setdefault("ckpt_a", str(ckpt_a))
    results.setdefault("ckpt_b", str(ckpt_b))
    results.setdefault("indices_file", str(idx_file))
    results.setdefault("width_multiplier", width_multiplier)
    results.setdefault("shortcut_option", shortcut_option)

    _save_json(results_json, results)
    if results_pt.exists():
        try:
            pt_obj = torch.load(results_pt, map_location="cpu")
            if isinstance(pt_obj, dict):
                pt_obj["interpolation"] = interp
                torch.save(pt_obj, results_pt)
        except Exception:
            pass

    print(f"[ok] {run_dir}")
    print(f"     wrote: {run_dir/'interp_acc.png'}")
    print(f"            {run_dir/'interp_loss.png'}")
    print(f"            {interp_csv}")


def _find_run_dirs(root: Path) -> List[Path]:
    root = root.expanduser()
    if (root / "permutations.json").exists() or (root / "permutations.pt").exists():
        return [root]
    out: List[Path] = []
    if root.exists() and root.is_dir():
        for ch in sorted(root.iterdir()):
            if ch.is_dir() and ((ch / "permutations.json").exists() or (ch / "permutations.pt").exists()):
                out.append(ch)
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--out-dir", type=str, required=True)
    p.add_argument("--data-root", type=str, default="./data")
    p.add_argument("--num-lambdas", type=int, default=25)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--eval-samples", type=int, default=0, help="<=0 means full split.")
    p.add_argument("--overwrite", action="store_true")
    args = p.parse_args()

    root = Path(args.out_dir)
    run_dirs = _find_run_dirs(root)
    if not run_dirs:
        raise FileNotFoundError(f"No run directories found under {root}.")

    for rd in run_dirs:
        _run_one_dir(
            rd,
            data_root=args.data_root,
            num_lambdas=int(args.num_lambdas),
            batch_size=int(args.batch_size),
            num_workers=int(args.num_workers),
            eval_samples=int(args.eval_samples),
            overwrite=bool(args.overwrite),
        )


if __name__ == "__main__":
    main()