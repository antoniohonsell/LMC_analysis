from __future__ import annotations

import argparse
import json
import os
import pickle
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

_HERE = Path(__file__).resolve().parent
_SRC  = _HERE.parent.parent / "src"
for _p in (str(_SRC), str(_HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import architectures  # type: ignore
import datasets  # type: ignore
import utils  # type: ignore
from linear_mode_connectivity.weight_matching_torch import (  # type: ignore
    apply_permutation,
    permutation_spec_from_axes_to_perm,
    resnet20_layernorm_permutation_spec,
    resnet50_permutation_spec,
    weight_matching,
)


def _setup_plotting_style() -> None:
    try:
        utils.apply_stitching_trend_style()  # type: ignore[attr-defined]
    except Exception:
        pass


def _strip_common_prefixes(state: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    prefixes = ("module.", "model.", "net.")
    out = dict(state)
    changed = True
    while changed and out:
        changed = False
        keys = list(out.keys())
        for p in prefixes:
            if all(k.startswith(p) for k in keys):
                out = {k[len(p):]: v for k, v in out.items()}
                changed = True
                break
    return out


def load_ckpt_state_dict(path: str) -> Dict[str, torch.Tensor]:
    obj = torch.load(path, map_location="cpu")
    if isinstance(obj, dict) and "state_dict" in obj:
        sd = obj["state_dict"]
    elif isinstance(obj, dict):
        sd = obj
    else:
        raise ValueError(f"Unrecognized checkpoint format at: {path}")
    return _strip_common_prefixes(sd)


def filter_to_spec_keys(state: Dict[str, torch.Tensor], spec_keys: set[str]) -> Dict[str, torch.Tensor]:
    return {k: v for k, v in state.items() if k in spec_keys}


def interpolate_state_dict(
    a: Dict[str, torch.Tensor],
    b: Dict[str, torch.Tensor],
    lam: float,
) -> Dict[str, torch.Tensor]:
    if a.keys() != b.keys():
        missing_in_b = sorted(set(a.keys()) - set(b.keys()))
        missing_in_a = sorted(set(b.keys()) - set(a.keys()))
        raise KeyError(
            "State dict keysets differ; cannot interpolate safely.\n"
            f"Missing in B: {missing_in_b[:20]}{' ...' if len(missing_in_b) > 20 else ''}\n"
            f"Missing in A: {missing_in_a[:20]}{' ...' if len(missing_in_a) > 20 else ''}"
        )
    out: Dict[str, torch.Tensor] = {}
    for k in a.keys():
        va, vb = a[k], b[k]
        if va.shape != vb.shape:
            raise ValueError(f"Shape mismatch for key {k}: {tuple(va.shape)} vs {tuple(vb.shape)}")
        out[k] = (1.0 - lam) * va + lam * vb if va.dtype.is_floating_point else va
    return out


@torch.no_grad()
def eval_loss_acc(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    max_batches: Optional[int] = None,
) -> Tuple[float, float]:
    model.eval()
    crit = nn.CrossEntropyLoss(reduction="sum")
    total_loss = 0.0
    total_correct = 0
    total_seen = 0
    for bi, (x, y) in enumerate(loader):
        if max_batches is not None and bi >= max_batches:
            break
        x, y = x.to(device), y.to(device)
        logits = model(x)
        total_loss += float(crit(logits, y).item())
        total_correct += int((logits.argmax(dim=1) == y).sum().item())
        total_seen += int(y.numel())
    denom = max(1, total_seen)
    return total_loss / denom, total_correct / denom


class GenericFCMLP(nn.Module):
    def __init__(self, layer_sizes: Sequence[int]):
        super().__init__()
        self.n_layers = len(layer_sizes) - 1
        for i in range(1, self.n_layers + 1):
            setattr(self, f"fc{i}", nn.Linear(layer_sizes[i - 1], layer_sizes[i]))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.view(x.size(0), -1)
        for i in range(1, self.n_layers):
            x = torch.relu(getattr(self, f"fc{i}")(x))
        x = getattr(self, f"fc{self.n_layers}")(x)
        return x


def infer_mlp_fc_layer_numbers(state: Dict[str, torch.Tensor]) -> List[int]:
    pat = re.compile(r"^fc(\d+)\.weight$")
    layers = sorted({int(m.group(1)) for k in state.keys() if (m := pat.match(k))})
    if not layers:
        raise KeyError("No fc{n}.weight keys found in checkpoint state_dict.")
    return layers


def infer_mlp_layer_sizes(state: Dict[str, torch.Tensor]) -> List[int]:
    layer_ids = infer_mlp_fc_layer_numbers(state)
    expected = list(range(1, max(layer_ids) + 1))
    if layer_ids != expected:
        raise ValueError(f"Expected contiguous fc layers {expected}, found {layer_ids}")
    sizes = [int(state["fc1.weight"].shape[1])]
    for i in layer_ids:
        sizes.append(int(state[f"fc{i}.weight"].shape[0]))
    return sizes


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
        axes[f"fc{i}.weight"] = (p_out, prev_p)
        axes[f"fc{i}.bias"] = (p_out,)
        prev_p = p_out

    axes[f"fc{n_layers}.weight"] = (None, prev_p)
    axes[f"fc{n_layers}.bias"] = (None,)
    axes = {k: v for k, v in axes.items() if k in state}
    return permutation_spec_from_axes_to_perm(axes)


def infer_num_classes_from_state(state: Dict[str, torch.Tensor]) -> int:
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
        return int(state[last].shape[0])
    raise KeyError("Could not infer num_classes from checkpoint state_dict")


def infer_in_channels_from_state(state: Dict[str, torch.Tensor]) -> Optional[int]:
    if "conv1.weight" in state and state["conv1.weight"].ndim >= 2:
        return int(state["conv1.weight"].shape[1])
    return None


def infer_resnet20_width_multiplier(state: Dict[str, torch.Tensor]) -> int:
    if "conv1.weight" not in state:
        return 1
    return max(1, int(state["conv1.weight"].shape[0] // 16))


def infer_resnet20_shortcut_option(state: Dict[str, torch.Tensor]) -> str:
    return "C" if any(k.endswith("shortcut.0.weight") for k in state.keys()) else "A"


def infer_norm_from_state(state: Dict[str, torch.Tensor]) -> str:
    keys = state.keys()
    if any(k.endswith("running_mean") or k.endswith("running_var") for k in keys):
        return "bn"
    if any(".ln.weight" in k or ".ln.bias" in k for k in keys):
        return "flax_ln"
    return "ln"


def has_batch_norm(model: nn.Module) -> bool:
    return any(isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d))
               for m in model.modules())


@torch.no_grad()
def reset_bn_stats(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    max_batches: int = 50,
) -> None:
    """
    Recalculate BatchNorm running statistics for an interpolated model.

    After weight interpolation the stored running_mean/running_var are no longer
    representative of the merged model's activations, causing an artificial loss
    barrier.  Running a few forward passes in train mode (no weight update) fixes
    this.  Using momentum=None gives a cumulative moving average, which is more
    accurate than the default exponential average for a small number of batches.
    """
    bn_mods = [m for m in model.modules()
               if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d))]
    if not bn_mods:
        return
    for m in bn_mods:
        m.reset_running_stats()
        m.momentum = None  # cumulative moving average: uses 1/n each step
    model.train()
    for bi, (x, _) in enumerate(loader):
        if bi >= max_batches:
            break
        model(x.to(device))
    model.eval()


def make_loaders(
    dataset_name: str,
    data_root: str,
    batch_size: int,
    num_workers: int,
) -> Tuple[DataLoader, DataLoader]:
    _train_full, eval_full, test_ds = datasets.build_datasets(
        dataset_name,
        root=data_root,
        download=True,
        augment_train=False,
        normalize=True,
    )
    device = utils.get_device()
    train_loader = DataLoader(
        eval_full,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
    )
    return train_loader, test_loader


def plot_interp_loss(title: str, lambdas: Sequence[float], train_naive, test_naive, train_perm, test_perm):
    _setup_plotting_style()
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(lambdas, train_naive, color="grey", linewidth=2, alpha=0.85)
    ax.plot(lambdas, test_naive, color="grey", linewidth=2, linestyle="dashed", alpha=0.85)
    ax.plot(lambdas, train_perm, linewidth=2, marker="^")
    ax.plot(lambdas, test_perm, linewidth=2, linestyle="dashed", marker="^")
    ax.set_xlabel(r"$\lambda$")
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Model $A$", "Model $B$"])
    ax.set_ylabel("Loss")
    ax.set_title(title)
    ax.legend(["Train, naive", "Test, naive", "Train, permuted", "Test, permuted"], loc="best")
    fig.tight_layout()
    return fig


def plot_interp_acc(title: str, lambdas: Sequence[float], train_naive, test_naive, train_perm, test_perm):
    _setup_plotting_style()
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(lambdas, 100.0 * torch.as_tensor(train_naive).cpu().numpy(), color="grey", linewidth=2, alpha=0.85)
    ax.plot(lambdas, 100.0 * torch.as_tensor(test_naive).cpu().numpy(), color="grey", linewidth=2, linestyle="dashed", alpha=0.85)
    ax.plot(lambdas, 100.0 * torch.as_tensor(train_perm).cpu().numpy(), linewidth=2, marker="^")
    ax.plot(lambdas, 100.0 * torch.as_tensor(test_perm).cpu().numpy(), linewidth=2, linestyle="dashed", marker="^")
    ax.set_xlabel(r"$\lambda$")
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Model $A$", "Model $B$"])
    ax.set_ylabel("Accuracy (%)")
    ax.set_title(title)
    ax.legend(["Train, naive", "Test, naive", "Train, permuted", "Test, permuted"], loc="best")
    fig.tight_layout()
    return fig


def _clean_ckpt_label(ckpt_path: str) -> str:
    """
    Extract a short human-readable label from a checkpoint path.
    E.g. 'mlp_FASHIONMNIST_muon_seed0_final.pth' → 'Muon s0'
         'resnet20_CIFAR10_sgd_seed2_final.pth'   → 'SGD s2'
    Falls back to the bare filename stem if parsing fails.
    """
    stem = Path(ckpt_path).stem  # e.g. 'mlp_FASHIONMNIST_muon_seed0_final'
    stem = stem.replace("_final", "").replace("_best", "")
    # Try to find 'seed<N>' and the token before it (optimizer name)
    import re
    m = re.search(r"_([\w]+)_seed(\d+)$", stem)
    if m:
        opt = m.group(1).upper().replace("ADAMW", "AdamW").replace("SGD", "SGD").replace("ADAM", "Adam").replace("MUON", "Muon")
        return f"{opt} s{m.group(2)}"
    return stem


def run_weight_matching_interp(
    *,
    arch: str,
    dataset_name: str,
    ckpt_a: str,
    ckpt_b: str,
    out_dir: str,
    data_root: str = "./data",
    batch_size: int = 256,
    num_workers: int = 0,
    eval_samples: int = 0,
    num_lambdas: int = 25,
    seed: int = 0,
    max_iter: int = 100,
    width_multiplier: Optional[int] = None,
    shortcut_option: Optional[str] = None,
    norm: Optional[str] = None,
    silent: bool = False,
    bn_reset_batches: int = 50,
    plot_title: Optional[str] = None,
) -> Dict[str, Any]:
    arch_l = arch.strip().lower()
    os.makedirs(out_dir, exist_ok=True)
    device = utils.get_device()
    train_loader, test_loader = make_loaders(dataset_name, data_root, batch_size, num_workers)

    state_a_raw = load_ckpt_state_dict(ckpt_a)
    state_b_raw = load_ckpt_state_dict(ckpt_b)

    if arch_l == "mlp":
        layer_sizes = infer_mlp_layer_sizes(state_a_raw)
        model = GenericFCMLP(layer_sizes).to(device)
        model_keys = set(model.state_dict().keys())
        state_a_full = {k: v for k, v in state_a_raw.items() if k in model_keys}
        state_b_full = {k: v for k, v in state_b_raw.items() if k in model_keys}
        ps = mlp_permutation_spec_from_state(state_a_full)
        width_multiplier_out = None
        shortcut_out = None
        norm_out = None
    elif arch_l == "resnet20":
        if width_multiplier is None:
            width_multiplier = infer_resnet20_width_multiplier(state_a_raw)
        if shortcut_option is None:
            shortcut_option = infer_resnet20_shortcut_option(state_a_raw)
        if norm is None:
            norm = infer_norm_from_state(state_a_raw)
        model = architectures.build_model(
            "resnet20",
            num_classes=infer_num_classes_from_state(state_a_raw),
            in_channels=infer_in_channels_from_state(state_a_raw) or int(datasets.DATASET_STATS[dataset_name]["in_channels"]),
            norm=norm,
            width_multiplier=int(width_multiplier),
            shortcut_option=str(shortcut_option),
        ).to(device)
        model_keys = set(model.state_dict().keys())
        state_a_full = {k: v for k, v in state_a_raw.items() if k in model_keys}
        state_b_full = {k: v for k, v in state_b_raw.items() if k in model_keys}
        ps = resnet20_layernorm_permutation_spec(shortcut_option=str(shortcut_option), state_dict=state_a_full)
        width_multiplier_out = int(width_multiplier)
        shortcut_out = str(shortcut_option)
        norm_out = str(norm)
    elif arch_l == "resnet50":
        if norm is None:
            norm = infer_norm_from_state(state_a_raw)
        model = architectures.build_model(
            "resnet50",
            num_classes=infer_num_classes_from_state(state_a_raw),
            in_channels=infer_in_channels_from_state(state_a_raw) or int(datasets.DATASET_STATS[dataset_name]["in_channels"]),
            norm=norm,
        ).to(device)
        model_keys = set(model.state_dict().keys())
        state_a_full = {k: v for k, v in state_a_raw.items() if k in model_keys}
        state_b_full = {k: v for k, v in state_b_raw.items() if k in model_keys}
        ps = resnet50_permutation_spec(state_dict=state_a_full)
        width_multiplier_out = None
        shortcut_out = None
        norm_out = str(norm)
    else:
        raise ValueError(f"Unsupported arch: {arch}")

    spec_keys = set(ps.axes_to_perm.keys())
    state_a_match = filter_to_spec_keys(state_a_full, spec_keys)
    state_b_match = filter_to_spec_keys(state_b_full, spec_keys)

    final_perm = weight_matching(
        seed=int(seed),
        ps=ps,
        params_a=state_a_match,
        params_b=state_b_match,
        max_iter=int(max_iter),
        silent=bool(silent),
    )

    perm_path = os.path.join(out_dir, f"permutation_seed{seed}.pkl")
    with open(perm_path, "wb") as f:
        pickle.dump({k: v.detach().cpu().numpy() for k, v in final_perm.items()}, f)

    state_b_perm_full = apply_permutation(ps, final_perm, state_b_full)
    state_a_dev = {k: v.to(device) for k, v in state_a_full.items()}
    state_b_dev = {k: v.to(device) for k, v in state_b_full.items()}
    state_b_perm_dev = {k: v.to(device) for k, v in state_b_perm_full.items()}

    def compute_max_batches(ds_len: int) -> Optional[int]:
        if eval_samples is None or int(eval_samples) <= 0:
            return None
        return int((int(eval_samples) + int(batch_size) - 1) // int(batch_size))

    max_batches_train = compute_max_batches(len(train_loader.dataset))
    max_batches_test = compute_max_batches(len(test_loader.dataset))
    lambdas = torch.linspace(0.0, 1.0, steps=int(num_lambdas)).tolist()

    train_loss_naive: List[float] = []
    test_loss_naive: List[float] = []
    train_acc_naive: List[float] = []
    test_acc_naive: List[float] = []
    train_loss_perm: List[float] = []
    test_loss_perm: List[float] = []
    train_acc_perm: List[float] = []
    test_acc_perm: List[float] = []

    for lam in lambdas:
        interp_sd = interpolate_state_dict(state_a_dev, state_b_dev, float(lam))
        model.load_state_dict(interp_sd, strict=True)
        if bn_reset_batches > 0 and has_batch_norm(model):
            reset_bn_stats(model, train_loader, device, max_batches=bn_reset_batches)
        tl, ta = eval_loss_acc(model, train_loader, device, max_batches=max_batches_train)
        vl, va = eval_loss_acc(model, test_loader, device, max_batches=max_batches_test)
        train_loss_naive.append(tl)
        train_acc_naive.append(ta)
        test_loss_naive.append(vl)
        test_acc_naive.append(va)

    for lam in lambdas:
        interp_sd = interpolate_state_dict(state_a_dev, state_b_perm_dev, float(lam))
        model.load_state_dict(interp_sd, strict=True)
        if bn_reset_batches > 0 and has_batch_norm(model):
            reset_bn_stats(model, train_loader, device, max_batches=bn_reset_batches)
        tl, ta = eval_loss_acc(model, train_loader, device, max_batches=max_batches_train)
        vl, va = eval_loss_acc(model, test_loader, device, max_batches=max_batches_test)
        train_loss_perm.append(tl)
        train_acc_perm.append(ta)
        test_loss_perm.append(vl)
        test_acc_perm.append(va)

    if plot_title is not None:
        title = plot_title
    else:
        dataset_pretty = dataset_name.replace("FASHIONMNIST", "FashionMNIST").replace("CIFAR10", "CIFAR-10").replace("CIFAR100", "CIFAR-100")
        arch_pretty = arch.upper().replace("RESNET20", "ResNet-20").replace("RESNET50", "ResNet-50")
        label_a = _clean_ckpt_label(ckpt_a)
        label_b = _clean_ckpt_label(ckpt_b)
        title = f"{dataset_pretty} — {arch_pretty}: {label_a} vs {label_b}"
    fig = plot_interp_loss(title, lambdas, train_loss_naive, test_loss_naive, train_loss_perm, test_loss_perm)
    loss_path = os.path.join(out_dir, "interp_loss.png")
    fig.savefig(loss_path, dpi=300)
    plt.close(fig)

    fig = plot_interp_acc(title, lambdas, train_acc_naive, test_acc_naive, train_acc_perm, test_acc_perm)
    acc_path = os.path.join(out_dir, "interp_acc.png")
    fig.savefig(acc_path, dpi=300)
    plt.close(fig)

    results: Dict[str, Any] = {
        "arch": arch_l,
        "dataset": dataset_name,
        "lambdas": lambdas,
        "train_loss_naive": train_loss_naive,
        "test_loss_naive": test_loss_naive,
        "train_acc_naive": train_acc_naive,
        "test_acc_naive": test_acc_naive,
        "train_loss_perm": train_loss_perm,
        "test_loss_perm": test_loss_perm,
        "train_acc_perm": train_acc_perm,
        "test_acc_perm": test_acc_perm,
        "permutation_path": perm_path,
        "loss_plot": loss_path,
        "acc_plot": acc_path,
        "ckpt_a": str(ckpt_a),
        "ckpt_b": str(ckpt_b),
    }
    if width_multiplier_out is not None:
        results["width_multiplier"] = width_multiplier_out
    if shortcut_out is not None:
        results["shortcut_option"] = shortcut_out
    if norm_out is not None:
        results["norm"] = norm_out
    if arch_l == "mlp":
        results["layer_sizes"] = layer_sizes

    torch.save(results, os.path.join(out_dir, "interp_results.pt"))
    with open(os.path.join(out_dir, "interp_results.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    return results


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--arch", type=str, required=True, choices=["mlp", "resnet20", "resnet50"])
    p.add_argument("--dataset", type=str, required=True)
    p.add_argument("--data-root", type=str, default="./data")
    p.add_argument("--ckpt-a", type=str, required=True)
    p.add_argument("--ckpt-b", type=str, required=True)
    p.add_argument("--out-dir", type=str, required=True)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--eval-samples", type=int, default=0)
    p.add_argument("--num-lambdas", type=int, default=25)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--max-iter", type=int, default=100)
    p.add_argument("--width-multiplier", type=int, default=None)
    p.add_argument("--shortcut-option", type=str, default=None, choices=[None, "A", "B", "C"])
    p.add_argument("--norm", type=str, default=None)
    p.add_argument("--silent", action="store_true")
    p.add_argument("--bn-reset-batches", type=int, default=50,
                   help="Batches used to recalculate BN stats after each interpolation. "
                        "0 = disabled (no recalculation).")
    args = p.parse_args()

    results = run_weight_matching_interp(
        arch=args.arch,
        dataset_name=args.dataset,
        ckpt_a=args.ckpt_a,
        ckpt_b=args.ckpt_b,
        out_dir=args.out_dir,
        data_root=args.data_root,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        eval_samples=args.eval_samples,
        num_lambdas=args.num_lambdas,
        seed=args.seed,
        max_iter=args.max_iter,
        width_multiplier=args.width_multiplier,
        shortcut_option=args.shortcut_option,
        norm=args.norm,
        silent=args.silent,
        bn_reset_batches=args.bn_reset_batches,
    )
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
