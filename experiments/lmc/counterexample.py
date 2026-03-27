#!/usr/bin/env python3


from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


# -----------------------------
# Device helpers (MPS default)
# -----------------------------
def get_default_device_mps_first() -> torch.device:
    if torch.backends.mps.is_available() and torch.backends.mps.is_built():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def parse_device(s: str) -> torch.device:
    s = s.strip().lower()
    if s in ("auto", "default"):
        return get_default_device_mps_first()
    if s in ("mps", "cuda", "cpu"):
        return torch.device(s)
    raise ValueError(f"Unknown device '{s}'. Use one of: auto|mps|cuda|cpu")


def seed_torch_for_init(seed: int) -> None:
    # Seeds CPU RNG; CUDA is additionally seeded if present.
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# -----------------------------
# Data + model
# -----------------------------
def make_strict_dataset_cpu(n: int, seed: int = 0) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Deterministic dataset generation on CPU only (MPS-safe).

    Returns CPU tensors (x, y). Move to device later.
    """
    g = torch.Generator(device="cpu")
    g.manual_seed(seed)

    x = (2.0 * torch.rand((n, 2), generator=g, device="cpu")) - 1.0
    y = ((x[:, 0] < 0.0) & (x[:, 1] > 0.0)).float().unsqueeze(1)
    return x, y


class MLP2x2(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(2, 2, bias=True)
        self.fc2 = nn.Linear(2, 2, bias=True)
        self.fc3 = nn.Linear(2, 1, bias=False)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)  # logits


def set_manual_params(model: nn.Module, params: Dict[str, Any]) -> None:
    with torch.no_grad():
        for name, tensor in model.named_parameters():
            if name not in params:
                raise KeyError(f"Missing param: {name}")
            src = torch.tensor(params[name], dtype=tensor.dtype, device=tensor.device)
            if src.shape != tensor.shape:
                raise ValueError(f"Shape mismatch for {name}: got {src.shape}, expected {tensor.shape}")
            tensor.copy_(src)


@torch.no_grad()
def accuracy_on(model: nn.Module, x: torch.Tensor, y: torch.Tensor) -> float:
    model.eval()
    probs = torch.sigmoid(model(x))
    preds = (probs >= 0.5).float()
    return (preds.eq(y)).float().mean().item()


# -----------------------------------------
# Training with explicit SGD "randomness"
# -----------------------------------------
def train_sgd_seeded(
    *,
    n_samples: int = 8192,
    data_seed: int = 0,
    val_frac: float = 0.2,
    device: torch.device,
    epochs: int = 300,
    optimizer : str = "SGD",
    lr: float = 0.1,
    adam_betas=(0.9, 0.999), 
    adam_eps= 1e-8, 
    weight_decay=0.0,
    batch_size: int = 256,
    sgd_seed: int = 0,
    strict_init_params: Optional[Dict[str, Any]] = None,
) -> Tuple[nn.Module, Dict[str, List[float]]]:
    # Fix dataset on CPU (MPS-safe deterministic RNG)
    x_cpu, y_cpu = make_strict_dataset_cpu(n_samples, seed=data_seed)

    # Train/val split (deterministic)
    n_val = int(math.floor(n_samples * val_frac))
    x_val_cpu, y_val_cpu = x_cpu[:n_val], y_cpu[:n_val]
    x_tr_cpu, y_tr_cpu = x_cpu[n_val:], y_cpu[n_val:]

    x_tr = x_tr_cpu.to(device)
    y_tr = y_tr_cpu.to(device)

    # DataLoader shuffle generator MUST be CPU
    g = torch.Generator(device="cpu")
    g.manual_seed(sgd_seed)


    # loader = DataLoader(
    #     TensorDataset(x_tr_cpu, y_tr_cpu),
    #     batch_size=batch_size,
    #     shuffle=True,
    #     generator=g,
    #     drop_last=False,
    #     num_workers=0,
    #     pin_memory=(device.type == "cuda"),
    # )

    # Seed parameter init (and any torch randomness)
    seed_torch_for_init(sgd_seed)

    model = MLP2x2().to(device)
    if strict_init_params is not None:
        set_manual_params(model, strict_init_params)

    if optimizer=="SGD":
        opt = torch.optim.SGD(model.parameters(), lr=lr)
        
    elif optimizer == "Adam":
        opt = torch.optim.Adam(
            model.parameters(),
            lr=lr,
            betas=adam_betas,
            eps=adam_eps,
            weight_decay=weight_decay,
        )

    elif optimizer == "AdamW":
        opt = torch.optim.AdamW(
            model.parameters(),
            lr=lr,
            betas=adam_betas,
            eps=adam_eps,
            weight_decay=weight_decay,
        )
    loss_fn = nn.BCEWithLogitsLoss()

    # Keep val tensors on device for fast evaluation
    x_val = x_val_cpu.to(device)
    y_val = y_val_cpu.to(device)

    hist = {"train_loss": [], "val_acc": []}

    for _ in range(epochs):
        model.train()
        total_loss, total = 0.0, 0
        perm = torch.randperm(x_tr.size(0), generator=g, device="cpu").to(device) 
        for i in range(0, x_tr.size(0), batch_size):
            idx = perm[i:i+batch_size]
            xb = x_tr[idx]
            yb = y_tr[idx]

            opt.zero_grad(set_to_none=True)
            logits = model(xb)
            loss = loss_fn(logits, yb)
            loss.backward()
            opt.step()

            bs = xb.size(0)
            total_loss += loss.item() * bs
            total += bs

        hist["train_loss"].append(total_loss / max(1, total))
        hist["val_acc"].append(accuracy_on(model, x_val, y_val))

    return model, hist


@dataclass
class TuneBest:
    lr: float
    final_val_acc: float
    peak_val_acc: float
    final_train_loss: float
    sgd_seed: int
    data_seed: int


def tune_learning_rate_for_seed(
    lr_grid: Iterable[float],
    *,
    epochs: int,
    batch_size: int,
    n_samples: int,
    data_seed: int,
    val_frac: float,
    optimizer: str,
    sgd_seed: int,
    device: torch.device,
    strict_init_params: Optional[Dict[str, Any]] = None,
    criterion: str = "best_val_acc",  # or "best_val_acc_peak"
) -> Tuple[TuneBest, List[Dict[str, float]], nn.Module]:
    results: List[Dict[str, float]] = []
    best_score: Optional[float] = None
    best_model: Optional[nn.Module] = None
    best_best: Optional[TuneBest] = None

    for lr in lr_grid:
        model, hist = train_sgd_seeded(
            n_samples=n_samples,
            data_seed=data_seed,
            val_frac=val_frac,
            device=device,
            epochs=epochs,
            lr=float(lr),
            batch_size=batch_size,
            optimizer=optimizer,
            sgd_seed=sgd_seed,
            strict_init_params=strict_init_params,
        )

        final_val = float(hist["val_acc"][-1])
        peak_val = float(max(hist["val_acc"]))
        final_loss = float(hist["train_loss"][-1])

        if criterion == "best_val_acc":
            score = final_val
        elif criterion == "best_val_acc_peak":
            score = peak_val
        else:
            raise ValueError(f"Unknown criterion: {criterion}")

        row = {
            "lr": float(lr),
            "final_val_acc": final_val,
            "peak_val_acc": peak_val,
            "final_train_loss": final_loss,
            "sgd_seed": float(sgd_seed),
            "data_seed": float(data_seed),
        }
        results.append(row)

        print(
            f"[sgd_seed={sgd_seed}] lr={lr:.6g} | final_val_acc={final_val:.4f} "
            f"| peak_val_acc={peak_val:.4f} | final_loss={final_loss:.6f}"
        )

        if best_score is None or score > best_score:
            best_score = score
            best_model = model
            best_best = TuneBest(
                lr=float(lr),
                final_val_acc=final_val,
                peak_val_acc=peak_val,
                final_train_loss=final_loss,
                sgd_seed=int(sgd_seed),
                data_seed=int(data_seed),
            )

    assert best_model is not None and best_best is not None
    results_sorted = sorted(results, key=lambda d: d["final_val_acc"], reverse=True)
    return best_best, results_sorted, best_model


def train_two_disjoint_models_with_lr_tuning(
    lr_grid: Iterable[float],
    *,
    sgd_seed_a: int,
    sgd_seed_b: int,
    data_seed: int,
    n_samples: int,
    val_frac: float,
    optimizer: str,
    epochs: int,
    batch_size: int,
    device: torch.device,
    strict_init_params: Optional[Dict[str, Any]] = None,
    criterion: str = "best_val_acc",
) -> Tuple[Tuple[nn.Module, TuneBest, List[Dict[str, float]]], Tuple[nn.Module, TuneBest, List[Dict[str, float]]]]:
    best_a, res_a, model_a = tune_learning_rate_for_seed(
        lr_grid,
        epochs=epochs,
        batch_size=batch_size,
        n_samples=n_samples,
        data_seed=data_seed,
        val_frac=val_frac,
        sgd_seed=sgd_seed_a,
        device=device,
        optimizer=optimizer,
        strict_init_params=strict_init_params,
        criterion=criterion,
    )

    best_b, res_b, model_b = tune_learning_rate_for_seed(
        lr_grid,
        epochs=epochs,
        batch_size=batch_size,
        n_samples=n_samples,
        data_seed=data_seed,
        val_frac=val_frac,
        sgd_seed=sgd_seed_b,
        optimizer=optimizer,
        device=device,
        strict_init_params=strict_init_params,
        criterion=criterion,
    )

    return (model_a, best_a, res_a), (model_b, best_b, res_b)


# -----------------------------
# Table export (CSV)
# -----------------------------
def tensor_to_json_cell(t: torch.Tensor) -> str:
    return json.dumps(t.detach().cpu().tolist())


def model_to_table_row(
    *,
    trial_index: int,
    label: str,
    best: TuneBest,
    model: nn.Module,
) -> Dict[str, Any]:
    row: Dict[str, Any] = {
        "trial": int(trial_index),
        "label": str(label),
        "sgd_seed": int(best.sgd_seed),
        "best_lr": float(best.lr),
        "final_val_acc": float(best.final_val_acc),
        "peak_val_acc": float(best.peak_val_acc),
        "final_train_loss": float(best.final_train_loss),
    }
    for name, p in model.named_parameters():
        row[name] = tensor_to_json_cell(p)
    return row


def write_csv_table(path: Path, rows: List[Dict[str, Any]], param_names: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    base_cols = ["trial", "label", "sgd_seed", "best_lr", "final_val_acc", "peak_val_acc", "final_train_loss"]
    fieldnames = base_cols + param_names

    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fieldnames})


# -----------------------------
# Main
# -----------------------------
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="mps", help="auto|mps|cuda|cpu (auto = MPS-first)")
    parser.add_argument("--num_trials", type=int, default=20)
    parser.add_argument("--seed_start", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--n_samples", type=int, default=8192)
    parser.add_argument("--val_frac", type=float, default=0.2)
    parser.add_argument("--data_seed", type=int, default=0)
    parser.add_argument(
        "--criterion",
        type=str,
        default="best_val_acc",
        choices=["best_val_acc", "best_val_acc_peak"],
    )
    parser.add_argument(
        "--lr_grid",
        type=float,
        nargs="+",
        default= [1e-4, 3e-4, 1e-3, 3e-3, 1e-2],
    )
    parser.add_argument("--out_csv", type=str, default="counterexample_params_table_Adam.csv")
    parser.add_argument("--optimizer", type=str, default="SGD", choices=["SGD","Adam","AdamW"])
    args = parser.parse_args()

    device = parse_device(args.device)
    print(f"Using device: {device}")

    # Stable parameter column ordering from a fresh model
    template = MLP2x2()
    param_names = [name for name, _ in template.named_parameters()]

    rows: List[Dict[str, Any]] = []

    # 20 trials; each trial trains TWO networks with distinct SGD seeds
    for t in range(int(args.num_trials)):
        base = int(args.seed_start) + t
        seed_a = 2 * base
        seed_b = 2 * base + 1

        print(f"\n=== Trial {t}/{args.num_trials - 1} | seeds: A={seed_a}, B={seed_b} ===")

        (model_a, best_a, _res_a), (model_b, best_b, _res_b) = train_two_disjoint_models_with_lr_tuning(
            args.lr_grid,
            sgd_seed_a=seed_a,
            sgd_seed_b=seed_b,
            data_seed=int(args.data_seed),
            n_samples=int(args.n_samples),
            val_frac=float(args.val_frac),
            epochs=int(args.epochs),
            optimizer=str(args.optimizer),
            batch_size=int(args.batch_size),
            device=device,
            strict_init_params=None,
            criterion=str(args.criterion),
        )

        rows.append(model_to_table_row(trial_index=t, label="A", best=best_a, model=model_a))
        rows.append(model_to_table_row(trial_index=t, label="B", best=best_b, model=model_b))

    out_csv = Path(args.out_csv)
    write_csv_table(out_csv, rows, param_names)
    print(f"\nSaved table with {len(rows)} networks to: {out_csv.resolve()}")


if __name__ == "__main__":
    main()