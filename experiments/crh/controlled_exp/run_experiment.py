#!/usr/bin/env python3
"""
CRH/controlled_exp/run_experiment.py

Controlled experiment: two FC ReLU networks (A and B) trained on the same
sequence of batches, differing only in random initialisation (SEED_A / SEED_B).

Backward gram matrices at layer l:
  Mh = h̃ᵀh̃/N    h̃ = row-unit-normalised, column-mean-centred pre-act input
  Mg = g̃ᵀg̃/N    g̃ = same treatment of dL/dh
  Mw = WᵀW

Alignment pairs (from hz_metrics):
  RWA : Mh vs Mw
  RGA : Mg vs Mh
  GWA : Mg vs Mw

Theorem-30 quantities (per layer, per network):
  S_i      = tr(WᵀW)
  C_hat    = Mh / tr(Mh)         trace-normalised input second moment
  G_hat    = WᵀW / S_i           trace-normalised weight gram

  C1_delta = ‖C_hat^A - G_hat^A‖₂    (LHS of assumption C1, for network A)
  C2_delta = ‖C_hat^B - G_hat^B‖₂    (LHS of assumption C2, for network B)
  C3_tau   = ‖C_hat^A - C_hat^B‖₂    (LHS of assumption C3, cross-network)
"""

import sys
import os
import math
import json
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_CRH  = _HERE.parent
_SRC  = _CRH.parent.parent / "src"
for _p in (str(_SRC), str(_CRH)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import hz_metrics as hzm
import utils

# =============================================================================
# CONFIG
# =============================================================================
D            = 100
WIDTH        = 100
DEPTH        = 6

SEED_A       = 42     # initialisation seed for network A
SEED_B       = 123    # initialisation seed for network B
DATA_SEED_A  = 999    # minibatch sequence for network A
DATA_SEED_B  = 888    # minibatch sequence for network B (different → realistic stochasticity)

OPTIMIZER    = "sgd"    # "adam" | "sgd"
LR           = 1e-2
MOMENTUM     = 0.9
WEIGHT_DECAY = 3e-5

TRAIN_BS     = 32
EVAL_BS      = 3000
N_STEPS      = 30_001
EVAL_EVERY   = 250

TARGET         = "teacher_clf"   # "linear" | "sin" | "teacher_mlp" | "teacher_clf"
TEACHER_HIDDEN = 50              # hidden dim of the shallow teacher (< D → low-rank structure)
NUM_CLASSES    = 10              # number of classes for teacher_clf

K_ANGLES     = 10

DEVICE  = str(utils.get_device())   # cuda > mps > cpu
OUT_DIR = os.path.join(os.path.dirname(__file__), f"results_{TARGET}")
# =============================================================================


# -----------------------------------------------------------------------------
# Model
# -----------------------------------------------------------------------------
class FCNet(nn.Module):
    def __init__(self, seed: int):
        super().__init__()
        torch.manual_seed(seed)
        out_dim = NUM_CLASSES if TARGET == "teacher_clf" else D
        sizes = [D] + [WIDTH] * (DEPTH - 1) + [out_dim]
        self.layers = nn.ModuleList(
            nn.Linear(sizes[i], sizes[i + 1], bias=True) for i in range(DEPTH)
        )
        self.pre_act: list = []
        self.weights: list = []
        self._record_grads: bool = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.pre_act = []
        self.weights = []
        for i, layer in enumerate(self.layers):
            if self._record_grads:
                x.retain_grad()
            self.pre_act.append(x)
            self.weights.append(layer.weight.data.cpu())
            x = layer(x)
            if i < len(self.layers) - 1:
                x = F.relu(x)
        return x


# -----------------------------------------------------------------------------
# Target
# -----------------------------------------------------------------------------
torch.manual_seed(0)   # fixed teacher seed — independent of network seeds
_W_true    = torch.randn(D, D) * 0.3
_Z         = torch.empty(D, D).uniform_(0.0, 0.5)
# Shallow teacher: Y = ReLU(X @ W1 / √D) @ W2  +  noise
# W1: (D, TEACHER_HIDDEN),  W2: (TEACHER_HIDDEN, D or NUM_CLASSES)
# Dividing by √D keeps pre-activations O(1) → ~half of ReLU neurons fire
_W1_teacher     = torch.randn(D, TEACHER_HIDDEN) / math.sqrt(D)
_W2_teacher     = torch.randn(TEACHER_HIDDEN, D) / math.sqrt(TEACHER_HIDDEN)
_W2_teacher_clf = torch.randn(TEACHER_HIDDEN, NUM_CLASSES) / math.sqrt(TEACHER_HIDDEN)


def get_target(X: torch.Tensor) -> torch.Tensor:
    if TARGET == "linear":
        W = _W_true.to(X.device)
        return X @ W + 0.05 * torch.randn(X.shape[0], D, device=X.device)
    elif TARGET == "teacher_mlp":
        W1 = _W1_teacher.to(X.device)
        W2 = _W2_teacher.to(X.device)
        hidden = F.relu(X @ W1)          # (N, TEACHER_HIDDEN)
        return hidden @ W2 + 0.05 * torch.randn(X.shape[0], D, device=X.device)
    elif TARGET == "teacher_clf":
        W1 = _W1_teacher.to(X.device)
        W2 = _W2_teacher_clf.to(X.device)
        hidden = F.relu(X @ W1)          # (N, TEACHER_HIDDEN)
        logits = hidden @ W2             # (N, NUM_CLASSES)
        return logits.argmax(dim=1)      # (N,) long — hard labels
    else:   # sin
        Z = _Z.to(X.device)
        return torch.sin(X @ Z) + 0.1 * torch.randn(X.shape[0], D, device=X.device)


def compute_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    if TARGET == "teacher_clf":
        return F.cross_entropy(logits, targets)
    return F.mse_loss(logits, targets)


def compute_acc(logits: torch.Tensor, targets: torch.Tensor) -> float:
    if TARGET != "teacher_clf":
        return float("nan")
    return float((logits.argmax(dim=1) == targets).float().mean())


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
def gram(x: torch.Tensor) -> torch.Tensor:
    """(N, d) → (d, d) double sample covariance; rows unit-normalised, mean-centred."""
    x = x.float()
    x = x / (x.norm(dim=1, keepdim=True) + 1e-10)
    x = x - x.mean(dim=0, keepdim=True)
    return ((x.T @ x) / x.shape[0]).double()


def spectral_norm_sym(M: torch.Tensor) -> float:
    """Spectral norm of a symmetric matrix = largest |eigenvalue|."""
    return float(torch.linalg.eigvalsh(M).abs().max())


def _flatten(prefix: str, d: dict) -> dict:
    out = {}
    for k, v in d.items():
        key = f"{prefix}.{k}"
        if isinstance(v, dict):
            out.update(_flatten(key, v))
        elif not isinstance(v, list):
            try:
                out[key] = float(v)
            except (TypeError, ValueError):
                pass
    return out


# -----------------------------------------------------------------------------
# Per-network evaluation
# -----------------------------------------------------------------------------
def _eval_single(net: FCNet, X: torch.Tensor, Y: torch.Tensor) -> tuple:
    """
    Forward + backward pass for one network on the given (X, Y).
    Returns (layer_metrics_list, eval_loss, C_hats).
    C_hats[l] is the trace-normalised Mh (C_hat) tensor for layer l.
    """
    X_in = X.detach().requires_grad_(True)
    out  = net(X_in)
    loss = compute_loss(out, Y)
    loss.backward()

    layer_metrics = []
    C_hats        = []

    for l in range(DEPTH):
        h = net.pre_act[l]
        g = h.grad
        W = net.weights[l]

        if g is None:
            layer_metrics.append({})
            C_hats.append(None)
            continue

        Mh = gram(h.detach().cpu())
        Mg = gram(g.cpu())
        Mw = W.double().T @ W.double()

        def hz(A, B, pair):
            return _flatten(pair, hzm.compute_all_metrics(
                A, B, sym=True, k=K_ANGLES,
                alpha_min=0.25, alpha_max=3.0, alpha_steps=56))

        m = {}
        m.update(hz(Mh, Mw, "RWA"))
        m.update(hz(Mg, Mh, "RGA"))
        m.update(hz(Mg, Mw, "GWA"))

        # C1 / C2 quantity for this network
        S_i   = float(torch.trace(Mw).clamp_min(1e-30))
        tr_C  = float(torch.trace(Mh).clamp_min(1e-30))
        C_hat = Mh / tr_C
        G_hat = Mw / S_i
        delta = spectral_norm_sym(C_hat - G_hat)

        m["C1_delta"] = delta   # C1_delta for A, C2_delta for B
        m["S_i"]      = S_i
        m["tr_C"]     = tr_C
        m["svd_vals"] = torch.linalg.svdvals(W.float()).tolist()

        layer_metrics.append(m)
        C_hats.append(C_hat)

    acc = compute_acc(out.detach(), Y)
    return layer_metrics, float(loss), C_hats, acc


def evaluate_pair(net_A: FCNet, net_B: FCNet) -> tuple:
    """
    Evaluate both networks on the same fresh batch.
    Adds C3_tau = ‖C_hat^A - C_hat^B‖₂ to every layer's metrics.
    Returns (lm_A, loss_A, lm_B, loss_B).
    """
    X = torch.randn(EVAL_BS, D, device=DEVICE)
    Y = get_target(X)

    net_A._record_grads = True
    lm_A, loss_A, C_hats_A, acc_A = _eval_single(net_A, X, Y)
    net_A._record_grads = False
    net_A.zero_grad()

    net_B._record_grads = True
    lm_B, loss_B, C_hats_B, acc_B = _eval_single(net_B, X, Y)
    net_B._record_grads = False
    net_B.zero_grad()

    # C3: cross-network spectral gap of trace-normalised representations
    # mu: relative weight-scale mismatch  |S_i^B / S_i^A - 1|  (condition iii)
    for l in range(DEPTH):
        if C_hats_A[l] is not None and C_hats_B[l] is not None:
            tau = spectral_norm_sym(C_hats_A[l] - C_hats_B[l])
            lm_A[l]["C3_tau"] = tau
            lm_B[l]["C3_tau"] = tau

        if lm_A[l] and lm_B[l]:
            S_A = lm_A[l].get("S_i", 0.0)
            S_B = lm_B[l].get("S_i", 0.0)
            mu  = abs(S_B / S_A - 1.0) if S_A > 1e-30 else float("nan")
            lm_A[l]["mu_i"] = mu
            lm_B[l]["mu_i"] = mu

            # RHS of Theorem-30 bound:
            #   sqrt( S_i^(A) * (delta_i^(A) + delta_i^(B) + tau_i + mu_i) )
            delta_A = lm_A[l].get("C1_delta", float("nan"))
            delta_B = lm_B[l].get("C1_delta", float("nan"))
            tau     = lm_A[l].get("C3_tau",   float("nan"))
            rhs = math.sqrt(S_A * (delta_A + delta_B + tau + mu)) if S_A > 1e-30 else float("nan")
            lm_A[l]["theorem30_rhs"] = rhs
            lm_B[l]["theorem30_rhs"] = rhs

    return lm_A, loss_A, acc_A, lm_B, loss_B, acc_B


# -----------------------------------------------------------------------------
# Training loop
# -----------------------------------------------------------------------------
def make_opt(net: FCNet):
    if OPTIMIZER == "adam":
        return optim.Adam(net.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    return optim.SGD(net.parameters(), lr=LR, momentum=MOMENTUM,
                     weight_decay=WEIGHT_DECAY)


def train() -> list:
    net_A, net_B = FCNet(SEED_A).to(DEVICE), FCNet(SEED_B).to(DEVICE)
    opt_A, opt_B = make_opt(net_A), make_opt(net_B)

    # separate data generators — each network sees its own minibatch sequence
    data_rng_A = torch.Generator(device=DEVICE)
    data_rng_A.manual_seed(DATA_SEED_A)
    data_rng_B = torch.Generator(device=DEVICE)
    data_rng_B.manual_seed(DATA_SEED_B)

    print(f"Device={DEVICE} | D={D} width={WIDTH} depth={DEPTH} | "
          f"{OPTIMIZER} lr={LR} wd={WEIGHT_DECAY} | target={TARGET} | "
          f"{N_STEPS} steps | init seeds A={SEED_A} B={SEED_B} | "
          f"data seeds A={DATA_SEED_A} B={DATA_SEED_B}")

    log = []
    for step in range(N_STEPS):
        # each network gets its own batch
        X_tr_A = torch.randn(TRAIN_BS, D, device=DEVICE, generator=data_rng_A)
        Y_tr_A = get_target(X_tr_A)
        X_tr_B = torch.randn(TRAIN_BS, D, device=DEVICE, generator=data_rng_B)
        Y_tr_B = get_target(X_tr_B)

        for net, opt, X_tr, Y_tr in [(net_A, opt_A, X_tr_A, Y_tr_A),
                                      (net_B, opt_B, X_tr_B, Y_tr_B)]:
            net._record_grads = False
            opt.zero_grad()
            compute_loss(net(X_tr), Y_tr).backward()
            opt.step()

        if step % EVAL_EVERY == 0:
            with torch.enable_grad():
                lm_A, loss_A, acc_A, lm_B, loss_B, acc_B = evaluate_pair(net_A, net_B)

            # recompute training loss/acc on each network's own last batch
            with torch.no_grad():
                out_tr_A  = net_A(X_tr_A)
                out_tr_B  = net_B(X_tr_B)
                tr_loss_A = float(compute_loss(out_tr_A, Y_tr_A))
                tr_loss_B = float(compute_loss(out_tr_B, Y_tr_B))
                tr_acc_A  = compute_acc(out_tr_A, Y_tr_A)
                tr_acc_B  = compute_acc(out_tr_B, Y_tr_B)

            log.append(dict(
                step=step,
                A=dict(train_loss=tr_loss_A, eval_loss=loss_A,
                       train_acc=tr_acc_A, eval_acc=acc_A, layers=lm_A),
                B=dict(train_loss=tr_loss_B, eval_loss=loss_B,
                       train_acc=tr_acc_B, eval_acc=acc_B, layers=lm_B),
            ))
            if TARGET == "teacher_clf":
                print(f"  step {step:6d} | "
                      f"A train={tr_loss_A:.4f} acc={tr_acc_A:.3f} | "
                      f"B train={tr_loss_B:.4f} acc={tr_acc_B:.3f}")
            else:
                print(f"  step {step:6d} | "
                      f"A train={tr_loss_A:.4f} eval={loss_A:.4f} | "
                      f"B train={tr_loss_B:.4f} eval={loss_B:.4f}")

    return log


# -----------------------------------------------------------------------------
# Plotting
# -----------------------------------------------------------------------------
def plot(log: list) -> None:
    utils.apply_stitching_trend_style()
    palette = utils.get_deep_palette()
    COL_A, COL_B = palette[0], palette[1]   # blue, orange

    plots_dir = os.path.join(OUT_DIR, "plots")
    os.makedirs(plots_dir, exist_ok=True)
    steps = np.array([e["step"] for e in log])

    def s(net: str, layer: int, key: str) -> np.ndarray:
        return np.array([e[net]["layers"][layer].get(key, np.nan) for e in log])

    # helper: 2×3 grid, overlay A and B
    def grid6_AB(key: str, title: str, ylabel: str, ylim=None) -> plt.Figure:
        fig, axes = plt.subplots(2, 3, figsize=(15, 8), constrained_layout=True)
        fig.suptitle(title, fontsize=14, fontweight="bold")
        for l, ax in enumerate(axes.flat):
            if l >= DEPTH:
                ax.set_visible(False)
                continue
            ax.plot(steps, s("A", l, key), color=COL_A, lw=2.0, label="A")
            ax.plot(steps, s("B", l, key), color=COL_B, lw=2.0, label="B", ls="--")
            ax.set(title=f"Layer {l + 1}", xlabel="step", ylabel=ylabel)
            if ylim:
                ax.set_ylim(ylim)
            ax.legend()
        return fig

    # 1. RWA cosine_fro
    fig = grid6_AB("RWA.cosine_sim_fro",
                   "Backward RWA  –  cosine_fro(Mh, Mw)", "cosine_fro", ylim=(-0.05, 1.05))
    fig.savefig(f"{plots_dir}/rwa_cosine_fro.png", dpi=150); plt.close(fig)

    # 2. RWA Pearson correlation
    fig = grid6_AB("RWA.pearson_correlation",
                   "Backward RWA  –  Pearson correlation(Mh, Mw)", "pearson_corr",
                   ylim=(-0.1, 1.05))
    fig.savefig(f"{plots_dir}/rwa_pearson.png", dpi=150); plt.close(fig)

    # 3. RWA eps_lin
    fig = grid6_AB("RWA.eps_lin",
                   "Backward RWA  –  eps_lin(Mh, Mw)  [lower = more linear]", "eps_lin")
    fig.savefig(f"{plots_dir}/rwa_eps_lin.png", dpi=150); plt.close(fig)

    # 4. RWA commutator norm
    fig = grid6_AB("RWA.eps_comm",
                   "Backward RWA  –  commutator_norm(Mh, Mw)  [lower = more commuting]",
                   "eps_comm")
    fig.savefig(f"{plots_dir}/rwa_eps_comm.png", dpi=150); plt.close(fig)

    # 5. RWA principal angles
    fig = grid6_AB("RWA.principal_angles_topk.mean_angle_deg",
                   f"Backward RWA  –  mean principal angle, top-{K_ANGLES} eigvecs",
                   "mean angle (deg)")
    fig.savefig(f"{plots_dir}/rwa_principal_angles.png", dpi=150); plt.close(fig)

    # 6. RWA best power fit
    for net_id, col in [("A", COL_A), ("B", COL_B)]:
        fig, axes = plt.subplots(2, 3, figsize=(15, 8), constrained_layout=True)
        fig.suptitle(f"Backward RWA  –  best power fit  Mh ≈ c·Mw^α  (net {net_id})",
                     fontsize=14, fontweight="bold")
        for l, ax in enumerate(axes.flat):
            if l >= DEPTH:
                ax.set_visible(False)
                continue
            ax2 = ax.twinx()
            ax.plot(steps,  s(net_id, l, "RWA.best_power_fit.eps_pow"),
                    color=col,          lw=2.0, label="eps_pow")
            ax2.plot(steps, s(net_id, l, "RWA.best_power_fit.alpha"),
                     color=palette[3],  lw=2.0, ls="--", label="best α")
            ax.set(title=f"Layer {l + 1}", xlabel="step", ylabel="eps_pow")
            ax2.set_ylabel("best α", color=palette[3])
            h1, lb1 = ax.get_legend_handles_labels()
            h2, lb2 = ax2.get_legend_handles_labels()
            ax.legend(h1 + h2, lb1 + lb2, loc="upper right")
        fig.savefig(f"{plots_dir}/rwa_power_fit_{net_id}.png", dpi=150); plt.close(fig)

    # 7. RWA R² power fit
    fig = grid6_AB("RWA.best_power_fit.r2_pow",
                   "Backward RWA  –  R² of best power fit", "r2_pow", ylim=(-0.05, 1.05))
    fig.savefig(f"{plots_dir}/rwa_r2_pow.png", dpi=150); plt.close(fig)

    # 8. RWA power-law eigenvalue fit
    for net_id, col in [("A", COL_A), ("B", COL_B)]:
        fig, axes = plt.subplots(2, 3, figsize=(15, 8), constrained_layout=True)
        fig.suptitle(f"Backward RWA  –  power-law eigenvalue fit  (net {net_id})",
                     fontsize=14, fontweight="bold")
        for l, ax in enumerate(axes.flat):
            if l >= DEPTH:
                ax.set_visible(False)
                continue
            ax2 = ax.twinx()
            ax.plot(steps,  s(net_id, l, "RWA.powerlaw_eigs.alpha"),
                    color=col,         lw=2.0, label="alpha")
            ax2.plot(steps, s(net_id, l, "RWA.powerlaw_eigs.r2"),
                     color=palette[2], lw=2.0, ls="--", label="R²")
            ax.set(title=f"Layer {l + 1}", xlabel="step", ylabel="slope α")
            ax2.set_ylabel("R²", color=palette[2])
            h1, lb1 = ax.get_legend_handles_labels()
            h2, lb2 = ax2.get_legend_handles_labels()
            ax.legend(h1 + h2, lb1 + lb2)
        fig.savefig(f"{plots_dir}/rwa_powerlaw_eigs_{net_id}.png", dpi=150); plt.close(fig)

    # 9. All pairs cosine_fro (per layer, overlay A and B)
    for l in range(DEPTH):
        fig, axes = plt.subplots(1, 3, figsize=(14, 4), constrained_layout=True)
        fig.suptitle(f"Layer {l + 1}  –  backward alignment cosine_fro, all pairs",
                     fontsize=14, fontweight="bold")
        pair_colors = {"RWA": palette[0], "RGA": palette[1], "GWA": palette[2]}
        for ax, pair in zip(axes, ["RWA", "RGA", "GWA"]):
            ax.plot(steps, s("A", l, f"{pair}.cosine_sim_fro"),
                    color=pair_colors[pair], lw=2.0, label="A")
            ax.plot(steps, s("B", l, f"{pair}.cosine_sim_fro"),
                    color=pair_colors[pair], lw=2.0, ls="--", alpha=0.8, label="B")
            ax.set(title=pair, xlabel="step", ylabel="cosine_fro", ylim=(-0.05, 1.05))
            ax.legend()
        fig.savefig(f"{plots_dir}/layer{l + 1}_all_pairs.png", dpi=150)
        plt.close(fig)

    # 10. C1 / C2 spectral gap (overlay A and B)
    fig = grid6_AB("C1_delta",
                   "C1/C2 spectral gap  ‖Ĉ - Ĝ‖₂  per layer  [lower = C1/C2 better satisfied]",
                   "spectral gap")
    fig.savefig(f"{plots_dir}/C12_spectral_gap.png", dpi=150); plt.close(fig)

    # 11. C3 tau (same for both networks, just plot once)
    fig, axes = plt.subplots(2, 3, figsize=(15, 8), constrained_layout=True)
    fig.suptitle("C3  –  ‖Ĉ^A − Ĉ^B‖₂  per layer  [cross-network representation gap]",
                 fontsize=14, fontweight="bold")
    for l, ax in enumerate(axes.flat):
        if l >= DEPTH:
            ax.set_visible(False)
            continue
        ax.plot(steps, s("A", l, "C3_tau"), color=palette[3], lw=2.0)
        ax.set(title=f"Layer {l + 1}", xlabel="step", ylabel="τ  (spectral norm)")
    fig.savefig(f"{plots_dir}/C3_tau.png", dpi=150); plt.close(fig)

    # 12. mu_i = |S_i^B / S_i^A - 1|  (layerwise scale matching, condition iii)
    fig, axes = plt.subplots(2, 3, figsize=(15, 8), constrained_layout=True)
    fig.suptitle("Condition (iii)  –  μ_i = |S_i^B / S_i^A − 1|  per layer  [lower = better scale match]",
                 fontsize=14, fontweight="bold")
    for l, ax in enumerate(axes.flat):
        if l >= DEPTH:
            ax.set_visible(False)
            continue
        ax.plot(steps, s("A", l, "mu_i"), color=palette[4], lw=2.0)
        ax.set(title=f"Layer {l + 1}", xlabel="step", ylabel="μ_i")
    fig.savefig(f"{plots_dir}/mu_scale_mismatch.png", dpi=150); plt.close(fig)

    # 13. Theorem-30 RHS bound per layer
    fig, axes = plt.subplots(2, 3, figsize=(15, 8), constrained_layout=True)
    fig.suptitle(
        r"Theorem-30 RHS  –  $\sqrt{S_i^{(A)}(\delta_i^{(A)}+\delta_i^{(B)}+\tau_i+\mu_i)}$  per layer",
        fontsize=14, fontweight="bold")
    for l, ax in enumerate(axes.flat):
        if l >= DEPTH:
            ax.set_visible(False)
            continue
        ax.plot(steps, s("A", l, "theorem30_rhs"), color=palette[5], lw=2.0)
        ax.set(title=f"Layer {l + 1}", xlabel="step", ylabel="RHS bound")
    fig.savefig(f"{plots_dir}/theorem30_rhs.png", dpi=150); plt.close(fig)

    # 14. S_i and tr(C) per layer (overlay A and B)
    for l in range(DEPTH):
        fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
        fig.suptitle(f"Layer {l + 1}  –  normalisation denominators",
                     fontsize=14, fontweight="bold")
        for ax, key, ylabel in zip(axes, ["S_i", "tr_C"], ["S_i = tr(WᵀW)", "tr(C)"]):
            ax.plot(steps, s("A", l, key), color=COL_A, lw=2.0, label="A")
            ax.plot(steps, s("B", l, key), color=COL_B, lw=2.0, ls="--", label="B")
            ax.set(title=ylabel, xlabel="step", ylabel=ylabel)
            ax.legend()
        fig.savefig(f"{plots_dir}/layer{l + 1}_Si_trC.png", dpi=150)
        plt.close(fig)

    # 15. Singular value spectrum — one plot per layer, all checkpoints overlaid
    n_log = len(log)
    # use all logged checkpoints, coloured by training progress
    all_indices = list(range(n_log))
    cmap = plt.get_cmap("plasma")
    sv_svd_dir = os.path.join(plots_dir, "svd_per_layer")
    os.makedirs(sv_svd_dir, exist_ok=True)

    for l in range(DEPTH):
        fig, axes = plt.subplots(1, 2, figsize=(12, 4), constrained_layout=True)
        fig.suptitle(f"Layer {l + 1}  –  singular value spectrum  (A left, B right)",
                     fontsize=13, fontweight="bold")
        for ax, net_id in zip(axes, ["A", "B"]):
            for idx in all_indices:
                frac  = idx / max(n_log - 1, 1)
                color = cmap(frac)
                sv = log[idx][net_id]["layers"][l].get("svd_vals")
                if sv:
                    sv_sorted = sorted(sv, reverse=True)
                    ax.plot(range(1, len(sv_sorted) + 1), sv_sorted,
                            color=color, lw=0.8, alpha=0.7)
            # colour-bar-style legend: mark a few representative steps
            sm = plt.cm.ScalarMappable(cmap=cmap,
                                       norm=plt.Normalize(vmin=0, vmax=steps[-1]))
            sm.set_array([])
            fig.colorbar(sm, ax=ax, label="training step")
            ax.set(title=f"Net {net_id}", xlabel="singular value index", ylabel="σ")
        fig.savefig(f"{sv_svd_dir}/layer{l + 1}_svd_spectrum.png", dpi=150)
        plt.close(fig)

    # 15b. Summary grid (5 key checkpoints, both nets) — kept for quick overview
    ckpt_indices = sorted(set([
        0,
        n_log // 4,
        n_log // 2,
        3 * n_log // 4,
        n_log - 1,
    ]))
    ckpt_colors = [palette[i % len(palette)] for i in range(len(ckpt_indices))]

    fig, axes = plt.subplots(2, 3, figsize=(15, 8), constrained_layout=True)
    fig.suptitle("Singular value spectrum of W per layer  (solid=A, dashed=B)",
                 fontsize=14, fontweight="bold")
    for l, ax in enumerate(axes.flat):
        if l >= DEPTH:
            ax.set_visible(False)
            continue
        for idx, col in zip(ckpt_indices, ckpt_colors):
            entry = log[idx]
            step_lbl = f"step {entry['step']}"
            sv_A = entry["A"]["layers"][l].get("svd_vals")
            sv_B = entry["B"]["layers"][l].get("svd_vals")
            if sv_A:
                ax.plot(range(1, len(sv_A) + 1), sorted(sv_A, reverse=True),
                        color=col, lw=1.8, label=step_lbl)
            if sv_B:
                ax.plot(range(1, len(sv_B) + 1), sorted(sv_B, reverse=True),
                        color=col, lw=1.8, ls="--", alpha=0.75)
        ax.set(title=f"Layer {l + 1}", xlabel="singular value index", ylabel="σ")
        ax.legend(fontsize=7)
    fig.savefig(f"{plots_dir}/svd_spectrum.png", dpi=150); plt.close(fig)

    # 17. Loss curves
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), constrained_layout=True)
    fig.suptitle("Training curves", fontsize=14, fontweight="bold")
    ylabel = "cross-entropy" if TARGET == "teacher_clf" else "MSE"
    for ax, split in zip(axes, ["train_loss", "eval_loss"]):
        ax.plot(steps, [e["A"][split] for e in log], color=COL_A, lw=2.0, label="A")
        ax.plot(steps, [e["B"][split] for e in log], color=COL_B, lw=2.0, ls="--", label="B")
        ax.set(title=split, xlabel="step", ylabel=ylabel)
        ax.legend()
    fig.savefig(f"{plots_dir}/loss_curves.png", dpi=150); plt.close(fig)

    # 17b. Accuracy curves (classification only)
    if TARGET == "teacher_clf":
        fig, axes = plt.subplots(1, 2, figsize=(12, 4), constrained_layout=True)
        fig.suptitle("Accuracy curves", fontsize=14, fontweight="bold")
        for ax, split in zip(axes, ["train_acc", "eval_acc"]):
            ax.plot(steps, [e["A"][split] for e in log], color=COL_A, lw=2.0, label="A")
            ax.plot(steps, [e["B"][split] for e in log], color=COL_B, lw=2.0, ls="--", label="B")
            ax.set(title=split, xlabel="step", ylabel="accuracy", ylim=(-0.05, 1.05))
            ax.axhline(1.0 / NUM_CLASSES, color="grey", lw=1.2, ls=":", label="chance")
            ax.legend()
        fig.savefig(f"{plots_dir}/accuracy_curves.png", dpi=150); plt.close(fig)

    print(f"Plots saved to {plots_dir}/")


# -----------------------------------------------------------------------------
# JSON serialiser
# -----------------------------------------------------------------------------
def _json_default(x):
    if isinstance(x, float) and (math.isnan(x) or math.isinf(x)):
        return None
    return float(x)


# -----------------------------------------------------------------------------
# Entry point
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    os.makedirs(OUT_DIR, exist_ok=True)

    log = train()

    path = os.path.join(OUT_DIR, "metrics.json")
    with open(path, "w") as f:
        json.dump(log, f, indent=2, default=_json_default)
    print(f"Results saved to {path}")

    plot(log)
