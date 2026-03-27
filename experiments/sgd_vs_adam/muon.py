"""
muon.py — Muon optimizer (Momentum Orthogonalized by Newton-Schulz).

Reference: Keller Jordan et al., https://github.com/KellerJordan/modded-nanogpt

Key idea:
  For 2-D weight matrices W, instead of applying the raw gradient G, apply
  the Newton-Schulz orthogonalization of the Nesterov momentum buffer.
  This keeps the update on (or near) the Stiefel manifold and removes the
  dependence of the step size on the scale of the activations / gradients.

  For 1-D parameters (biases, LayerNorm scales/shifts) the update falls
  back to standard SGD with Nesterov momentum.
"""
from __future__ import annotations

import torch
import torch.nn as nn
from typing import Iterable, List, Optional


# ---------------------------------------------------------------------------
# Newton-Schulz iteration
# ---------------------------------------------------------------------------

def _zeropower_via_newtonschulz(G: torch.Tensor, steps: int = 5) -> torch.Tensor:
    """
    Compute an approximation to G / ||G||_F projected onto orthogonal matrices
    via the degree-5 Newton-Schulz iteration.

    Returns X such that X.T @ X ≈ I (scaled so ||X||_F ≈ ||G||_F).
    Works for both wide (rows < cols) and tall (rows > cols) matrices.
    """
    assert G.ndim == 2, f"Expected 2-D tensor, got shape {G.shape}"
    eps = 1e-7
    a, b, c = 3.4445, -4.7750, 2.0315

    # Work in float32 for numerical stability
    X = G.float()
    norm = X.norm()
    X = X / (norm + eps)

    transposed = X.shape[0] > X.shape[1]
    if transposed:
        X = X.T  # ensure wide matrix (rows <= cols)

    for _ in range(steps):
        A = X @ X.T
        B = b * A + c * (A @ A)
        X = a * X + B @ X

    if transposed:
        X = X.T

    return (X * norm).to(G.dtype)


# ---------------------------------------------------------------------------
# Muon optimizer
# ---------------------------------------------------------------------------

class Muon(torch.optim.Optimizer):
    """
    Muon optimizer.

    Applies Nesterov momentum and then orthogonalizes the update via
    Newton-Schulz for 2-D weight matrices. Falls back to plain Nesterov SGD
    for 1-D / scalar parameters (biases, norm parameters).

    Recommended settings (MLP / small CNN on image classification):
      lr = 0.02, momentum = 0.95, ns_steps = 5

    Args:
        params:    iterable of parameters or param groups
        lr:        learning rate (default: 0.02)
        momentum:  momentum coefficient (default: 0.95)
        nesterov:  use Nesterov momentum (default: True)
        ns_steps:  number of Newton-Schulz iterations (default: 5; more = more accurate)
        weight_decay: L2 penalty applied to all parameters before the update (default: 0)
    """

    def __init__(
        self,
        params,
        lr: float = 0.02,
        momentum: float = 0.95,
        nesterov: bool = True,
        ns_steps: int = 5,
        weight_decay: float = 0.0,
        grad_clip: float = 1.0,
    ):
        defaults = dict(
            lr=lr,
            momentum=momentum,
            nesterov=nesterov,
            ns_steps=ns_steps,
            weight_decay=weight_decay,
            grad_clip=grad_clip,
        )
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr           = float(group["lr"])
            momentum     = float(group["momentum"])
            nesterov     = bool(group["nesterov"])
            ns_steps     = int(group["ns_steps"])
            weight_decay = float(group["weight_decay"])

            grad_clip = float(group["grad_clip"])

            for p in group["params"]:
                if p.grad is None:
                    continue

                g = p.grad.clone()

                # Skip NaN/Inf gradients
                if not torch.isfinite(g).all():
                    continue

                # Gradient clipping per-parameter
                if grad_clip > 0.0:
                    g_norm = g.norm()
                    if g_norm > grad_clip:
                        g = g * (grad_clip / g_norm)

                # Optional weight decay (applied as L2 before momentum)
                if weight_decay != 0.0:
                    g = g.add(p, alpha=weight_decay)

                state = self.state[p]
                if "buf" not in state:
                    state["buf"] = torch.zeros_like(g)

                buf = state["buf"]
                buf.mul_(momentum).add_(g)

                # Nesterov lookahead
                update = g.add(buf, alpha=momentum) if nesterov else buf.clone()

                # Orthogonalize weight matrices (2-D) and conv kernels (4-D)
                # Conv: reshape (out, in, h, w) → (out, in*h*w), orthogonalize, reshape back
                if update.ndim >= 2:
                    orig_shape = update.shape
                    update = _zeropower_via_newtonschulz(
                        update.reshape(orig_shape[0], -1), steps=ns_steps
                    ).reshape(orig_shape)

                p.add_(update, alpha=-lr)

        return loss


# ---------------------------------------------------------------------------
# Convenience builder (mirrors build_sgd_with_param_groups in common.py)
# ---------------------------------------------------------------------------

def build_muon(
    model: nn.Module,
    lr: float,
    momentum: float = 0.95,
    ns_steps: int = 5,
    weight_decay: float = 0.0,
) -> Muon:
    """
    Build a Muon optimizer with separate param groups:
      - 2-D weight matrices  → Muon (orthogonalized update, no weight decay)
      - biases / norms / 1-D → Muon fallback (Nesterov SGD, no weight decay)

    Weight decay is not applied to any group by default, since the
    orthogonalized update already implicitly regularizes the weight scale.
    """
    matrix_params: List[nn.Parameter] = []
    scalar_params: List[nn.Parameter] = []

    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        # Apply Muon to weight matrices (2-D) and conv kernels (4-D)
        # Everything else (biases, norms) gets plain Nesterov SGD
        if p.ndim >= 2:
            matrix_params.append(p)
        else:
            scalar_params.append(p)

    param_groups = [
        {"params": matrix_params, "weight_decay": 0.0},
        {"params": scalar_params, "weight_decay": 0.0},
    ]
    return Muon(param_groups, lr=lr, momentum=momentum, ns_steps=ns_steps, weight_decay=weight_decay)
