# hz_metrics.py
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, Iterable, Optional, Tuple

import numpy as np
import torch


def _to_tensor(x: Any, *, dtype=torch.float64, device="cpu") -> torch.Tensor:
    if torch.is_tensor(x):
        return x.detach().to(device=device, dtype=dtype)
    if isinstance(x, np.ndarray):
        return torch.from_numpy(x).to(device=device, dtype=dtype)
    raise TypeError(f"Unsupported type for matrix: {type(x)}")


def load_matrix(path: str, key: Optional[str] = None) -> torch.Tensor:
    """
    Loads a square matrix from:
      - .pt/.pth (torch.load)
      - .npy (numpy.load)
    Accepts:
      - a tensor
      - a dict containing exactly one tensor
      - a dict with a provided key
    """
    if path.endswith(".npy"):
        arr = np.load(path)
        return _to_tensor(arr)

    obj = torch.load(path, map_location="cpu")

    if torch.is_tensor(obj):
        return _to_tensor(obj)

    if isinstance(obj, dict):
        if key is not None:
            if key not in obj:
                raise KeyError(f"Key '{key}' not found in {path}. Keys: {list(obj.keys())[:20]}")
            return _to_tensor(obj[key])

        # If key not specified: pick the only tensor value if unambiguous
        tensor_vals = [(k, v) for k, v in obj.items() if torch.is_tensor(v) or isinstance(v, np.ndarray)]
        if len(tensor_vals) == 1:
            return _to_tensor(tensor_vals[0][1])

        raise ValueError(
            f"Ambiguous dict in {path}. Provide --H_key/--Z_key. "
            f"Candidate tensor keys: {[k for k, _ in tensor_vals][:20]}"
        )

    raise ValueError(f"Unsupported file payload in {path}: {type(obj)}")


def symmetrize(M: torch.Tensor) -> torch.Tensor:
    return 0.5 * (M + M.T)


def fro_inner(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    return torch.sum(A * B)


def fro_norm(A: torch.Tensor) -> torch.Tensor:
    return torch.linalg.norm(A, ord="fro")


def best_fit_scalar(H: torch.Tensor, Z: torch.Tensor, eps: float = 1e-30) -> float:
    denom = float(fro_inner(Z, Z).clamp_min(eps))
    return float(fro_inner(H, Z) / denom)


def epsilon_lin(H: torch.Tensor, Z: torch.Tensor, eps: float = 1e-30) -> Tuple[float, float]:
    """
    Returns (eps_lin, c_star) where:
      eps_lin = ||H - c Z||_F / ||H||_F   with c chosen to minimize numerator.
    """
    nH = float(fro_norm(H).clamp_min(eps))
    c = best_fit_scalar(H, Z, eps=eps)
    resid = fro_norm(H - c * Z)
    return float(resid / nH), float(c)


def cosine_fro(H: torch.Tensor, Z: torch.Tensor, eps: float = 1e-30) -> float:
    nH = float(fro_norm(H).clamp_min(eps))
    nZ = float(fro_norm(Z).clamp_min(eps))
    return float(fro_inner(H, Z) / (nH * nZ))


def commutator_norm(H: torch.Tensor, Z: torch.Tensor, eps: float = 1e-30) -> float:
    nH = float(fro_norm(H).clamp_min(eps))
    nZ = float(fro_norm(Z).clamp_min(eps))
    C = H @ Z - Z @ H
    return float(fro_norm(C) / (nH * nZ))


def _eigh_sorted(M: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    # M assumed symmetric
    evals, evecs = torch.linalg.eigh(M)
    # sort descending by eval
    idx = torch.argsort(evals, descending=True)
    return evals[idx], evecs[:, idx]


def topk_eigenspace(M: torch.Tensor, k: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Returns (evals_topk, U_topk) where columns of U_topk are top-k eigenvectors.
    """
    evals, evecs = _eigh_sorted(M)
    k = int(min(k, evecs.shape[1]))
    return evals[:k], evecs[:, :k]


def principal_angles_topk(H: torch.Tensor, Z: torch.Tensor, k: int, eps: float = 1e-12) -> Dict[str, float]:
    """
    Principal angles between top-k eigenspaces of H and Z.
    Returns summary stats:
      mean_cos, min_cos, max_angle_deg, mean_angle_deg
    """
    _, UH = topk_eigenspace(H, k)
    _, UZ = topk_eigenspace(Z, k)
    M = UH.T @ UZ
    s = torch.linalg.svdvals(M).clamp(0.0, 1.0)  # cosines of angles
    angles = torch.acos(s.clamp(0.0, 1.0))
    return {
        "k": float(min(k, UH.shape[1], UZ.shape[1])),
        "mean_cos": float(s.mean()),
        "min_cos": float(s.min()),
        "mean_angle_deg": float(angles.mean() * 180.0 / math.pi),
        "max_angle_deg": float(angles.max() * 180.0 / math.pi),
    }


def powerlaw_fit_eigs(
    H: torch.Tensor,
    Z: torch.Tensor,
    eig_tol: float = 1e-12,
    max_points: Optional[int] = None,
) -> Dict[str, float]:
    """
    Fit log mu_i ≈ log c + alpha log lambda_i using eigenvalues of H (mu) and Z (lambda).
    Uses paired order statistics: sort both descending and match by index.
    """
    evals_H, _ = _eigh_sorted(H)
    evals_Z, _ = _eigh_sorted(Z)

    mu = evals_H.detach().cpu().double().numpy()
    la = evals_Z.detach().cpu().double().numpy()

    mu = mu[mu > eig_tol]
    la = la[la > eig_tol]

    n = int(min(len(mu), len(la)))
    if max_points is not None:
        n = int(min(n, max_points))

    if n < 3:
        return {"alpha": float("nan"), "c": float("nan"), "r2": float("nan"), "n_points": float(n)}

    mu = mu[:n]
    la = la[:n]

    x = np.log(la)
    y = np.log(mu)

    x_mean = float(x.mean())
    y_mean = float(y.mean())
    var_x = float(((x - x_mean) ** 2).sum())
    if var_x <= 0:
        return {"alpha": float("nan"), "c": float("nan"), "r2": float("nan"), "n_points": float(n)}

    cov_xy = float(((x - x_mean) * (y - y_mean)).sum())
    alpha = cov_xy / var_x
    intercept = y_mean - alpha * x_mean  # = log c
    yhat = intercept + alpha * x

    ss_res = float(((y - yhat) ** 2).sum())
    ss_tot = float(((y - y_mean) ** 2).sum())
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    c = float(np.exp(intercept))

    return {"alpha": float(alpha), "c": float(c), "r2": float(r2), "n_points": float(n)}


def spectral_power(
    Z: torch.Tensor,
    alpha: float,
    *,
    allow_indefinite: bool = True,
) -> torch.Tensor:
    """
    Z^alpha via eigendecomposition.
    If allow_indefinite=True: uses signed power: sign(λ)*|λ|^alpha (keeps real output).
    If False: clips λ<0 to 0 (PSD projection).
    """
    evals, evecs = _eigh_sorted(Z)
    if allow_indefinite:
        lam = evals
        lam_pow = torch.sign(lam) * torch.pow(torch.abs(lam), alpha)
    else:
        lam = torch.clamp(evals, min=0.0)
        lam_pow = torch.pow(lam, alpha)
    return (evecs * lam_pow.unsqueeze(0)) @ evecs.T


def best_fit_power_residual(
    H: torch.Tensor,
    Z: torch.Tensor,
    alpha_grid: Iterable[float],
    *,
    allow_indefinite: bool = True,
    eps: float = 1e-30,
) -> Dict[str, float]:
    """
    eps_pow(H,Z) = min_{alpha in grid} min_c ||H - c Z^alpha||_F / ||H||_F
    Returns best {eps_pow, alpha, c}.
    """
    nH = float(fro_norm(H).clamp_min(eps))
    best = {"eps_pow": float("inf"), "alpha": float("nan"), "c": float("nan")}

    for a in alpha_grid:
        Za = spectral_power(Z, float(a), allow_indefinite=allow_indefinite)
        denom = float(fro_inner(Za, Za).clamp_min(eps))
        c = float(fro_inner(H, Za) / denom)
        e = float(fro_norm(H - c * Za) / nH)
        if e < best["eps_pow"]:
            best = {"eps_pow": e, "alpha": float(a), "c": c}

    return best


def compute_all_metrics(
    H: torch.Tensor,
    Z: torch.Tensor,
    *,
    sym: bool = True,
    k: int = 20,
    eig_tol: float = 1e-12,
    alpha_min: float = 0.25,
    alpha_max: float = 3.0,
    alpha_steps: int = 56,
    allow_indefinite: bool = True,
) -> Dict[str, Any]:
    if H.ndim != 2 or Z.ndim != 2 or H.shape[0] != H.shape[1] or Z.shape[0] != Z.shape[1]:
        raise ValueError(f"H and Z must be square matrices. Got H={tuple(H.shape)} Z={tuple(Z.shape)}")
    if H.shape != Z.shape:
        raise ValueError(f"H and Z must have the same shape. Got H={tuple(H.shape)} Z={tuple(Z.shape)}")

    H = _to_tensor(H)
    Z = _to_tensor(Z)
    if sym:
        H = symmetrize(H)
        Z = symmetrize(Z)

    eps_lin_val, c_lin = epsilon_lin(H, Z)
    rho = cosine_fro(H, Z)
    eps_comm = commutator_norm(H, Z)
    pa = principal_angles_topk(H, Z, k=k)
    pl = powerlaw_fit_eigs(H, Z, eig_tol=eig_tol)

    alpha_grid = np.linspace(alpha_min, alpha_max, int(alpha_steps)).tolist()
    powfit = best_fit_power_residual(H, Z, alpha_grid, allow_indefinite=allow_indefinite)

    return {
        "shape": list(H.shape),
        "eps_lin": eps_lin_val,
        "c_lin": c_lin,
        "rho_fro": rho,
        "eps_comm": eps_comm,
        "principal_angles_topk": pa,
        "powerlaw_eigs": pl,
        "best_power_fit": powfit,
        "settings": {
            "symmetrize": bool(sym),
            "k": int(k),
            "eig_tol": float(eig_tol),
            "alpha_grid": {"min": float(alpha_min), "max": float(alpha_max), "steps": int(alpha_steps)},
            "allow_indefinite": bool(allow_indefinite),
        },
    }