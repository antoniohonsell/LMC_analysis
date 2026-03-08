#!/usr/bin/env python3
# scripts/compute_hz_metrics_sweep.py
"""
Example of usage:

export PYTHONPATH="$(pwd)"

python scripts/compute_hz_metrics_sweep.py \
  --sweep_root runs_sweep_full/mnist_mlp_reg \
  --run_glob "**/MNIST/full/**/final_train" \
  --H_name H.pt \
  --Z_name Z.pt \
  --k 20
"""


from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import torch

from hz_metrics import compute_all_metrics, load_matrix


def _read_json(p: Path) -> Dict[str, Any]:
    with p.open("r", encoding="utf-8") as f:
        return json.load(f)


def infer_lr_wd(run_dir: Path) -> Tuple[Optional[float], Optional[float]]:
    """
    Best-effort:
      1) read config.json and take optimizer.lr / optimizer.weight_decay
      2) parse from path like .../ep40_lr_1e-3_wd_0/...
    """
    cfg = run_dir / "config.json"
    if cfg.exists():
        obj = _read_json(cfg)
        opt = obj.get("optimizer", {}) if isinstance(obj, dict) else {}
        lr = opt.get("lr", None)
        wd = opt.get("weight_decay", None)
        try:
            return (float(lr) if lr is not None else None, float(wd) if wd is not None else None)
        except Exception:
            pass

    m = re.search(r"lr_([^_]+)_wd_([^/]+)", str(run_dir).replace("\\", "/"))
    if m:
        lr_s, wd_s = m.group(1), m.group(2)
        try:
            return float(lr_s), float(wd_s)
        except Exception:
            return None, None

    return None, None


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--sweep_root", type=str, required=True,
                   help="Root folder that contains the ep*_lr_*_wd_* folders.")
    p.add_argument("--run_glob", type=str,
                   default="**/final_train",
                   help="Glob (relative to sweep_root) that matches each run dir containing H/Z and config.json.")
    p.add_argument("--H_name", type=str, default="H.pt", help="Filename of H inside each run dir (or .npy).")
    p.add_argument("--Z_name", type=str, default="Z.pt", help="Filename of Z inside each run dir (or .npy).")
    p.add_argument("--H_key", type=str, default=None, help="If H.pt is a dict, key to select.")
    p.add_argument("--Z_key", type=str, default=None, help="If Z.pt is a dict, key to select.")
    p.add_argument("--out_json", type=str, default="hz_metrics.json", help="Output json name in each run dir.")
    p.add_argument("--out_csv", type=str, default="hz_metrics_summary.csv", help="Summary CSV saved under sweep_root.")
    p.add_argument("--k", type=int, default=20, help="Top-k eigenspace for principal angles.")
    p.add_argument("--eig_tol", type=float, default=1e-12, help="Eigenvalue cutoff for power-law fit.")
    p.add_argument("--alpha_min", type=float, default=0.25)
    p.add_argument("--alpha_max", type=float, default=3.0)
    p.add_argument("--alpha_steps", type=int, default=56)
    p.add_argument("--no_symmetrize", action="store_true", help="Disable (H+H^T)/2 and (Z+Z^T)/2.")
    p.add_argument("--no_allow_indefinite", action="store_true",
                   help="Force PSD projection for Z^alpha (clip negative eigs to 0).")
    args = p.parse_args()

    sweep_root = Path(args.sweep_root).expanduser().resolve()
    run_dirs = sorted(sweep_root.glob(args.run_glob))

    rows: List[Dict[str, Any]] = []

    for rd in run_dirs:
        H_path = rd / args.H_name
        Z_path = rd / args.Z_name
        if not H_path.exists() or not Z_path.exists():
            print(f"[SKIP] Missing H/Z in {rd} (expected {H_path.name}, {Z_path.name})")
            continue

        try:
            H = load_matrix(str(H_path), key=args.H_key)
            Z = load_matrix(str(Z_path), key=args.Z_key)

            metrics = compute_all_metrics(
                H, Z,
                sym=(not args.no_symmetrize),
                k=int(args.k),
                eig_tol=float(args.eig_tol),
                alpha_min=float(args.alpha_min),
                alpha_max=float(args.alpha_max),
                alpha_steps=int(args.alpha_steps),
                allow_indefinite=(not args.no_allow_indefinite),
            )

            lr, wd = infer_lr_wd(rd)

            payload = {
                "run_dir": str(rd),
                "lr": lr,
                "weight_decay": wd,
                **metrics,
            }

            out_path = rd / args.out_json
            out_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
            print(f"[OK] wrote {out_path}")

            # flatten a few key fields for CSV
            row = {
                "run_dir": str(rd),
                "lr": lr,
                "weight_decay": wd,
                "eps_lin": payload["eps_lin"],
                "c_lin": payload["c_lin"],
                "rho_fro": payload["rho_fro"],
                "eps_comm": payload["eps_comm"],
                "pa_mean_cos": payload["principal_angles_topk"]["mean_cos"],
                "pa_min_cos": payload["principal_angles_topk"]["min_cos"],
                "pa_max_angle_deg": payload["principal_angles_topk"]["max_angle_deg"],
                "powerlaw_alpha": payload["powerlaw_eigs"]["alpha"],
                "powerlaw_r2": payload["powerlaw_eigs"]["r2"],
                "powerlaw_c": payload["powerlaw_eigs"]["c"],
                "pow_eps": payload["best_power_fit"]["eps_pow"],
                "pow_alpha": payload["best_power_fit"]["alpha"],
                "pow_c": payload["best_power_fit"]["c"],
            }
            rows.append(row)

        except Exception as e:
            print(f"[FAIL] {rd}: {e}")

    if not rows:
        raise SystemExit("No runs processed (did not find any run dirs with H/Z).")

    df = pd.DataFrame(rows)
    df = df.sort_values(["weight_decay", "lr"], na_position="last")
    out_csv = sweep_root / args.out_csv
    df.to_csv(out_csv, index=False)
    print(f"\nSaved summary CSV: {out_csv}")
    print(df.head(20).to_string(index=False))


if __name__ == "__main__":
    main()