#!/usr/bin/env python3
"""
Quick script to compute H and Z matrices for the MNIST MLP sweep.
"""

import subprocess
import sys

# Command to run
cmd = [
    "python", "CRH/compute_hessian_jacobian_sweep.py",
    "--sweep_root", "runs_sweep_full/mnist_mlp_reg",
    "--run_glob", "**/final_train",
    "--batch_size", "256",
    "--max_samples", "5000",  # Limit to 5000 samples for speed (use None for all)
]

print(f"Running: {' '.join(cmd)}")
print()

result = subprocess.run(cmd, cwd="/Users/antonio2/Bachelor_Thesis")
sys.exit(result.returncode)
