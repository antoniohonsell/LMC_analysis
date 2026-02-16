# LLFC/plot_llfc_mlp_mnist_fmnist.py
import argparse
import os

import torch
import matplotlib.pyplot as plt

# ---- EDIT THIS IMPORT IF YOUR PROJECT USES DIFFERENT MODULE PATHS ----
try:
    import utils
except Exception:
    utils = None


"""
how to run:
python LLFC/plot_llfc_mlp.py \
    --results runs/llfc_mlp_MNIST/llfc_cos_fmnist_mlp.pt \
    --heatmap
"""


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--results", type=str, required=True, help="Path to llfc_cos_*.pt")
    p.add_argument("--out", type=str, default=None, help="Output folder (default: alongside results)")
    p.add_argument("--heatmap", action="store_true", help="Also save layer x lambda heatmap")
    args = p.parse_args()

    data = torch.load(args.results, map_location="cpu")

    out_dir = args.out if args.out is not None else os.path.join(os.path.dirname(args.results), "plots_llfc")
    os.makedirs(out_dir, exist_ok=True)

    if utils is None:
        print("Warning: could not import utils. Plot style will be default matplotlib.")
    else:
        # use your repo’s plotting style hook
        if hasattr(utils, "apply_stitching_trend_style"):
            utils.apply_stitching_trend_style()

    lambdas = data["lambdas"].numpy()
    layers = data["layers"]
    cos_mean = data["cos_mean"].numpy()          # [L, K]
    cos_std = data["cos_std"].numpy()            # [L, K]
    cos_mean_avg = data["cos_mean_layeravg"].numpy()
    cos_std_avg = data["cos_std_layeravg"].numpy()

    title = f"LLFC cosine similarity ({data['arch']}, {data['dataset']})"

    # --- Plot 1: layer-average curve + a few layer curves ---
    plt.figure()
    # thin layer curves
    for i, lname in enumerate(layers):
        plt.plot(lambdas, cos_mean[i], alpha=0.25, linewidth=1.0)

    # layer-average with band
    plt.plot(lambdas, cos_mean_avg, linewidth=2.5, label="Layer-average")
    plt.fill_between(lambdas, cos_mean_avg - cos_std_avg, cos_mean_avg + cos_std_avg, alpha=0.20)

    plt.xlabel(r"$\lambda$")
    plt.ylabel("cosine similarity\n$\\cos(h_\\lambda, \\lambda h_A + (1-\\lambda) h_B)$")
    plt.title(title)
    plt.ylim(-1.05, 1.05)
    plt.legend()
    f1 = os.path.join(out_dir, "llfc_cos_curve.png")
    plt.tight_layout()
    plt.savefig(f1, dpi=200)
    plt.close()
    print(f"Saved: {f1}")

    # --- Plot 2: heatmap (layer x lambda) ---
    if args.heatmap:
        plt.figure()
        im = plt.imshow(cos_mean, aspect="auto", origin="lower",
                        extent=[lambdas[0], lambdas[-1], 0, len(layers) - 1])
        plt.colorbar(im, label="cosine similarity")
        plt.xlabel(r"$\lambda$")
        plt.ylabel("layer index")
        plt.title(title + " (layerwise)")
        f2 = os.path.join(out_dir, "llfc_cos_heatmap.png")
        plt.tight_layout()
        plt.savefig(f2, dpi=200)
        plt.close()
        print(f"Saved: {f2}")


if __name__ == "__main__":
    main()