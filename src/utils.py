import torch
from typing import List

_DEEP_PALETTE: List[str] = ["#4c72b0", "#dd8452", "#55a868", "#c44e52", "#8172b3", "#937860"]
_PLOT_STYLE_APPLIED: bool = False


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available() and torch.backends.mps.is_built():
        return torch.device("mps")
    return torch.device("cpu")


def get_deep_palette() -> List[str]:
    return list(_DEEP_PALETTE)


def apply_stitching_trend_style(*, force: bool = False) -> None:
    global _PLOT_STYLE_APPLIED
    if _PLOT_STYLE_APPLIED and not force:
        return

    import matplotlib as mpl
    import matplotlib.pyplot as plt

    try:
        plt.style.use("seaborn-v0_8-whitegrid")
    except OSError:
        plt.style.use("seaborn-whitegrid")

    mpl.rcParams["axes.prop_cycle"] = mpl.cycler(color=_DEEP_PALETTE)

    mpl.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "#f6efe7",
            "grid.color": "#a7a29f",
            "grid.linewidth": 1.2,
            "grid.alpha": 1.0,
            "axes.edgecolor": "black",
            "axes.linewidth": 2.6,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.labelweight": "bold",
            "axes.labelsize": 22,
            "xtick.labelsize": 14,
            "ytick.labelsize": 14,
            "legend.framealpha": 1.0,
            "legend.facecolor": "white",
            "legend.edgecolor": "#808080",
            "legend.fontsize": 14,
        }
    )

    _PLOT_STYLE_APPLIED = True