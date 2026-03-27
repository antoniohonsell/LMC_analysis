import os
import glob
import argparse
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_SRC  = _HERE.parent.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
import torchvision
import torchvision.transforms as transforms
from tqdm.auto import tqdm

import metrics_platonic as metrics
from architectures import build_model
import utils


DATASET_STATS = {
    "CIFAR10": {
        "mean": (0.49139968, 0.48215841, 0.44653091),
        "std":  (0.24703223, 0.24348513, 0.26158784),
    },
    "CIFAR100": {
        "mean": (0.50707516, 0.48654887, 0.44091784),
        "std":  (0.26733429, 0.25643846, 0.27615047),
    },
}


def load_resnet_from_ckpt(ckpt_path: str, device: torch.device, num_classes: int):
    ckpt = torch.load(ckpt_path, map_location=device)

    state_dict = ckpt["state_dict"]
    if any(k.startswith("module.") for k in state_dict.keys()):
        state_dict = {k.replace("module.", "", 1): v for k, v in state_dict.items()}

    # Heuristic: your checkpoints contain keys like "norm1.ln.weight"
    is_old_ln_ckpt = any(".ln." in k and k.startswith(("norm1", "layer")) for k in state_dict.keys())

    if is_old_ln_ckpt:
        import old_architectures.resnet20_arch_LayerNorm as rn20_ln
        model = rn20_ln.resnet20(num_classes=num_classes)  # add width_multiplier=... if you trained with it
    else:
        model = build_model("resnet20", num_classes=num_classes, norm="bn")

    model.load_state_dict(state_dict, strict=True)
    model.to(device).eval()
    return model



def _pick_latest(path_glob: str) -> str:
    cand = glob.glob(path_glob)
    if len(cand) == 0:
        raise FileNotFoundError(f"No files match: {path_glob}")
    return max(cand, key=lambda p: os.path.getmtime(p))


def build_cifar_loader(
    split: str,
    runs_dir: str,
    split_seed: int,
    batch_size: int,
    num_workers: int,
    data_root: str = "./data",
    dataset: str = "CIFAR10",
    disjoint: bool = False,
    indices_path: str | None = None,
):
    # deterministic transform (do NOT use train-time random aug for representation comparison)
    if dataset not in DATASET_STATS:
        raise ValueError(f"Unsupported dataset: {dataset}")
    stats = DATASET_STATS[dataset]
    tfm = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(stats["mean"], stats["std"]),
    ])

    if dataset == "CIFAR10":
        DS = torchvision.datasets.CIFAR10
    elif dataset == "CIFAR100":
        DS = torchvision.datasets.CIFAR100
    else:
        raise ValueError(f"Unsupported dataset: {dataset}")

    if split == "test":
        ds = DS(root=data_root, train=False, download=True, transform=tfm)

    elif split in ("train", "val"):
        full = DS(root=data_root, train=True, download=True, transform=tfm)

        if not disjoint:
            # produced by main_non_disjoint.py
            split_path = os.path.join(runs_dir, f"split_indices_{dataset}_seed{split_seed}.pt")
            if not os.path.exists(split_path):
                raise FileNotFoundError(
                    f"Missing split file: {split_path}\n"
                    f"Run main_non_disjoint.py (or point --runs_dir to the correct folder)."
                )
        else:
            # produced by main_disjoint.py:
            # indices_{dataset}_splitseed{split_seed}_subsetseed{...}_val{...}.pt
            if indices_path is not None:
                split_path = indices_path
                if not os.path.exists(split_path):
                    raise FileNotFoundError(f"--indices_path does not exist: {split_path}")
            else:
                split_path = _pick_latest(os.path.join(runs_dir, f"indices_{dataset}_splitseed{split_seed}_*.pt"))

        idx = torch.load(split_path)

        # Important for comparability: both models should be evaluated on the SAME examples & order.
        # For disjoint training, 'train_indices' is the shared train split (union) from main_disjoint.py.
        indices = idx["train_indices"] if split == "train" else idx["val_indices"]
        ds = Subset(full, indices)

        print(f"[data] using indices file: {split_path}")
        print(f"[data] split={split} | N={len(indices)} | disjoint={disjoint}")

    else:
        raise ValueError(f"Unknown split: {split}")

    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,  # critical: keep identical ordering across models
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )


def extract_layer_features(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    layers: list[str],
    pool: str = "avg",          # avg = spatial mean for conv outputs
    max_samples: int | None = None,
):
    named_modules = dict(model.named_modules())
    for ln in layers:
        if ln not in named_modules:
            raise ValueError(f"Layer '{ln}' not found. Available: {list(named_modules.keys())[:20]} ...")

    buckets = {ln: [] for ln in layers}
    handles = []

    def make_hook(layer_name: str):
        def hook(_module, _inp, out):
            x = out
            if isinstance(x, (tuple, list)):
                x = x[0]

            if x.ndim == 4:  # B,C,H,W
                if pool == "avg":
                    x = x.mean(dim=(2, 3))
                elif pool == "flatten":
                    x = x.flatten(1)
                else:
                    raise ValueError(f"Unknown pool: {pool}")
            else:
                x = x.flatten(1)

            buckets[layer_name].append(x.detach().cpu())
        return hook

    for ln in layers:
        handles.append(named_modules[ln].register_forward_hook(make_hook(ln)))

    n_seen = 0
    model.eval()
    with torch.no_grad():
        for xb, _yb in tqdm(loader, desc="Extracting features", leave=False):
            xb = xb.to(device)
            _ = model(xb)
            n_seen += xb.size(0)
            if max_samples is not None and n_seen >= max_samples:
                break

    for h in handles:
        h.remove()

    feats = []
    for ln in layers:
        f = torch.cat(buckets[ln], dim=0)
        if max_samples is not None:
            f = f[:max_samples]
        feats.append(f)
    return feats  # list of [N,D] CPU tensors


def prepare_features(feats: torch.Tensor, device: torch.device, q: float = 0.95, exact: bool = False):
    feats = metrics.remove_outliers(feats.float(), q=q, exact=exact)
    return feats.to(device)


def layerwise_scores(
    feats_A: list[torch.Tensor],
    feats_B: list[torch.Tensor],
    device: torch.device,
    metric: str,
    topk: int,
    q: float,
    exact: bool,
    normalize: bool = True,
    pairing: str = "diagonal",   # diagonal or best
):
    if pairing not in ("diagonal", "best"):
        raise ValueError("pairing must be 'diagonal' or 'best'")

    # Move + preprocess once (still modest size at CIFAR layer dims)
    A = [prepare_features(f, device=device, q=q, exact=exact) for f in feats_A]
    B = [prepare_features(f, device=device, q=q, exact=exact) for f in feats_B]

    if normalize:
        A = [F.normalize(f, p=2, dim=-1) for f in A]
        B = [F.normalize(f, p=2, dim=-1) for f in B]

    def score(fa, fb):
        kwargs = {"topk": topk} if "knn" in metric else {}
        return metrics.AlignmentMetrics.measure(metric, fa, fb, **kwargs)

    if pairing == "diagonal":
        assert len(A) == len(B), "Diagonal pairing requires same number of layers"
        scores = [score(A[i], B[i]) for i in range(len(A))]
        return np.array(scores, dtype=float), None

    # pairing == "best": search best (i,j)
    best = -1e9
    best_ij = None
    for i in range(len(A)):
        for j in range(len(B)):
            s = score(A[i], B[j])
            if s > best:
                best = s
                best_ij = (i, j)
    return np.array([best], dtype=float), best_ij


def main():
    parser = argparse.ArgumentParser()

    # Dataset
    parser.add_argument("--dataset", type=str, default="CIFAR10", choices=["CIFAR10", "CIFAR100"])

    # Provide either explicit checkpoints OR seeds + runs_dir + which
    parser.add_argument("--ckpt_a", type=str, default=None)
    parser.add_argument("--ckpt_b", type=str, default=None)
    parser.add_argument("--seed_a", type=int, default=None)
    parser.add_argument("--seed_b", type=int, default=None)

    parser.add_argument("--runs_dir", type=str, default="./runs_resnet20_CIFAR10")
    parser.add_argument("--which", type=str, default="best", choices=["best", "final"])

    # Disjoint A/B (for main_disjoint.py checkpoints + indices)
    parser.add_argument("--disjoint", action="store_true",
                        help="Use disjoint run layout (seed_x/subset_A|B) + indices_{dataset}_splitseed*.pt")
    parser.add_argument("--subset_a", type=str, default=None, choices=["A", "B"],
                        help="Subset for model A (only with --disjoint)")
    parser.add_argument("--subset_b", type=str, default=None, choices=["A", "B"],
                        help="Subset for model B (only with --disjoint)")
    parser.add_argument("--indices_path", type=str, default=None,
                        help="Optional explicit path to indices_*.pt (only used with --disjoint)")

    parser.add_argument("--split", type=str, default="val", choices=["train", "val", "test"])
    parser.add_argument("--split_seed", type=int, default=50)

    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--max_samples", type=int, default=None)

    parser.add_argument("--layers", nargs="+", default=["norm1", "layer1", "layer2", "layer3", "linear"])
    parser.add_argument("--pool", type=str, default="avg", choices=["avg", "flatten"])

    parser.add_argument("--metric", type=str, default="mutual_knn",
                        choices=["mutual_knn", "cka", "unbiased_cka", "cknna"])
    parser.add_argument("--topk", type=int, default=10)
    parser.add_argument("--pairing", type=str, default="diagonal", choices=["diagonal", "best"])

    parser.add_argument("--q", type=float, default=0.95)         # outlier clamp quantile
    parser.add_argument("--exact", action="store_true")          # exact quantile
    parser.add_argument("--no_normalize", action="store_true")   # skip L2 normalize
    parser.add_argument("--output_dir", type=str, default="./results/alignment_resnet")
    args = parser.parse_args()

    device = utils.get_device()

    # Resolve checkpoint paths
    if args.ckpt_a is None or args.ckpt_b is None:
        if args.seed_a is None or args.seed_b is None:
            raise ValueError("Provide either --ckpt_a/--ckpt_b or --seed_a/--seed_b")

        if args.disjoint:
            if args.subset_a is None or args.subset_b is None:
                raise ValueError("With --disjoint, you must provide --subset_a and --subset_b (A/B).")

            def ckpt_from_seed_subset(seed: int, subset: str):
                run_dir = os.path.join(args.runs_dir, f"seed_{seed}", f"subset_{subset}")
                fname = f"resnet20_{args.dataset}_seed{seed}_subset{subset}_{args.which}.pth"
                return os.path.join(run_dir, fname)

            ckpt_a = ckpt_from_seed_subset(args.seed_a, args.subset_a)
            ckpt_b = ckpt_from_seed_subset(args.seed_b, args.subset_b)
        else:
            def ckpt_from_seed(seed: int):
                run_dir = os.path.join(args.runs_dir, f"seed_{seed}")
                fname = f"resnet20_{args.dataset}_seed{seed}_{args.which}.pth"
                return os.path.join(run_dir, fname)

            ckpt_a = ckpt_from_seed(args.seed_a)
            ckpt_b = ckpt_from_seed(args.seed_b)
    else:
        ckpt_a, ckpt_b = args.ckpt_a, args.ckpt_b

    assert os.path.exists(ckpt_a), ckpt_a
    assert os.path.exists(ckpt_b), ckpt_b

    # Data
    loader = build_cifar_loader(
        split=args.split,
        runs_dir=args.runs_dir,
        split_seed=args.split_seed,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        dataset=args.dataset,
        disjoint=args.disjoint,
        indices_path=args.indices_path,
    )

    # Models
    num_classes = 10 if args.dataset == "CIFAR10" else 100
    model_a = load_resnet_from_ckpt(ckpt_a, device=device, num_classes=num_classes)
    model_b = load_resnet_from_ckpt(ckpt_b, device=device, num_classes=num_classes)

    # Features
    feats_a = extract_layer_features(model_a, loader, device, layers=args.layers,
                                     pool=args.pool, max_samples=args.max_samples)
    feats_b = extract_layer_features(model_b, loader, device, layers=args.layers,
                                     pool=args.pool, max_samples=args.max_samples)

    # Scores
    scores, best_ij = layerwise_scores(
        feats_a, feats_b,
        device=device,
        metric=args.metric,
        topk=args.topk,
        q=args.q,
        exact=args.exact,
        normalize=(not args.no_normalize),
        pairing=args.pairing,
    )

    # Report
    print(f"ckpt_a: {ckpt_a}")
    print(f"ckpt_b: {ckpt_b}")
    print(f"dataset: {args.dataset} | split: {args.split} | metric: {args.metric} | pairing: {args.pairing}")
    if "knn" in args.metric:
        print(f"topk: {args.topk}")
    if args.disjoint:
        print(f"disjoint: True | subsets: A={args.subset_a} B={args.subset_b} | split_seed={args.split_seed}")

    if args.pairing == "diagonal":
        for i, ln in enumerate(args.layers):
            print(f"{ln:>10}: {scores[i]:.6f}")
        print(f"{'mean':>10}: {scores.mean():.6f}")
    else:
        print(f"best_score: {scores[0]:.6f} at layers (a,b) = {best_ij}")

    # Save (include topk in filename for knn metrics to avoid overwriting)
    os.makedirs(args.output_dir, exist_ok=True)
    base_a = os.path.basename(ckpt_a).replace(".pth", "")
    base_b = os.path.basename(ckpt_b).replace(".pth", "")

    metric_tag = args.metric
    if "knn" in args.metric:
        metric_tag = f"{args.metric}_topk{args.topk}"

    out_name = f"{base_a}__vs__{base_b}__{args.split}__{metric_tag}__{args.pairing}.npz"
    out_path = os.path.join(args.output_dir, out_name)

    np.savez(
        out_path,
        scores=scores,
        best_ij=np.array(best_ij) if best_ij is not None else None,
        layers=np.array(args.layers),
        ckpt_a=ckpt_a,
        ckpt_b=ckpt_b,
        dataset=args.dataset,
        split=args.split,
        metric=args.metric,
        metric_tag=metric_tag,
        pairing=args.pairing,
        topk=args.topk,
        q=args.q,
        pool=args.pool,
        max_samples=args.max_samples,
        disjoint=args.disjoint,
        split_seed=args.split_seed,
        runs_dir=args.runs_dir,
        subset_a=args.subset_a,
        subset_b=args.subset_b,
        indices_path=args.indices_path,
    )
    print(f"saved: {out_path}")


if __name__ == "__main__":
    main()
