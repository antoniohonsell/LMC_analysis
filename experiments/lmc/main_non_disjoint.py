import os
import random
import argparse
import sys
import numpy as np
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_SRC  = _HERE.parent.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import random_split

import old_architectures.resnet18_arch_BatchNorm as resnet18_arch_BatchNorm
import old_architectures.resnet20_arch_BatchNorm as resnet20_arch_BatchNorm
import old_architectures.resnet20_arch_LayerNorm as resnet20_arch_LayerNorm
import train_loop
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


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # Determinism knobs (best-effort; some backends/ops may still be nondeterministic)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def seed_worker(worker_id: int):
    # Ensures dataloader workers are deterministically seeded
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def lr_lambda_10(epoch):
    if epoch < 80-1:
        return 1.0
    elif epoch < 120-1:
        return 0.1
    else:
        return 0.01
    
def lr_lambda_100(epoch):
    if epoch < 100-1:
        return 1.0
    elif epoch < 150-1:
        return 0.1
    else:
        return 0.01

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="CIFAR10")
    parser.add_argument("--model", type=str, default="resnet20")
    parser.add_argument("--seeds", nargs="+", type=int, default=list(range(0,2)))
    parser.add_argument("--split_seed", type=int, default=50)   # fixed split across all runs
    parser.add_argument("--epochs", type=int, default=128)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--num_workers", type=int, default=0)   
    parser.add_argument("--out_dir", type=str, default=None)
    args = parser.parse_args()
    if args.out_dir is None:
        args.out_dir = f"./runs_{args.model}_{args.dataset}"

    device = utils.get_device()
    os.makedirs(args.out_dir, exist_ok=True)


    if args.dataset not in DATASET_STATS:
        raise ValueError(f"Unsupported dataset: {args.dataset}")
    stats = DATASET_STATS[args.dataset]
    normalize = transforms.Normalize(stats["mean"], stats["std"])

    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ])

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        normalize,
    ])

    if args.dataset == "CIFAR10":
        ds_train_full = torchvision.datasets.CIFAR10(root="./data", train=True, download=True, transform=train_transform)
        ds_eval_full  = torchvision.datasets.CIFAR10(root="./data", train=True, download=True, transform=test_transform)
        test_ds       = torchvision.datasets.CIFAR10(root="./data", train=False, download=True, transform=test_transform)
    elif args.dataset == "CIFAR100":
        ds_train_full = torchvision.datasets.CIFAR100(root="./data", train=True, download=True, transform=train_transform)
        ds_eval_full  = torchvision.datasets.CIFAR100(root="./data", train=True, download=True, transform=test_transform)
        test_ds       = torchvision.datasets.CIFAR100(root="./data", train=False, download=True, transform=test_transform)
    else:
        raise ValueError(f"Unsupported dataset: {args.dataset}")

    val_size = 5000
    train_size = len(ds_train_full) - val_size
    g_split = torch.Generator().manual_seed(args.split_seed)

    # split indices ONCE using a dummy split of indices
    all_indices = list(range(len(ds_train_full)))
    train_idx, val_idx = random_split(all_indices, [train_size, val_size], generator=g_split)

    # Subset datasets with appropriate transforms
    train_ds = torch.utils.data.Subset(ds_train_full, train_idx.indices)
    val_ds   = torch.utils.data.Subset(ds_eval_full,  val_idx.indices)


    # Save split indices once for reproducibility in later analysis
    split_path = os.path.join(args.out_dir, f"split_indices_{args.dataset}_seed{args.split_seed}.pt")
    if not os.path.exists(split_path):
        torch.save({"train_indices": train_idx.indices, "val_indices": val_idx.indices}, split_path)


    # Test loader never shuffled
    test_loader = torch.utils.data.DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )

    num_classes = 100 if args.dataset == "CIFAR100" else 10

    if args.model == "resnet18":
        model = resnet18_arch_BatchNorm.resnet_18_cifar(num_classes=num_classes)
    elif args.model == "resnet20":
        model = resnet20_arch_LayerNorm.resnet20(num_classes=num_classes)
    else:
        raise ValueError(f"Unsupported model: {args.model}")

    model = model.to(device)


    for seed in args.seeds:
        print(f"\n==============================\nRunning seed = {seed}\n==============================")
        set_seed(seed)

        run_dir = os.path.join(args.out_dir, f"seed_{seed}")
        os.makedirs(run_dir, exist_ok=True)

        # Per-seed generator controls shuffle order deterministically
        g_loader = torch.Generator().manual_seed(seed)

        train_loader = torch.utils.data.DataLoader(
            train_ds,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            worker_init_fn=seed_worker if args.num_workers > 0 else None,
            generator=g_loader,
            pin_memory=(device.type == "cuda"),
        )

        val_loader = torch.utils.data.DataLoader(
            val_ds,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=(device.type == "cuda"),
        )

        if args.model == "resnet18":
            model = resnet18_arch_BatchNorm.resnet_18_cifar(num_classes=num_classes)
        elif args.model == "resnet20":
            model = resnet20_arch_LayerNorm.resnet20(num_classes=num_classes)
        else:
            raise ValueError(f"Unsupported model: {args.model}")

        model = model.to(device)
        criterion = nn.CrossEntropyLoss()

        decay, no_decay = [], []
        for name, p in model.named_parameters():
            if not p.requires_grad:
                continue
            # no weight decay for LayerNorm params and biases
            if p.ndim == 1 or name.endswith(".bias") or "norm" in name or "ln" in name:
                no_decay.append(p)
            else:
                decay.append(p)

        optimizer = optim.SGD(
            [
                {"params": decay, "weight_decay": 5e-4},
                {"params": no_decay, "weight_decay": 0.0},
            ],
            lr=args.lr,
            momentum=0.9,
        )

        scheduler = LambdaLR(optimizer, lr_lambda=lr_lambda_10 if args.epochs==128 else lr_lambda_100)




        history = train_loop.train(
            model=model,
            criterion=criterion,
            optimizer=optimizer,
            scheduler=scheduler,
            train_loader=train_loader,
            val_loader=val_loader,
            epochs=args.epochs,
            device=device,
            save_dir=run_dir,
            run_name=f"{args.model}_{args.dataset}_seed{seed}",
            save_every=1,
            save_last=True,
        )

        # Also save a lightweight summary file
        torch.save(
            {"seed": seed, "history": history},
            os.path.join(run_dir, "history.pt"),
        )

        # Optional: evaluate on test after training (you already created test_loader)
        # If you want, you can add a small test() helper similar to validate().


if __name__ == "__main__":
    main()
