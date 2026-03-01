
"""
architectures.py

Single, uniform entrypoint for all architectures currently present in the
generalization_experiments repo:

- CIFAR-style ResNet family (resnet20/32/44/56/110/1202) originally in:
  resnet20_arch_BN.py
- ResNet18 CIFAR variant originally in:
  resnet18_arch_BatchNorm.py
- LightNet and LightNet2 originally in:
  resnet18_arch_BatchNorm.py

Design goals:
- One compatible build API: build_model(name, ...)
- Configurable normalization for conv nets via norm="bn" | "ln" | "none"
- CIFAR-ResNet family supports width_multiplier (including for LayerNorm mode)

Notes on "LayerNorm" for conv feature maps:
- True nn.LayerNorm over (C,H,W) requires knowing H,W at construction time.
- A common, resolution-agnostic substitute is GroupNorm(num_groups=1),
  which normalizes per-sample across channels (LN-like behavior for conv).
  This module uses GroupNorm(1, C) when norm="ln".
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------
# Normalization factory
# ---------------------------

def _norm2d(kind: str, num_channels: int) -> nn.Module:
    """
    Returns a normalization module appropriate for 2D conv features.

    kind:
      - "bn": BatchNorm2d
      - "ln": GroupNorm(1, C)  (LN-like for conv features)
      - "none": Identity
    """
    kind = (kind or "bn").lower()
    if kind in ("bn", "batchnorm", "batch_norm"):
        return nn.BatchNorm2d(num_channels)
    
    if kind in ("flax_ln", "ln2d", "layernorm2d", "channel_ln"):
        return LayerNorm2d(num_channels, eps=1e-6)

    # this part here does not match the FLAX implementation of git re-basin paper
    if kind in ("ln", "layernorm", "layer_norm"):
        # LN-like normalization for convs without requiring spatial dimensions.
        return nn.GroupNorm(1, num_channels)
    if kind in ("none", "identity", "no", "null"):
        return nn.Identity()
    raise ValueError(f"Unsupported norm kind: {kind}. Use 'bn', 'ln', or 'none'.")

class LayerNorm2d(nn.Module):
    """
    Channel-only LayerNorm for NCHW tensors.
    Matches Flax LayerNorm() on NHWC feature maps (normalizes over C).
    """
    def __init__(self, num_channels: int, eps: float = 1e-6, affine: bool = True):
        super().__init__()
        self.ln = nn.LayerNorm(num_channels, eps=eps, elementwise_affine=affine)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.permute(0, 2, 3, 1).contiguous()
        x = self.ln(x)
        x = x.permute(0, 3, 1, 2).contiguous()
        return x



# ---------------------------
# CIFAR-ResNet (resnet20 family)
# ---------------------------

def _weights_init(m: nn.Module) -> None:
    # Matches resnet20_arch_BN.py behavior: Kaiming init for Conv2d/Linear
    if isinstance(m, (nn.Linear, nn.Conv2d)):
        nn.init.kaiming_normal_(m.weight)


class LambdaLayer(nn.Module):
    def __init__(self, lambd: Callable[[torch.Tensor], torch.Tensor]):
        super().__init__()
        self.lambd = lambd

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.lambd(x)

class MNISTMLPReg(nn.Module):
    """
    Regularized MNIST MLP (Option A):
      Flatten -> 256 -> 128 -> 10
      ReLU + Dropout after each hidden layer.

    Notes:
      - Designed to generalize well on MNIST without racing to 100% train acc.
      - Uses explicit fc1/fc2/fc3 names (nice for checkpoint readability).
    """
    def __init__(
        self,
        num_classes: int = 10,
        input_shape: Tuple[int, int, int] = (1, 28, 28),
        hidden_dims: Tuple[int, ...] = (256, 128),
        dropout: float = 0.25,
    ):
        super().__init__()
        c, h, w = input_shape
        self._flat = c * h * w

        # Robust: allow hidden_dims of length 0/1/2+
        if hidden_dims is None or len(hidden_dims) == 0:
            h1, h2 = 256, 128
        elif len(hidden_dims) == 1:
            h1 = int(hidden_dims[0])
            h2 = max(16, h1 // 2)
        else:
            h1, h2 = int(hidden_dims[0]), int(hidden_dims[1])

        self.fc1 = nn.Linear(self._flat, h1)
        self.fc2 = nn.Linear(h1, h2)
        self.fc3 = nn.Linear(h2, num_classes)

        p = float(dropout)
        self.drop1 = nn.Dropout(p=p) if p > 0 else nn.Identity()
        self.drop2 = nn.Dropout(p=p) if p > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.view(x.size(0), -1)
        x = self.drop1(F.relu(self.fc1(x)))
        x = self.drop2(F.relu(self.fc2(x)))
        x = self.fc3(x)
        return x


def mnist_mlp_reg(
    num_classes: int = 10,
    *,
    input_shape: Tuple[int, int, int] = (1, 28, 28),
    hidden_dims: Tuple[int, ...] = (256, 128),
    dropout: float = 0.25,
    **_,  # swallow extra kwargs passed by build_model (in_channels/norm/width_multiplier/...)
) -> nn.Module:
    return MNISTMLPReg(
        num_classes=num_classes,
        input_shape=input_shape,
        hidden_dims=hidden_dims,
        dropout=dropout,
    )


class CIFARBasicBlock(nn.Module):
    expansion = 1

    def __init__(
        self,
        in_planes: int,
        planes: int,
        stride: int = 1,
        option: str = "A",
        norm: str = "bn",
    ):
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.n1 = _norm2d(norm, planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.n2 = _norm2d(norm, planes)

        self.shortcut: nn.Module = nn.Identity()
        if stride != 1 or in_planes != planes:
            if option == "A":
                # CIFAR ResNet paper option A: strided spatial downsample + channel pad
                # This expects stage transitions to double planes (which this family does).
                self.shortcut = LambdaLayer(
                    lambda x: F.pad(
                        x[:, :, ::2, ::2],
                        (0, 0, 0, 0, planes // 4, planes // 4),
                        "constant",
                        0,
                    )
                )
            elif option == "B":
                self.shortcut = nn.Sequential(
                    nn.Conv2d(in_planes, planes * self.expansion, kernel_size=1, stride=stride, bias=False),
                    _norm2d(norm, planes * self.expansion),
                )
            elif option == "C":
                # Flax-like: 3x3 stride-2 conv shortcut (+ norm)
                self.shortcut = nn.Sequential(
                    nn.Conv2d(in_planes, planes * self.expansion, kernel_size=3, stride=stride, padding=1, bias=False),
                    _norm2d(norm, planes * self.expansion),
                )
            else:
                raise ValueError(f"Unsupported shortcut option: {option} (expected 'A' or 'B').")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.relu(self.n1(self.conv1(x)))
        out = self.n2(self.conv2(out))
        out = out + self.shortcut(x)
        out = F.relu(out)
        return out


class CIFARResNet(nn.Module):
    """
    CIFAR-style ResNet family used in resnet20_arch_BN.py, generalized to:
      - configurable norm (bn/ln/none)
      - configurable width via width_multiplier

    width_multiplier scales the stage widths:
      base=16*w, then 32*w, 64*w.
    """

    def __init__(
        self,
        num_blocks: Tuple[int, int, int],
        num_classes: int = 10,
        in_channels: int = 3,
        norm: str = "bn",
        width_multiplier: int = 1,
        shortcut_option: str = "A",
    ):
        super().__init__()
        if width_multiplier < 1:
            raise ValueError("width_multiplier must be >= 1")

        base = 16 * width_multiplier
        self.in_planes = base

        self.conv1 = nn.Conv2d(in_channels, base, kernel_size=3, stride=1, padding=1, bias=False)
        self.n1 = _norm2d(norm, base)

        self.layer1 = self._make_layer(base, num_blocks[0], stride=1, option=shortcut_option, norm=norm)
        self.layer2 = self._make_layer(base * 2, num_blocks[1], stride=2, option=shortcut_option, norm=norm)
        self.layer3 = self._make_layer(base * 4, num_blocks[2], stride=2, option=shortcut_option, norm=norm)

        self.linear = nn.Linear(base * 4, num_classes)

        self.apply(_weights_init)
        # Stabilize residual learning (torchvision calls this zero_init_residual).
        # This is especially important for LayerNorm / wide models.
        for m in self.modules():
            if isinstance(m, CIFARBasicBlock):
                n2 = m.n2
                # GroupNorm/BatchNorm have .weight
                if hasattr(n2, "weight") and n2.weight is not None:
                    nn.init.zeros_(n2.weight)
                # Your LayerNorm2d wrapper stores the actual LN as .ln
                elif hasattr(n2, "ln") and hasattr(n2.ln, "weight") and n2.ln.weight is not None:
                    nn.init.zeros_(n2.ln.weight)


    def _make_layer(self, planes: int, num_blocks: int, stride: int, option: str, norm: str) -> nn.Sequential:
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for s in strides:
            layers.append(CIFARBasicBlock(self.in_planes, planes, stride=s, option=option, norm=norm))
            self.in_planes = planes * CIFARBasicBlock.expansion
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.relu(self.n1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        # global average pool
        out = F.avg_pool2d(out, out.size()[3])
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


def resnet20(
    num_classes: int = 10,
    *,
    in_channels: int = 3,
    norm: str = "bn",
    width_multiplier: int = 1,
    shortcut_option: str = "A",
) -> nn.Module:
    return CIFARResNet(
        (3, 3, 3),
        num_classes=num_classes,
        in_channels=in_channels,
        norm=norm,
        width_multiplier=width_multiplier,
        shortcut_option=shortcut_option,
    )

def resnet32(num_classes: int = 10, *, in_channels: int = 3, norm: str = "bn", width_multiplier: int = 1) -> nn.Module:
    return CIFARResNet((5, 5, 5), num_classes=num_classes, in_channels=in_channels, norm=norm, width_multiplier=width_multiplier)

def resnet44(num_classes: int = 10, *, in_channels: int = 3, norm: str = "bn", width_multiplier: int = 1) -> nn.Module:
    return CIFARResNet((7, 7, 7), num_classes=num_classes, in_channels=in_channels, norm=norm, width_multiplier=width_multiplier)

def resnet56(num_classes: int = 10, *, in_channels: int = 3, norm: str = "bn", width_multiplier: int = 1) -> nn.Module:
    return CIFARResNet((9, 9, 9), num_classes=num_classes, in_channels=in_channels, norm=norm, width_multiplier=width_multiplier)

def resnet110(num_classes: int = 10, *, in_channels: int = 3, norm: str = "bn", width_multiplier: int = 1) -> nn.Module:
    return CIFARResNet((18, 18, 18), num_classes=num_classes, in_channels=in_channels, norm=norm, width_multiplier=width_multiplier)

def resnet1202(num_classes: int = 10, *, in_channels: int = 3, norm: str = "bn", width_multiplier: int = 1) -> nn.Module:
    return CIFARResNet((200, 200, 200), num_classes=num_classes, in_channels=in_channels, norm=norm, width_multiplier=width_multiplier)


# ---------------------------
# ResNet18 (torchvision-style) with CIFAR-friendly conv1
# ---------------------------

def conv3x3(in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, dilation: int = 1) -> nn.Conv2d:
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=dilation,
        groups=groups,
        bias=False,
        dilation=dilation,
    )

def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class TVBasicBlock(nn.Module):
    expansion = 1

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Optional[Callable[[int], nn.Module]] = None,
    ):
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError("TVBasicBlock only supports groups=1 and base_width=64")
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in TVBasicBlock")

        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out = out + identity
        out = self.relu(out)
        return out


class TVBottleneck(nn.Module):
    expansion = 4

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Optional[Callable[[int], nn.Module]] = None,
    ):
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.0)) * groups

        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out = out + identity
        out = self.relu(out)
        return out


class ResNet18CIFAR(nn.Module):
    """
    Minimal re-wrap of the repo's ResNet (torchvision-style), but with:
      - configurable norm (bn/ln/none) via norm_layer factory
      - optional CIFAR-friendly conv1 (3x3, stride 1, padding 1)
    """

    def __init__(
        self,
        num_classes: int = 10,
        in_channels: int = 3,
        norm: str = "bn",
        cifar_conv1: bool = True,
        zero_init_residual: bool = False,
    ):
        super().__init__()

        def norm_layer(c: int) -> nn.Module:
            return _norm2d(norm, c)

        self._norm_layer = norm_layer
        self.inplanes = 64
        self.dilation = 1
        self.groups = 1
        self.base_width = 64

        # Keep original defaults; optionally swap to CIFAR-friendly conv1.
        if cifar_conv1:
            self.conv1 = nn.Conv2d(in_channels, self.inplanes, kernel_size=3, stride=1, padding=1, bias=False)
            self.maxpool = nn.Identity()
        else:
            self.conv1 = nn.Conv2d(in_channels, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
            self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)

        self.layer1 = self._make_layer(TVBasicBlock, 64, 2)
        self.layer2 = self._make_layer(TVBasicBlock, 128, 2, stride=2)
        self.layer3 = self._make_layer(TVBasicBlock, 256, 2, stride=2)
        self.layer4 = self._make_layer(TVBasicBlock, 512, 2, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * TVBasicBlock.expansion, num_classes)

        # Init matches torchvision pattern used in the repo file.
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                if hasattr(m, "weight") and m.weight is not None:
                    nn.init.constant_(m.weight, 1)
                if hasattr(m, "bias") and m.bias is not None:
                    nn.init.constant_(m.bias, 0)

        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, TVBottleneck):
                    nn.init.constant_(m.bn3.weight, 0)  # type: ignore[attr-defined]
                elif isinstance(m, TVBasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)  # type: ignore[attr-defined]

    def _make_layer(self, block, planes: int, blocks: int, stride: int = 1) -> nn.Sequential:
        norm_layer = self._norm_layer
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups, self.base_width, self.dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups, base_width=self.base_width, dilation=self.dilation, norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


def resnet18(num_classes: int = 10, *, in_channels: int = 3, norm: str = "bn") -> nn.Module:
    # Alias to the CIFAR-friendly version to match the repo's usage pattern.
    return ResNet18CIFAR(num_classes=num_classes, in_channels=in_channels, norm=norm, cifar_conv1=True)


# ---------------------------
# LightNet family (from resnet18_arch_BatchNorm.py)
# ---------------------------

class MLP(nn.Module):
    """
    3 hidden layers of 512 unit each. ReLU activations, no normalization.
    """
    def __init__(self, num_classes: int = 10, input_shape: Tuple[int, int, int] = (1, 28, 28), hidden: int = 512):
        super().__init__()
        c, h, w = input_shape
        self._flat = c * h * w
        self.fc1 = nn.Linear(self._flat, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.fc3 = nn.Linear(hidden, hidden)
        self.fc4 = nn.Linear(hidden, num_classes)
        

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x) 
        return x

class LightNet(nn.Module):
    """
    Simple 2-layer MLP (MNIST-style default).
    Original expects inputs shaped [N, 1, 28, 28].
    """
    def __init__(self, num_classes: int = 10, input_shape: Tuple[int, int, int] = (1, 28, 28), hidden: int = 512):
        super().__init__()
        c, h, w = input_shape
        self._flat = c * h * w
        self.fc1 = nn.Linear(self._flat, hidden)
        self.fc2 = nn.Linear(hidden, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class LightNet2(nn.Module):
    """
    Tiny conv net (MNIST-style default).
    Original expects inputs shaped [N, 1, 28, 28].
    """
    def __init__(self, num_classes: int = 10, in_channels: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, 6, 3, padding="same")
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(6 * 14 * 14, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(F.relu(self.conv1(x)))
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        return x


def lightnet(num_classes: int = 10, *, input_shape: Tuple[int, int, int] = (1, 28, 28), hidden: int = 512) -> nn.Module:
    return LightNet(num_classes=num_classes, input_shape=input_shape, hidden=hidden)

def lightnet2(num_classes: int = 10, *, in_channels: int = 1) -> nn.Module:
    return LightNet2(num_classes=num_classes, in_channels=in_channels)

def mlp(num_classes: int = 10, *, input_shape=(1, 28, 28), hidden: int = 512, **_) -> nn.Module:
    return MLP(num_classes=num_classes, input_shape=input_shape, hidden=hidden)


# ---------------------------
# Registry / public API
# ---------------------------

_MODEL_REGISTRY: Dict[str, Callable[..., nn.Module]] = {
    # CIFAR ResNet family
    "resnet20": resnet20,
    "resnet32": resnet32,
    "resnet44": resnet44,
    "resnet56": resnet56,
    "resnet110": resnet110,
    "resnet1202": resnet1202,
    # ResNet18
    "resnet18": resnet18,
    "resnet_18_cifar": resnet18,  # legacy alias
    # LightNets
    "lightnet": lightnet,
    "lightnet2": lightnet2,
    "LightNet": lightnet,     # legacy-ish alias
    "LightNet2": lightnet2,   # legacy-ish alias
    "MLP": mlp,
    "mlp": mlp,
    # Regularized MNIST MLP (Option A)
    "mnist_mlp_reg": mnist_mlp_reg,
    "MNIST_MLP_REG": mnist_mlp_reg,  # optional alias
}


def available_models() -> Tuple[str, ...]:
    return tuple(sorted(_MODEL_REGISTRY.keys()))


def build_model(
    name: str,
    *,
    num_classes: int,
    in_channels: int = 3,
    norm: str = "bn",
    width_multiplier: int = 1,
    **kwargs,
) -> nn.Module:
    """
    Uniform constructor for all models.

    Common kwargs:
      - num_classes (required)
      - in_channels (default 3)
      - norm: "bn" | "ln" | "none"
      - width_multiplier: only used by CIFAR ResNet family

    Any additional kwargs are forwarded to the underlying builder when accepted.
    """
    if not isinstance(name, str):
        raise TypeError("name must be a string")
    key = name.strip()
    if key not in _MODEL_REGISTRY:
        raise KeyError(f"Unknown model '{name}'. Available: {available_models()}")

    builder = _MODEL_REGISTRY[key]

    # Best-effort forwarding: pass only what the builder is likely to accept.
    # (We keep this lightweight and explicit for repo compatibility.)
    if builder in (resnet20, resnet32, resnet44, resnet56, resnet110, resnet1202):
        return builder(num_classes=num_classes, in_channels=in_channels, norm=norm, width_multiplier=width_multiplier, **kwargs)
    if builder is resnet18:
        return builder(num_classes=num_classes, in_channels=in_channels, norm=norm, **kwargs)
    if builder is lightnet:
        # LightNet uses input_shape instead of in_channels
        input_shape = kwargs.pop("input_shape", (1, 28, 28))
        hidden = kwargs.pop("hidden", 512)
        return builder(num_classes=num_classes, input_shape=input_shape, hidden=hidden)
    if builder is lightnet2:
        # LightNet2 uses in_channels default 1
        return builder(num_classes=num_classes, in_channels=kwargs.pop("in_channels", 1))

    # Fallback (shouldn't be needed, but keep robust)
    return builder(num_classes=num_classes, in_channels=in_channels, norm=norm, width_multiplier=width_multiplier, **kwargs)


__all__ = [
    "build_model",
    "available_models",
    # CIFAR ResNets
    "CIFARResNet",
    "resnet20", "resnet32", "resnet44", "resnet56", "resnet110", "resnet1202",
    # ResNet18
    "ResNet18CIFAR",
    "resnet18",
    # LightNets
    "LightNet", "LightNet2",
    "lightnet", "lightnet2",
]



"""
USAGE :
from unified_architectures import build_model

# CIFAR ResNet20 with BatchNorm
m1 = build_model("resnet20", num_classes=100, norm="bn", width_multiplier=1)

# CIFAR ResNet20 with LayerNorm-style norm + width scaling
m2 = build_model("resnet20", num_classes=100, norm="ln", width_multiplier=2)

# ResNet18 CIFAR variant (conv1=3x3 stride1, no maxpool)
m3 = build_model("resnet18", num_classes=100, norm="bn")

# LightNets
m4 = build_model("lightnet",  num_classes=10)                 # default MNIST shape
m5 = build_model("lightnet2", num_classes=10, in_channels=1)  # MNIST-style conv net

"""