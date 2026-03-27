import numpy as np
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F

# CIFAR-style ResNet with LayerNorm (LN) and 3x3 conv downsample shortcut (stride=2),
# plus a width_multiplier (wm) defaulting to 1.
__all__ = [
    "ResNet",
    "resnet20",
    "resnet32",
    "resnet44",
    "resnet56",
    "resnet110",
    "resnet1202",
]


def _weights_init(m):
    # Convs: He init for ReLU networks (torchvision-style)
    if isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
        if m.bias is not None:
            nn.init.zeros_(m.bias)

    # Linear head: either keep Kaiming or use Xavier; this is a reasonable default
    elif isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)

    # LayerNorm: gamma=1, beta=0
    elif isinstance(m, nn.LayerNorm):
        nn.init.ones_(m.weight)
        nn.init.zeros_(m.bias)

    # If you use the LayerNorm2d wrapper, it contains an nn.LayerNorm as `m.ln`
    elif isinstance(m, LayerNorm2d):
        nn.init.ones_(m.ln.weight)
        nn.init.zeros_(m.ln.bias)



class LayerNorm2d(nn.Module):
    """
    LayerNorm over channels for NCHW tensors.
    Matches Flax LayerNorm behavior when model uses NHWC (normalizes over C).
    """

    def __init__(self, num_channels: int, eps: float = 1e-6, elementwise_affine: bool = True):
        super().__init__()
        self.ln = nn.LayerNorm(num_channels, eps=eps, elementwise_affine=elementwise_affine)

    def forward(self, x):
        # (N, C, H, W) -> (N, H, W, C) -> LN over C -> back
        x = x.permute(0, 2, 3, 1).contiguous()
        x = self.ln(x)
        x = x.permute(0, 3, 1, 2).contiguous()
        return x


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes: int, planes: int, stride: int = 1):
        super().__init__()

        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.norm1 = LayerNorm2d(planes)

        self.conv2 = nn.Conv2d(
            planes, planes, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.norm2 = LayerNorm2d(planes)

        # Match their code: when downsampling / changing channels, use 3x3 conv + LN on shortcut
        if stride != 1 or in_planes != planes:
            if stride != 1:
                assert stride == 2
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False
                ),
                LayerNorm2d(planes),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        y = self.conv1(x)
        y = self.norm1(y)
        y = F.relu(y)

        y = self.conv2(y)
        y = self.norm2(y)

        y = F.relu(y + self.shortcut(x))
        return y


class ResNet(nn.Module):
    """
    CIFAR ResNet (ResNet-20/32/44/56/110/1202) with:
      - LayerNorm (instead of BatchNorm)
      - 3x3 stride-2 conv + LN shortcut when downsampling
      - width_multiplier (wm) to scale channels: (16,32,64) -> (16*wm,32*wm,64*wm)
    """

    def __init__(
        self,
        block,
        num_blocks,
        num_classes: int = 10,
        width_multiplier: int = 1,
    ):
        super().__init__()
        assert width_multiplier >= 1 and isinstance(width_multiplier, int)

        wm = width_multiplier
        self.in_planes = 16 * wm

        self.conv1 = nn.Conv2d(
            3, 16 * wm, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.norm1 = LayerNorm2d(16 * wm)

        self.layer1 = self._make_layer(block, 16 * wm, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 32 * wm, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 64 * wm, num_blocks[2], stride=2)

        self.linear = nn.Linear(64 * wm, num_classes)

        self.apply(_weights_init)
        # Optional: start residual branch at ~0 so the block behaves like identity early on
        for m in self.modules():
            if isinstance(m, BasicBlock):
                nn.init.zeros_(m.norm2.ln.weight)  # gamma of the last norm in the block


    def _make_layer(self, block, planes: int, num_blocks: int, stride: int):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for st in strides:
            layers.append(block(self.in_planes, planes, st))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.norm1(out)
        out = F.relu(out)

        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)

        # Match their reduce(..., "mean") over H,W
        out = out.mean(dim=(2, 3))  # (N, C)

        out = self.linear(out)      # logits (do NOT apply log_softmax if using CrossEntropyLoss)
        return out


def resnet20(num_classes: int = 10, width_multiplier: int = 1):
    return ResNet(BasicBlock, [3, 3, 3], num_classes=num_classes, width_multiplier=width_multiplier)


def resnet32(num_classes: int = 10, width_multiplier: int = 1):
    return ResNet(BasicBlock, [5, 5, 5], num_classes=num_classes, width_multiplier=width_multiplier)


def resnet44(num_classes: int = 10, width_multiplier: int = 1):
    return ResNet(BasicBlock, [7, 7, 7], num_classes=num_classes, width_multiplier=width_multiplier)


def resnet56(num_classes: int = 10, width_multiplier: int = 1):
    return ResNet(BasicBlock, [9, 9, 9], num_classes=num_classes, width_multiplier=width_multiplier)


def resnet110(num_classes: int = 10, width_multiplier: int = 1):
    return ResNet(BasicBlock, [18, 18, 18], num_classes=num_classes, width_multiplier=width_multiplier)


def resnet1202(num_classes: int = 10, width_multiplier: int = 1):
    return ResNet(BasicBlock, [200, 200, 200], num_classes=num_classes, width_multiplier=width_multiplier)


def test(net):
    total_params = 0
    for p in filter(lambda p: p.requires_grad, net.parameters()):
        total_params += int(np.prod(p.detach().cpu().numpy().shape))

    total_layers = len(
        list(filter(lambda p: p.requires_grad and len(p.data.size()) > 1, net.parameters()))
    )

    print("Total number of params", total_params)
    print("Total layers", total_layers)


if __name__ == "__main__":
    for net_name in __all__:
        if net_name.startswith("resnet"):
            print(net_name)
            test(globals()[net_name]())
            print()
