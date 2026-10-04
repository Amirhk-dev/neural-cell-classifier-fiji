"""Cell classifier supporting ResNet-18 and MobileNetV3-Small backbones."""

from __future__ import annotations

import torch
import torch.nn as nn
from torchvision import models

SUPPORTED_ARCHITECTURES = ("resnet18", "mobilenet_v3_small")


def _adapt_first_conv(conv: nn.Conv2d, input_channels: int, pretrained: bool) -> nn.Conv2d:
    """Return a new Conv2d adapted to input_channels, with weights initialised from conv."""
    new_conv = nn.Conv2d(
        input_channels,
        conv.out_channels,
        kernel_size=conv.kernel_size,
        stride=conv.stride,
        padding=conv.padding,
        bias=conv.bias is not None,
    )
    if pretrained:
        with torch.no_grad():
            mean_w = conv.weight.data.mean(dim=1, keepdim=True)  # (C_out, 1, kH, kW)
            if input_channels == conv.in_channels:
                new_conv.weight.data = conv.weight.data.clone()
            elif input_channels == 4 and conv.in_channels == 3:
                # Append one extra channel initialised as the mean of RGB
                new_conv.weight.data = torch.cat([conv.weight.data, mean_w], dim=1)
            else:
                new_conv.weight.data = mean_w.repeat(1, input_channels, 1, 1)
    return new_conv


def create_resnet18_classifier(
    input_channels: int = 3,
    num_classes: int = 2,
    pretrained: bool = True,
    dropout: float = 0.5,
    freeze_backbone: bool = False,
) -> nn.Module:
    """ResNet-18 with modified input/output layers."""
    weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
    model = models.resnet18(weights=weights)

    if input_channels != 3:
        model.conv1 = _adapt_first_conv(model.conv1, input_channels, pretrained)

    if freeze_backbone:
        for param in model.parameters():
            param.requires_grad = False

    in_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(p=dropout),
        nn.Linear(in_features, num_classes),
    )
    for param in model.fc.parameters():
        param.requires_grad = True

    return model


def create_mobilenet_v3_small_classifier(
    input_channels: int = 3,
    num_classes: int = 2,
    pretrained: bool = True,
    dropout: float = 0.5,
    freeze_backbone: bool = False,
) -> nn.Module:
    """MobileNetV3-Small with modified input/output layers.

    ~2.5 M parameters total; designed for CPU / edge inference.
    With freeze_backbone=True only the ~2 K-param head is trained.
    """
    weights = models.MobileNet_V3_Small_Weights.IMAGENET1K_V1 if pretrained else None
    model = models.mobilenet_v3_small(weights=weights)

    # Adapt first conv (model.features[0][0] is the Conv2d)
    if input_channels != 3:
        model.features[0][0] = _adapt_first_conv(
            model.features[0][0], input_channels, pretrained
        )

    if freeze_backbone:
        for param in model.features.parameters():
            param.requires_grad = False

    # Replace classifier head; keep the MobileNetV3 two-layer structure
    in_features = model.classifier[0].in_features  # 576
    model.classifier = nn.Sequential(
        nn.Linear(in_features, 1024),
        nn.Hardswish(),
        nn.Dropout(p=dropout),
        nn.Linear(1024, num_classes),
    )
    for param in model.classifier.parameters():
        param.requires_grad = True

    return model


class CellClassifier(nn.Module):
    """Wrapper for cell classification with architecture-agnostic utilities."""

    def __init__(
        self,
        input_channels: int = 3,
        num_classes: int = 2,
        pretrained: bool = True,
        dropout: float = 0.5,
        freeze_backbone: bool = False,
        architecture: str = "mobilenet_v3_small",
    ):
        if architecture not in SUPPORTED_ARCHITECTURES:
            raise ValueError(
                f"Unknown architecture '{architecture}'. "
                f"Choose from {SUPPORTED_ARCHITECTURES}."
            )
        super().__init__()
        self.architecture = architecture
        self.input_channels = input_channels
        self.num_classes = num_classes

        if architecture == "resnet18":
            self.model = create_resnet18_classifier(
                input_channels=input_channels,
                num_classes=num_classes,
                pretrained=pretrained,
                dropout=dropout,
                freeze_backbone=freeze_backbone,
            )
        else:  # mobilenet_v3_small
            self.model = create_mobilenet_v3_small_classifier(
                input_channels=input_channels,
                num_classes=num_classes,
                pretrained=pretrained,
                dropout=dropout,
                freeze_backbone=freeze_backbone,
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract the penultimate-layer feature vector."""
        if self.architecture == "resnet18":
            x = self.model.conv1(x)
            x = self.model.bn1(x)
            x = self.model.relu(x)
            x = self.model.maxpool(x)
            x = self.model.layer1(x)
            x = self.model.layer2(x)
            x = self.model.layer3(x)
            x = self.model.layer4(x)
            x = self.model.avgpool(x)
            return torch.flatten(x, 1)
        else:  # mobilenet_v3_small
            x = self.model.features(x)
            x = self.model.avgpool(x)
            return torch.flatten(x, 1)

    def freeze_backbone(self) -> None:
        """Freeze all layers except the final classifier head."""
        if self.architecture == "resnet18":
            for name, param in self.model.named_parameters():
                if not name.startswith("fc"):
                    param.requires_grad = False
        else:
            for param in self.model.features.parameters():
                param.requires_grad = False

    def unfreeze_backbone(self) -> None:
        """Unfreeze all layers."""
        for param in self.model.parameters():
            param.requires_grad = True

    def count_parameters(self) -> dict[str, int]:
        """Count trainable and total parameters."""
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {"total": total, "trainable": trainable, "frozen": total - trainable}


def get_model_for_channel_mode(
    channel_mode: str,
    num_classes: int = 2,
    pretrained: bool = True,
    dropout: float = 0.5,
    freeze_backbone: bool = False,
    architecture: str = "mobilenet_v3_small",
) -> CellClassifier:
    """Return a CellClassifier with the right input channels for the given channel mode.

    Args:
        channel_mode: "all_channels" (5-ch), "multi_channel" (4-ch), or single-channel modes.
        num_classes: Number of output classes.
        pretrained: Whether to use ImageNet pretrained weights.
        dropout: Dropout rate.
        freeze_backbone: Whether to freeze backbone weights.
        architecture: "mobilenet_v3_small" (default, CPU-friendly) or "resnet18".
    """
    if channel_mode == "all_channels":
        input_channels = 5  # DAPI + OPC + RFP + B3-Tub + BF
    elif channel_mode == "multi_channel":
        input_channels = 4  # DAPI + OPC + RFP + B3-Tub
    elif channel_mode == "rfp_bf":
        input_channels = 2  # RFP + BF
    else:
        input_channels = 3  # Single channel replicated to 3

    return CellClassifier(
        input_channels=input_channels,
        num_classes=num_classes,
        pretrained=pretrained,
        dropout=dropout,
        freeze_backbone=freeze_backbone,
        architecture=architecture,
    )
