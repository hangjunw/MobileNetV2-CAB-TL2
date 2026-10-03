"""Backbone constructors — single source of truth for every architecture
compared in the manuscript.

Reuse map (all lifted verbatim from the original scripts):
  * build_mobilenet_v2        <- train.py:333
  * build_mobilenet_v2_color  <- train.py:342  (CAB inserted at index 2)
  * build_mobilenet_v2_eca    <- train.py:368
  * build_resnet18            <- train.py:378
  * build_efficientnet_b0     <- train.py:387
  * mbv3 / mobilevit          <- train.py:449-457 (timm)
  * mbv1 / densenet           <- local mobilenet_v1.py / models_builder.py
                                 (MUST be copied into this package; they were
                                  imported but never committed to the repo)
"""
import torch.nn as nn
from torchvision import models

from .cab import insert_cab
from .eca import add_eca_to_mobilenet_v2

# --- optional local modules; imported lazily so the package still imports
# --- cleanly before the user copies them in.
try:
    from .mobilenet_v1 import mobilenet_v1
except ImportError:  # pragma: no cover
    mobilenet_v1 = None

try:
    from .models_builder import build_densenet_model
except ImportError:  # pragma: no cover
    build_densenet_model = None

try:
    import timm
except ImportError:  # pragma: no cover
    timm = None


def _replace_head(model, num_classes: int) -> nn.Module:
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, num_classes)
    return model


def build_mobilenet_v2(num_classes: int) -> nn.Module:
    model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)
    return _replace_head(model, num_classes)


def build_mobilenet_v2_color(num_classes: int, cab_position: int = 2) -> nn.Module:
    """MobileNetV2 + CAB. This is the topology of BOTH `mbv2_color` and
    `mbv2_tl2` (they differ only in the training strategy, not in structure)."""
    backbone = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)
    _replace_head(backbone, num_classes)
    insert_cab(backbone, position=cab_position, reduction=8)
    return backbone


def build_mobilenet_v2_eca(num_classes: int) -> nn.Module:
    model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)
    model = add_eca_to_mobilenet_v2(model)
    return _replace_head(model, num_classes)


def build_resnet18(num_classes: int) -> nn.Module:
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def build_efficientnet_b0(num_classes: int) -> nn.Module:
    model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
    return _replace_head(model, num_classes)


def build_mobilenet_v1(num_classes: int) -> nn.Module:
    if mobilenet_v1 is None:
        raise ImportError("mobilenet_v1.py missing — restore it from the "
                          "repository (git pull) and reinstall the package")
    return mobilenet_v1(num_classes)


def build_densenet(name: str, num_classes: int, pretrained: bool = True) -> nn.Module:
    if build_densenet_model is None:
        raise ImportError("models_builder.py missing — restore it from the "
                          "repository (git pull) and reinstall the package")
    return build_densenet_model(model_name=name, num_classes=num_classes,
                                pretrained=pretrained)


def build_timm(name: str, num_classes: int, pretrained: bool = True) -> nn.Module:
    if timm is None:
        raise ImportError("timm not installed: pip install timm")
    return timm.create_model(name, pretrained=pretrained, num_classes=num_classes)


TIMM_ALIASES = {
    "mbv3": "mobilenetv3_large_100",
    "mobilevit": "mobilevit_xxs",
}
