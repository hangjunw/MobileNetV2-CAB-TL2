"""Model factory — one name in, one model out.

Centralises ``get_model``, which was previously duplicated in
``train.py`` (line 422) and ``test.py`` (line 548).

The optimisation side of TL2 lives in ``models/tl2.py``; it is re-exported
here so that ``from models import build_optimizer`` keeps working.
"""
from .backbone import (build_mobilenet_v2, build_mobilenet_v2_color,
                        build_mobilenet_v2_eca, build_resnet18,
                        build_efficientnet_b0, build_mobilenet_v1,
                        build_densenet, build_timm, TIMM_ALIASES)
from .tl2 import (TL2Config, DEFAULT_TL2, CAB_POSITION, build_param_groups,
                  build_optimizer, build_scheduler, EarlyStopping,
                  cab_parameter_names, describe_groups)
import torch.nn as nn

# Models whose *training strategy* is transfer learning (layered LR).
# Their topology is identical to the non-TL twin except for the CAB.
TL_MODELS = {"mbv2_tl", "mbv2_tl2"}


def build_mobilenetv2_cab_tl2(num_classes: int,
                              cab_position: int = CAB_POSITION) -> nn.Module:
    """MobileNetV2-CAB-TL2 — the proposed model (manuscript Sec. 2.3).

    Topology  : MobileNetV2 + ColorAttentionBlock at features[cab_position]
                -> models/backbone.py + models/cab.py
    Strategy  : TL2 differential learning rates -> models/tl2.py
    Entry     : get_model("mbv2_tl2", num_classes)

    `mbv2_color` shares this topology exactly; only the optimiser differs
    (see ``build_optimizer(flat_lr=True)`` for the no-DLR ablation row).
    """
    return build_mobilenet_v2_color(num_classes, cab_position=cab_position)


MODEL_REGISTRY = {
    "mbv1": build_mobilenet_v1,
    "mbv2": build_mobilenet_v2,
    "mbv2_tl": build_mobilenet_v2,              # same topology, TL training
    "mbv2_color": build_mobilenet_v2_color,     # CAB, single LR
    "mbv2_tl2": build_mobilenetv2_cab_tl2,      # proposed: CAB + TL2
    "mbv2_eca": build_mobilenet_v2_eca,
    "res18": build_resnet18,
    "effb0": build_efficientnet_b0,
    "mbv3": lambda n: build_timm(TIMM_ALIASES["mbv3"], n),
    "mobilevit": lambda n: build_timm(TIMM_ALIASES["mobilevit"], n),
    "dense121": lambda n: build_densenet("densenet121", n),
    "dense161": lambda n: build_densenet("densenet161", n),
}


def get_model(name: str, num_classes: int = 8):
    """Build a model by name. Case-insensitive.

    Raises:
        ValueError: unknown model name (the message lists every valid key).
    """
    key = name.lower()
    if key not in MODEL_REGISTRY:
        raise ValueError("unknown model: %s (available: %s)"
                         % (name, ", ".join(sorted(MODEL_REGISTRY))))
    return MODEL_REGISTRY[key](num_classes)


def is_tl_model(name: str) -> bool:
    """True when the *training strategy* is TL2 (layered learning rates)."""
    return name.lower() in TL_MODELS


def build_optimizer_for(model, name: str, lr: float = 1e-4,
                        tl_backbone_lr: float = 5e-5, weight_decay: float = 1e-4,
                        cab_position: int = CAB_POSITION):
    """AdamW with layered LR for TL models, flat LR for every other model."""
    return build_optimizer(model, lr=lr, tl_backbone_lr=tl_backbone_lr,
                           weight_decay=weight_decay, cab_position=cab_position,
                           flat_lr=not is_tl_model(name))
