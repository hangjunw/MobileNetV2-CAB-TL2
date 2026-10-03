"""models — 论文所有对比架构与 TL2 优化策略的统一入口。

论文提出的 MobileNetV2-CAB-TL2 由三部分组成：
  * CAB 拓扑   -> models.cab (insert_cab / ColorAttentionBlock)
  * 训练策略   -> models.tl2 (build_optimizer / build_scheduler / EarlyStopping)
  * 统一构建   -> models.factory (get_model / build_optimizer_for / is_tl_model)

其它基线（MobileNetV1 / DenseNet / ResNet18 / EfficientNet-B0 / ECA / timm）
也在此处集中导出，供 train.py / test.py 通过 ``from models import ...`` 复用，
避免在各脚本中重复定义网络结构。
"""
from .factory import (
    get_model,
    is_tl_model,
    build_optimizer_for,
    build_mobilenetv2_cab_tl2,
    MODEL_REGISTRY,
    TL_MODELS,
)
from .backbone import (
    build_mobilenet_v2,
    build_mobilenet_v2_color,
    build_mobilenet_v2_eca,
    build_resnet18,
    build_efficientnet_b0,
    build_mobilenet_v1,
    build_densenet,
    build_timm,
    TIMM_ALIASES,
)
from .cab import insert_cab, ColorAttentionBlock
from .eca import add_eca_to_mobilenet_v2, ECALayer
from .tl2 import (
    TL2Config,
    DEFAULT_TL2,
    CAB_POSITION,
    build_param_groups,
    build_optimizer,
    build_scheduler,
    EarlyStopping,
    cab_parameter_names,
    describe_groups,
)

__all__ = [
    "get_model", "is_tl_model", "build_optimizer_for",
    "build_mobilenetv2_cab_tl2",
    "MODEL_REGISTRY", "TL_MODELS",
    "build_mobilenet_v2", "build_mobilenet_v2_color", "build_mobilenet_v2_eca",
    "build_resnet18", "build_efficientnet_b0", "build_mobilenet_v1",
    "build_densenet", "build_timm", "TIMM_ALIASES",
    "insert_cab", "ColorAttentionBlock", "add_eca_to_mobilenet_v2", "ECALayer",
    "TL2Config", "DEFAULT_TL2", "CAB_POSITION", "build_param_groups",
    "build_optimizer", "build_scheduler", "EarlyStopping",
    "cab_parameter_names", "describe_groups",
]
