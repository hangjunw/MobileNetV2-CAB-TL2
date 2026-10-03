"""CAB — Channel Attention Block (论文中的 CAB).

论文提出的 MobileNetV2-CAB-TL2 由三部分组成：
  * CAB 拓扑   -> 本模块 (insert_cab / ColorAttentionBlock)
  * 训练策略   -> models.tl2 (build_optimizer / build_scheduler / EarlyStopping)
  * 统一构建   -> models.factory (get_model / build_optimizer_for)

原项目在 train.py 中以内联类 ``ColorAttentionBlock`` 实现同一逻辑，
test.py 又重复了一份。这里把它提取为独立模块，使 ``models`` 包可被
``from models import ...`` 直接复用，消除两处脚本中的重复定义。

CAB 默认插在 MobileNetV2 ``features`` 的第 2 个位置 (index 2，即第一个
inverted-residual block 之后)，与 ``models/backbone.py`` 的 ``CAB_POSITION=2``
以及 ``models/tl2.py`` 的 ``cab_parameter_names`` 前缀保持一致。
"""
import torch
import torch.nn as nn


class ColorAttentionBlock(nn.Module):
    """轻量颜色/通道注意力模块。

    利用通道间的颜色对比（去掉灰度分量）生成通道权重，
    再对输入做逐通道重标定。与论文 §2.3 的 CAB 对应。
    """

    def __init__(self, channels: int, reduction: int = 8):
        super().__init__()
        hidden = max(channels // reduction, 4)
        self.conv1 = nn.Conv2d(channels, hidden, kernel_size=1, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(hidden, channels, kernel_size=1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x: [B, C, H, W]
        color_feat = x - x.mean(dim=1, keepdim=True)
        y = torch.mean(color_feat, dim=(2, 3), keepdim=True)
        y = self.conv1(y)
        y = self.relu(y)
        y = self.conv2(y)
        y = self.sigmoid(y)
        return x * y


def _infer_out_channels(layer: nn.Module) -> int:
    """从某个 layer 中推断其输出通道数（取最后一个 Conv2d）。"""
    out = None
    for m in reversed(list(layer.modules())):
        if isinstance(m, nn.Conv2d):
            out = m.out_channels
            break
    if out is None:
        raise RuntimeError("无法从给定 layer 推断输出通道数")
    return out


def insert_cab(model: nn.Module, position: int = 2, reduction: int = 8) -> nn.Module:
    """在 ``model.features`` 列表的 ``position`` 处插入一个 CAB。

    ``position=2`` 时等价于原 ``train.py::build_mobilenet_v2_color`` 的插入方式：
    ``[stem, first_block, CAB, *rest]``。

    Args:
        model: 一个 MobileNetV2（或其变体），``model.features`` 为 ``nn.Sequential``。
        position: CAB 插入位置（0-indexed）。默认 2。
        reduction: CAB 中间通道的压缩比。默认 8。

    Returns:
        就地修改后的 ``model``（同时也返回，方便链式调用）。
    """
    features = list(model.features)
    if position < 0 or position > len(features):
        raise ValueError(
            f"position={position} 超出 features 长度 {len(features)}"
        )
    in_channels = (
        _infer_out_channels(features[position - 1]) if position > 0 else 16
    )
    cab = ColorAttentionBlock(in_channels, reduction=reduction)
    features.insert(position, cab)
    model.features = nn.Sequential(*features)
    return model
