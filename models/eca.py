"""ECA — Efficient Channel Attention (mbv2_eca 基线用).

从原 train.py 提取，使 ``models`` 包可独立 import，消除脚本内联重复。
"""
import torch
import torch.nn as nn


class ECALayer(nn.Module):
    def __init__(self, channels: int, k_size: int = 3):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(
            1, 1, kernel_size=k_size,
            padding=(k_size - 1) // 2,
            bias=False,
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.avg_pool(x)                       # [N, C, 1, 1]
        y = y.squeeze(-1).transpose(-1, -2)        # [N, 1, C]
        y = self.conv(y)                           # [N, 1, C]
        y = y.transpose(-1, -2).unsqueeze(-1)      # [N, C, 1, 1]
        y = self.sigmoid(y)
        return x * y.expand_as(x)


def add_eca_to_mobilenet_v2(model: nn.Module, k_size: int = 3) -> nn.Module:
    """在每个 InvertedResidual block 之后串接一个 ECA 模块。"""
    InvertedResidual = None
    for m in model.features:
        if m.__class__.__name__ == "InvertedResidual":
            InvertedResidual = m.__class__
            break

    if InvertedResidual is None:
        raise RuntimeError("未在 MobileNetV2.features 中找到 InvertedResidual")

    for i, m in enumerate(model.features):
        if isinstance(m, InvertedResidual):
            out_ch = None
            for layer in reversed(list(m.modules())):
                if isinstance(layer, nn.Conv2d):
                    out_ch = layer.out_channels
                    break
            if out_ch is None:
                raise RuntimeError(f"无法从 block {i} 中推断输出通道")

            eca = ECALayer(out_ch, k_size)
            model.features[i] = nn.Sequential(m, eca)

    return model
