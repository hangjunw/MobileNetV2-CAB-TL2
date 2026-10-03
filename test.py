# # test.py
# import os
# import sys
#
# FILE_DIR = os.path.dirname(os.path.abspath(__file__))
# if FILE_DIR not in sys.path:
#     sys.path.insert(0, FILE_DIR)
#
# from mobilenet_v1 import mobilenet_v1
# # 其他原来的 import 保持不变
#
# import argparse
#
# import torch
# import torch.nn as nn
# from torch.utils.data import DataLoader
# from torchvision import datasets, transforms, models
#
# import numpy as np
# from sklearn.metrics import classification_report, confusion_matrix
# import matplotlib.pyplot as plt
# from typing import List
# import torch.nn.functional as F
# from sklearn.metrics import classification_report
# from sklearn.metrics import precision_score
# # 计算宏平均预测率
#
#
#
# # -------------------------------------------------------
# # 数据：只加载 test 子集
# # -------------------------------------------------------
# def get_test_loader(
#     data_root: str,
#     img_size: int = 224,
#     batch_size: int = 32,
#     num_workers: int = 0,
# ):
#     mean = [0.485, 0.456, 0.406]
#     std = [0.229, 0.224, 0.225]
#
#     test_transform = transforms.Compose([
#         transforms.Resize((img_size, img_size)),
#         # transforms.CenterCrop(img_size),
#         transforms.ToTensor(),
#         transforms.Normalize(mean, std),
#     ])
#
#     test_dir = os.path.join(data_root, "test")
#     if not os.path.exists(test_dir):
#         raise FileNotFoundError(f"未在 {data_root} 下找到 test 文件夹")
#     test_dataset = datasets.ImageFolder(test_dir, transform=test_transform)
#
#     test_loader = DataLoader(
#         test_dataset,
#         batch_size=batch_size,
#         shuffle=False,
#         num_workers=num_workers,
#         pin_memory=True,
#     )
#
#     return test_loader, test_dataset.classes
# # 颜色感知模块
# class ColorAttentionBlock(nn.Module):
#     def __init__(self, channels: int, reduction: int = 8):
#         super().__init__()
#         hidden = max(channels // reduction, 4)
#         self.conv1 = nn.Conv2d(channels, hidden, kernel_size=1, bias=False)
#         self.relu = nn.ReLU(inplace=True)
#         self.conv2 = nn.Conv2d(hidden, channels, kernel_size=1, bias=False)
#         self.sigmoid = nn.Sigmoid()
#
#     def forward(self, x):
#         color_feat = x - x.mean(dim=1, keepdim=True)
#         y = F.adaptive_avg_pool2d(color_feat, 1)
#         y = self.conv1(y)
#         y = self.relu(y)
#         y = self.conv2(y)
#         y = self.sigmoid(y)
#         return x * y
#
#
# # -------------------------------------------------------
# # ECA 模块 & MobileNetV2 + ECA（与训练保持一致）
# # -------------------------------------------------------
# class ECALayer(nn.Module):
#     def __init__(self, channels: int, k_size: int = 3):
#         super().__init__()
#         self.avg_pool = nn.AdaptiveAvgPool2d(1)
#         self.conv = nn.Conv1d(
#             1, 1, kernel_size=k_size,
#             padding=(k_size - 1) // 2,
#             bias=False
#         )
#         self.sigmoid = nn.Sigmoid()
#
#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         y = self.avg_pool(x)
#         y = y.squeeze(-1).transpose(-1, -2)
#         y = self.conv(y)
#         y = y.transpose(-1, -2).unsqueeze(-1)
#         y = self.sigmoid(y)
#         return x * y.expand_as(x)
#
#
# def add_eca_to_mobilenet_v2(model: nn.Module, k_size: int = 3) -> nn.Module:
#     """
#     不再从 torchvision.models.mobilenet import InvertedResidual，
#     改为自动识别 block 类型，兼容不同版本。
#     """
#     InvertedResidual = None
#     for m in model.features:
#         if m.__class__.__name__ == "InvertedResidual":
#             InvertedResidual = m.__class__
#             break
#
#     if InvertedResidual is None:
#         raise RuntimeError("未在 MobileNetV2.features 中找到 InvertedResidual")
#
#
#     for i, m in enumerate(model.features):
#         if isinstance(m, InvertedResidual):
#             out_ch = None
#             for layer in reversed(list(m.modules())):
#                 if isinstance(layer, nn.Conv2d):
#                     out_ch = layer.out_channels
#                     break
#             if out_ch is None:
#                 raise RuntimeError(f"无法从 block {i} 中推断通道数")
#
#             eca = ECALayer(out_ch, k_size)
#             model.features[i] = nn.Sequential(m, eca)
#
#     return model
#
# def build_mobilenet_v2_color(num_classes: int):
#     try:
#         backbone = models.mobilenet_v2(
#             weights=models.MobileNet_V2_Weights.IMAGENET1K_V1
#         )
#     except:
#         backbone = models.mobilenet_v2(pretrained=True)
#
#     in_features = backbone.classifier[1].in_features
#     backbone.classifier[1] = nn.Linear(in_features, num_classes)
#
#     features = list(backbone.features)
#     first_block_channels = 16
#     cab = ColorAttentionBlock(first_block_channels, reduction=8)
#
#     new_features = []
#     new_features.append(features[0])
#     new_features.append(features[1])
#     new_features.append(cab)
#     new_features.extend(features[2:])
#
#     backbone.features = nn.Sequential(*new_features)
#     return backbone
#
# def build_mobilenet_v2(num_classes: int) -> nn.Module:
#     model = models.mobilenet_v2(
#         weights=models.MobileNet_V2_Weights.IMAGENET1K_V1
#     )
#     model.classifier[1] = nn.Linear(model.last_channel, num_classes)
#     return model
#
#
# def build_mobilenet_v2_eca(num_classes: int) -> nn.Module:
#     model = models.mobilenet_v2(
#         weights=models.MobileNet_V2_Weights.IMAGENET1K_V1
#     )
#     model = add_eca_to_mobilenet_v2(model)
#     model.classifier[1] = nn.Linear(model.last_channel, num_classes)
#     return model
#
# # def build_mobilenet_v2_tl(num_classes: int) -> nn.Module:
# #     model = models.mobilenet_v2_tl(
# #         weights=models.MobileNet_V2_Weights.IMAGENET1K_V1
# #     )
# #     model = add_eca_to_mobilenet_v2_tl(model)
# #     model.classifier[1] = nn.Linear(model.last_channel, num_classes)
# #     return model
#
#
#
# def build_resnet18(num_classes: int) -> nn.Module:
#     model = models.resnet18(
#         weights=models.ResNet18_Weights.IMAGENET1K_V1
#     )
#     model.fc = nn.Linear(model.fc.in_features, num_classes)
#     return model
#
#
# def build_efficientnet_b0(num_classes: int) -> nn.Module:
#     model = models.efficientnet_b0(
#         weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1
#     )
#     model.classifier[1] = nn.Linear(model.classifier[1].in_features,
#                                     num_classes)
#     return model
#
#
# def get_model(name: str, num_classes: int) -> nn.Module:
#     name = name.lower()
#     if name in ["mbv2", "mbv2_tl"]:     # ← 这里同时支持 mbv2 和 mbv2_tl
#         return build_mobilenet_v2(num_classes)
#     elif name == "mbv2_eca":
#         return build_mobilenet_v2_eca(num_classes)
#     elif name == "res18":
#         return build_resnet18(num_classes)
#     elif name == "effb0":
#         return build_efficientnet_b0(num_classes)
#     elif name == "mbv2_color":
#         return build_mobilenet_v2_color(num_classes)
#     elif name == "mbv2_tl2":
#         # TL2 的网络结构 = MobileNetV2-Color
#         return build_mobilenet_v2_color(num_classes)
#     elif name == "mbv1":
#         return mobilenet_v1(num_classes)
#     else:
#         raise ValueError(
#             "模型名必须是 mbv2 / mbv2_tl / mbv2_tl2 / mbv2_eca / res18 / effb0 / mbv2_color / mbv1"
#         )
#
#
#
# # -------------------------------------------------------
# # 混淆矩阵可视化
# # -------------------------------------------------------
# def plot_confusion_matrix(cm, classes: List[str], model_name: str):
#     plt.figure(figsize=(8, 6))
#     plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
#     plt.title(f"Confusion Matrix - {model_name}")
#     plt.colorbar()
#
#     tick_marks = np.arange(len(classes))
#     plt.xticks(tick_marks, classes, rotation=45, ha="right")
#     plt.yticks(tick_marks, classes)
#
#     plt.ylabel("True Label")
#     plt.xlabel("Predicted Label")
#     plt.tight_layout()
#
#     save_name = f"cm_{model_name}.png"
#     plt.savefig(save_name, dpi=300)
#     print(f"混淆矩阵已保存到：{save_name}")
#
#
# # -------------------------------------------------------
# # 主测试函数
# # -------------------------------------------------------
# def main():
#     parser = argparse.ArgumentParser(
#         description="Test termite classifier (mbv2 / mbv2_eca / res18 / effb0)"
#     )
#     parser.add_argument(
#         "--data",
#         type=str,
#         required=True,
#         help="TermiteData/dataset 根目录（包含 train/val/test、labels.txt）",
#     )
#     parser.add_argument(
#         "--model",
#         type=str,
#         default="mbv2",
#         choices=["mbv1","mbv2", "mbv2_eca", "res18", "effb0","mbv2_color","mbv2_tl","mbv2_tl2"],
#         help="模型名称",
#     )
#     parser.add_argument(
#         "--weights",
#         type=str,
#         required=True,
#         help="权重文件路径（best_xxx.pth）",
#     )
#     parser.add_argument(
#         "--batch",
#         type=int,
#         default=32,
#         help="测试 batch size",
#     )
#     args = parser.parse_args()
#
#     # 读取 labels.txt
#     labels_path = os.path.join(args.data, "labels.txt")
#     if not os.path.exists(labels_path):
#         raise FileNotFoundError(f"未在 {args.data} 下找到 labels.txt")
#
#     with open(labels_path, "r", encoding="utf-8") as f:
#         class_names = [line.strip() for line in f.readlines()]
#     num_classes = len(class_names)
#     print("类别顺序：", class_names)
#
#     # 数据
#     test_loader, _ = get_test_loader(
#         args.data,
#         img_size=224,
#         batch_size=args.batch,
#         num_workers=0,
#     )
#
#     # 设备 & 模型
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     print("使用设备：", device)
#
#     model = get_model(args.model, num_classes).to(device)
#
#     ckpt = torch.load(args.weights, map_location=device)
#     if isinstance(ckpt, dict) and "state_dict" in ckpt:
#         state_dict = ckpt["state_dict"]
#     else:
#         state_dict = ckpt
#     model.load_state_dict(state_dict)
#     model.eval()
#
#     # 推理
#     all_preds = []
#     all_labels = []
#
#     with torch.no_grad():
#         for imgs, labels in test_loader:
#             imgs = imgs.to(device)
#             labels = labels.to(device)
#
#             outputs = model(imgs)
#             preds = outputs.argmax(dim=1)
#
#             all_preds.extend(preds.cpu().numpy().tolist())
#             all_labels.extend(labels.cpu().numpy().tolist())
#
#     all_preds = np.array(all_preds)
#     all_labels = np.array(all_labels)
#
#     acc = (all_preds == all_labels).mean() * 100.0
#     print(f"\n[{args.model}] Test Accuracy = {acc:.2f}%\n")
#
#     print("分类报告（Precision / Recall / F1）：\n")
#     print(
#         classification_report(
#             all_labels,
#             all_preds,
#             target_names=class_names,
#             digits=4,
#         )
#     )
#     # —— 添加这两行以便 eval_all_models.py 能找到 macro-F1 ——
#     from sklearn.metrics import f1_score, recall_score
#     macro_f1 = f1_score(all_labels, all_preds, average='macro')
#     print(f"macro-F1 = {macro_f1:.4f}")
#     # ★ 新增：Macro-Accuracy（宏平均准确率）
#     macro_acc = recall_score(all_labels, all_preds, average='macro')
#     print(f"Macro-Accuracy = {macro_acc:.4f}")
#     # ★ 新增：Macro-Precision（宏平均预测率）
#     macro_prec = precision_score(all_labels, all_preds, average='macro')
#     print(f"Macro-Precision = {macro_prec:.4f}")  # 这里的打印格式要对应 eval 脚本的正则匹配
#     cm = confusion_matrix(all_labels, all_preds)
#     plot_confusion_matrix(cm, class_names, args.model)
#
#
# if __name__ == "__main__":
#     main()
# test.py
import os
import sys

FILE_DIR = os.path.dirname(os.path.abspath(__file__))
if FILE_DIR not in sys.path:
    sys.path.insert(0, FILE_DIR)

from models.mobilenet_v1 import mobilenet_v1

import argparse
import timm  # 💡 核心关键：引入 timm 库，保证测试时能成功构建新模型结构

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models

import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
from typing import List
import torch.nn.functional as F
from sklearn.metrics import precision_score


# -------------------------------------------------------
# 数据：只加载 test 子集
# -------------------------------------------------------
def get_test_loader(
        data_root: str,
        img_size: int = 224,
        batch_size: int = 32,
        num_workers: int = 0,
):
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    test_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    test_dir = os.path.join(data_root, "test")
    if not os.path.exists(test_dir):
        raise FileNotFoundError(f"未在 {data_root} 下找到 test 文件夹")
    test_dataset = datasets.ImageFolder(test_dir, transform=test_transform)

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return test_loader, test_dataset.classes


# 颜色感知模块
class ColorAttentionBlock(nn.Module):
    def __init__(self, channels: int, reduction: int = 8):
        super().__init__()
        hidden = max(channels // reduction, 4)
        self.conv1 = nn.Conv2d(channels, hidden, kernel_size=1, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(hidden, channels, kernel_size=1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        color_feat = x - x.mean(dim=1, keepdim=True)
        y = F.adaptive_avg_pool2d(color_feat, 1)
        y = self.conv1(y)
        y = self.relu(y)
        y = self.conv2(y)
        y = self.sigmoid(y)
        return x * y


# -------------------------------------------------------
# ECA 模块 & MobileNetV2 + ECA
# -------------------------------------------------------
class ECALayer(nn.Module):
    def __init__(self, channels: int, k_size: int = 3):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(
            1, 1, kernel_size=k_size,
            padding=(k_size - 1) // 2,
            bias=False
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.avg_pool(x)
        y = y.squeeze(-1).transpose(-1, -2)
        y = self.conv(y)
        y = y.transpose(-1, -2).unsqueeze(-1)
        y = self.sigmoid(y)
        return x * y.expand_as(x)


def add_eca_to_mobilenet_v2(model: nn.Module, k_size: int = 3) -> nn.Module:
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
                raise RuntimeError(f"无法从 block {i} 中推断通道数")

            eca = ECALayer(out_ch, k_size)
            model.features[i] = nn.Sequential(m, eca)

    return model


def build_mobilenet_v2_color(num_classes: int):
    try:
        backbone = models.mobilenet_v2(
            weights=models.MobileNet_V2_Weights.IMAGENET1K_V1
        )
    except:
        backbone = models.mobilenet_v2(pretrained=True)

    in_features = backbone.classifier[1].in_features
    backbone.classifier[1] = nn.Linear(in_features, num_classes)

    features = list(backbone.features)
    first_block_channels = 16
    cab = ColorAttentionBlock(first_block_channels, reduction=8)

    new_features = []
    new_features.append(features[0])
    new_features.append(features[1])
    new_features.append(cab)
    new_features.extend(features[2:])

    backbone.features = nn.Sequential(*new_features)
    return backbone


def build_mobilenet_v2(num_classes: int) -> nn.Module:
    model = models.mobilenet_v2(
        weights=models.MobileNet_V2_Weights.IMAGENET1K_V1
    )
    model.classifier[1] = nn.Linear(model.last_channel, num_classes)
    return model


def build_mobilenet_v2_eca(num_classes: int) -> nn.Module:
    model = models.mobilenet_v2(
        weights=models.MobileNet_V2_Weights.IMAGENET1K_V1
    )
    model = add_eca_to_mobilenet_v2(model)
    model.classifier[1] = nn.Linear(model.last_channel, num_classes)
    return model


def build_resnet18(num_classes: int) -> nn.Module:
    model = models.resnet18(
        weights=models.ResNet18_Weights.IMAGENET1K_V1
    )
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def build_efficientnet_b0(num_classes: int) -> nn.Module:
    model = models.efficientnet_b0(
        weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1
    )
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    return model


def get_model(name: str, num_classes: int) -> nn.Module:
    name = name.lower()
    if name in ["mbv2", "mbv2_tl"]:
        return build_mobilenet_v2(num_classes)
    elif name == "mbv2_eca":
        return build_mobilenet_v2_eca(num_classes)
    elif name == "res18":
        return build_resnet18(num_classes)
    elif name == "effb0":
        return build_efficientnet_b0(num_classes)
    elif name == "mbv2_color":
        return build_mobilenet_v2_color(num_classes)
    elif name == "mbv2_tl2":
        return build_mobilenet_v2_color(num_classes)
    elif name == "mbv1":
        return mobilenet_v1(num_classes)

    # --- 💡 新增 timm 现代化模型分支，保证能顺利加载权重并进行评测 ---
    elif name == "mbv3":
        return timm.create_model('mobilenetv3_large_100', pretrained=False, num_classes=num_classes)
    elif name == "mobilevit":
        return timm.create_model('mobilevit_xxs', pretrained=False, num_classes=num_classes)

    else:
        raise ValueError(
            "未知模型名"
        )


# -------------------------------------------------------
# 混淆矩阵可视化
# -------------------------------------------------------
def plot_confusion_matrix(cm, classes: List[str], model_name: str):
    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    plt.title(f"Confusion Matrix - {model_name}")
    plt.colorbar()

    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45, ha="right")
    plt.yticks(tick_marks, classes)

    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.tight_layout()

    save_name = f"cm_{model_name}.png"
    plt.savefig(save_name, dpi=300)
    print(f"混淆矩阵已保存到：{save_name}")


# -------------------------------------------------------
# 主测试函数
# -------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Test termite classifier"
    )
    parser.add_argument(
        "--data",
        type=str,
        required=True,
        help="TermiteData/dataset 根目录",
    )

    # --- 💡 核心修改点：在 choices 白名单中追加 'mbv3' 和 'mobilevit'，彻底解除拦截错误 ---
    parser.add_argument(
        "--model",
        type=str,
        default="mbv2",
        choices=["mbv1", "mbv2", "mbv2_eca", "res18", "effb0", "mbv2_color", "mbv2_tl", "mbv2_tl2", "mbv3",
                 "mobilevit"],
        help="模型名称",
    )
    parser.add_argument(
        "--weights",
        type=str,
        required=True,
        help="权重文件路径（best_xxx.pth）",
    )
    parser.add_argument(
        "--batch",
        type=int,
        default=32,
        help="测试 batch size",
    )
    args = parser.parse_args()

    # 读取 labels.txt
    labels_path = os.path.join(args.data, "labels.txt")
    if not os.path.exists(labels_path):
        raise FileNotFoundError(f"未在 {args.data} 下找到 labels.txt")

    with open(labels_path, "r", encoding="utf-8") as f:
        class_names = [line.strip() for line in f.readlines()]
    num_classes = len(class_names)
    print("类别顺序：", class_names)

    # 数据
    test_loader, _ = get_test_loader(
        args.data,
        img_size=224,
        batch_size=args.batch,
        num_workers=0,
    )

    # 设备 & 模型
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("使用设备：", device)

    model = get_model(args.model, num_classes).to(device)

    ckpt = torch.load(args.weights, map_location=device)
    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        state_dict = ckpt["state_dict"]
    else:
        state_dict = ckpt
    model.load_state_dict(state_dict)
    model.eval()

    # 推理
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for imgs, labels in test_loader:
            imgs = imgs.to(device)
            labels = labels.to(device)

            outputs = model(imgs)
            preds = outputs.argmax(dim=1)

            all_preds.extend(preds.cpu().numpy().tolist())
            all_labels.extend(labels.cpu().numpy().tolist())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    acc = (all_preds == all_labels).mean() * 100.0
    print(f"\n[{args.model}] Test Accuracy = {acc:.2f}%\n")

    print("分类报告（Precision / Recall / F1）：\n")
    print(
        classification_report(
            all_labels,
            all_preds,
            target_names=class_names,
            digits=4,
        )
    )

    from sklearn.metrics import f1_score, recall_score
    macro_f1 = f1_score(all_labels, all_preds, average='macro')
    print(f"macro-F1 = {macro_f1:.4f}")
    macro_acc = recall_score(all_labels, all_preds, average='macro')
    print(f"Macro-Accuracy = {macro_acc:.4f}")
    macro_prec = precision_score(all_labels, all_preds, average='macro')
    print(f"Macro-Precision = {macro_prec:.4f}")

    cm = confusion_matrix(all_labels, all_preds)
    plot_confusion_matrix(cm, class_names, args.model)


if __name__ == "__main__":
    main()