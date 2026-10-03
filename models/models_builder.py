# ========================================================
# 文件名: models/models_builder.py
# 作用: 统一管理实验中所有对比大模型的构建函数 (DenseNet, ResNet等)
# 说明: 此前由 train.py 引用但未提交到仓库，现随 models/ 包一并发布。
# ========================================================
import torch
import torch.nn as nn
import torchvision.models as models


def build_densenet_model(model_name="densenet121", num_classes=8, pretrained=True):
    """
    高效构建 DenseNet 并在白蚁分类数据集上运行
    """
    if model_name.lower() == "densenet121":
        try:
            weights = models.DenseNet121_Weights.DEFAULT if pretrained else None
            model = models.densenet121(weights=weights)
        except AttributeError:
            model = models.densenet121(pretrained=pretrained)

        in_features = model.classifier.in_features
        model.classifier = nn.Sequential(
            nn.Linear(in_features, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            nn.Linear(256, num_classes)
        )

    elif model_name.lower() == "densenet161":
        try:
            weights = models.DenseNet161_Weights.DEFAULT if pretrained else None
            model = models.densenet161(weights=weights)
        except AttributeError:
            model = models.densenet161(pretrained=pretrained)

        in_features = model.classifier.in_features
        model.classifier = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )
    else:
        raise ValueError(f"暂不支持的模型名字: {model_name}")

    return model