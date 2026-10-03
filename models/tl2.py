"""TL2 — two-level transfer learning (differential learning rates).

TL2 is the fine-tuning strategy of the proposed MobileNetV2-CAB-TL2 model.
The *topology* is identical to ``mbv2_color`` (see ``models/backbone.py``);
only the optimisation strategy differs, so the three parameter groups are
defined here and nowhere else.

Three groups (manuscript §2.3.3, Table 2 / Table 4 "DLR" row)
-------------------------------------------------------------
1. classification head   -- lr (1e-4)          : randomly re-initialised
2. CAB                   -- lr (1e-4)          : newly inserted, trained from
                                                 scratch, so it uses the same
                                                 (higher) LR as the head
3. pretrained backbone   -- tl_backbone_lr (5e-5) : ImageNet weights, smaller LR
                                                 to avoid catastrophic forgetting

Schedule: linear warm-up (5 epochs) -> cosine decay to ``min_lr`` (default
1e-6). Early stopping on validation accuracy with patience 8; the checkpoint
with the best validation accuracy is the one reported in the manuscript.

Usage
-----
    from models import get_model, build_optimizer, build_scheduler, EarlyStopping

    model = get_model("mbv2_tl2", num_classes=8)
    opt = build_optimizer(model, lr=1e-4, tl_backbone_lr=5e-5)
    sch = build_scheduler(opt, epochs=80, warmup_epochs=5)

Note on ``CAB_POSITION``
------------------------
CAB parameters are identified by the parameter-name prefix
``features.{CAB_POSITION}``. Changing it renames every downstream
``features.{i}`` key and makes all 35 published checkpoints unloadable with
``strict=True``. Keep it at 2.
"""
from dataclasses import dataclass, field
from typing import List, Optional

import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR

# Where the CAB lives inside ``model.features``. MUST stay 2 — see module docstring.
CAB_POSITION = 2


@dataclass
class TL2Config:
    """Hyper-parameters of the TL2 schedule (defaults = manuscript §2.3.3)."""

    lr: float = 1.0e-4              # head + CAB
    tl_backbone_lr: float = 5.0e-5  # pretrained backbone
    weight_decay: float = 1.0e-4
    epochs: int = 80
    warmup_epochs: int = 5
    min_lr: float = 1.0e-6          # cosine floor
    early_stop_patience: int = 8    # on validation accuracy
    cab_position: int = CAB_POSITION
    group_names: List[str] = field(default_factory=lambda: ["head", "cab", "backbone"])


DEFAULT_TL2 = TL2Config()


def cab_parameter_names(model, cab_position: int = CAB_POSITION):
    """Yield the parameter names that belong to the inserted CAB.

    Empty if the model has no CAB (e.g. plain ``mbv2``); the caller then simply
    gets an empty middle group, which AdamW tolerates.
    """
    prefix = "features.%d." % cab_position
    for name, _ in model.named_parameters():
        if name.startswith(prefix):
            yield name


def build_param_groups(model, lr: float = 1e-4, tl_backbone_lr: float = 5e-5,
                       weight_decay: float = 1e-4, cab_position: int = CAB_POSITION):
    """Split ``model.parameters()`` into head / CAB / backbone groups.

    MUST be called AFTER the CAB has been inserted (see ``models/cab.py``),
    otherwise the CAB parameters fall into the backbone group.

    Returns:
        (groups, weight_decay) — ``groups`` is a list of three dicts ready to
        be handed to ``optim.AdamW``.
    """
    head_params = list(model.classifier.parameters())
    ca_params, backbone_params = [], []
    cab_prefix = "features.%d." % cab_position

    for name, param in model.named_parameters():
        if name.startswith(cab_prefix):
            ca_params.append(param)
        elif "features" in name:
            backbone_params.append(param)
        # anything else (e.g. timm heads) is intentionally left out of the
        # backbone group; add it to the head group if you use TL2 with such a
        # model.

    groups = [
        {"params": head_params, "lr": lr},
        {"params": ca_params, "lr": lr},
        {"params": backbone_params, "lr": tl_backbone_lr},
    ]
    return groups, weight_decay


def build_optimizer(model, lr: float = 1e-4, tl_backbone_lr: float = 5e-5,
                    weight_decay: float = 1e-4, cab_position: int = CAB_POSITION,
                    flat_lr: bool = False):
    """AdamW with layered LR (TL2).

    Args:
        flat_lr: if True, use a single LR for every parameter (ablation
            control "MobileNetV2 + CAB" without DLR).
    """
    if flat_lr:
        return optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    groups, wd = build_param_groups(model, lr, tl_backbone_lr, weight_decay,
                                    cab_position)
    return optim.AdamW(groups, weight_decay=wd)


def build_scheduler(optimizer, epochs: int = 80, warmup_epochs: int = 5,
                    min_lr: float = 1e-6, warmup_start_factor: float = 0.1):
    """Linear warm-up for ``warmup_epochs`` then cosine decay to ``min_lr``.

    With ``warmup_epochs <= 0`` the warm-up stage is skipped and a plain
    cosine schedule over the full run is returned.
    """
    if warmup_epochs <= 0:
        return CosineAnnealingLR(optimizer, T_max=max(int(epochs), 1),
                                 eta_min=min_lr)

    warmup = LinearLR(optimizer, start_factor=warmup_start_factor,
                      end_factor=1.0, total_iters=int(warmup_epochs))
    cosine = CosineAnnealingLR(optimizer,
                               T_max=max(int(epochs) - int(warmup_epochs), 1),
                               eta_min=min_lr)
    return SequentialLR(optimizer, schedulers=[warmup, cosine],
                        milestones=[int(warmup_epochs)])


class EarlyStopping:
    """Stop training when validation accuracy stops improving.

    Args:
        patience: epochs to wait after the last improvement.
        min_delta: minimum change that counts as an improvement.
        higher_is_better: set False when watching a loss.
    """

    def __init__(self, patience: int = 8, min_delta: float = 0.0,
                 higher_is_better: bool = True):
        self.patience = int(patience)
        self.min_delta = float(min_delta)
        self.higher_is_better = higher_is_better
        self.counter = 0
        self.best: Optional[float] = None
        self.early_stop = False

    def step(self, metric: float) -> bool:
        """Feed one epoch's validation metric. Returns True if improved."""
        if self.best is None:
            self.best = metric
            self.counter = 0
            return True

        delta = metric - self.best if self.higher_is_better else self.best - metric
        if delta > self.min_delta:
            self.best = metric
            self.counter = 0
            return True

        self.counter += 1
        if self.counter >= self.patience:
            self.early_stop = True
        return False

    def state_dict(self):
        return {"counter": self.counter, "best": self.best,
                "early_stop": self.early_stop}

    def load_state_dict(self, state):
        self.counter = state.get("counter", 0)
        self.best = state.get("best", None)
        self.early_stop = state.get("early_stop", False)
        return self


def describe_groups(model, cab_position: int = CAB_POSITION) -> str:
    """One-line summary of the three groups — handy in training logs."""
    groups, _ = build_param_groups(model, cab_position=cab_position)
    names = ["head", "cab", "backbone"]
    parts = []
    for name, group in zip(names, groups):
        n = sum(p.numel() for p in group["params"])
        parts.append("%s: %d params @ lr=%g" % (name, n, group["lr"]))
    return " | ".join(parts)
