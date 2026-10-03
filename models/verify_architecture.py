"""Architecture self-check for MobileNetV2-CAB-TL2.

Run it with::

    python -m models                 # from the repository root
    python -m models --all           # also build every registry entry
    python -m models --offline       # no pretrained-weight download

Why this file exists
--------------------
The manuscript reports a fixed architecture: MobileNetV2 with a
``ColorAttentionBlock`` inserted at ``features[2]`` (128 parameters) and a
three-group TL2 optimiser (head 1e-4 / CAB 1e-4 / backbone 5e-5).  Those
numbers are quoted in ``models/README.md`` and in the paper, so they must be
re-derivable from the code at any time.

``train.py`` additionally carries an inline copy of ``ColorAttentionBlock`` and
``build_mobilenet_v2_color`` (``test.py`` mirrors them again).  All published
checkpoints were trained through that inline copy, so the package must stay
byte-compatible with it.  :func:`check_parity_with_train_py` turns that
requirement into an assertion instead of a hope: any future edit that makes the
two copies drift fails this check.

Exit code is 0 when nothing failed, 1 otherwise -- so it can also be wired into
CI as ``python -m models``.
"""
from __future__ import annotations

import argparse
import importlib
import sys
from pathlib import Path

from . import build_optimizer, get_model
from .cab import ColorAttentionBlock, insert_cab
from .tl2 import CAB_POSITION, build_param_groups

# --- reference values (manuscript Sec. 2.3 / models/README.md Sec. 5) --------
EXPECTED_TOTAL_PARAMS = 2_234_248
EXPECTED_NUM_FEATURES = 20
EXPECTED_GROUPS = {"head": 10_248, "cab": 128, "backbone": 2_223_872}
EXPECTED_LRS = {"head": 1e-4, "cab": 1e-4, "backbone": 5e-5}
EXPECTED_CAB_KEYS = {"features.2.conv1.weight", "features.2.conv2.weight"}

_PASSED: list[str] = []
_FAILED: list[str] = []
_SKIPPED: list[str] = []


def _report(label: str, ok: bool, detail: str = "") -> bool:
    tag = "PASS" if ok else "FAIL"
    (_PASSED if ok else _FAILED).append(label)
    line = "    %-26s %-34s %s" % (label, detail, tag)
    print(line)
    return ok


def _skip(label: str, reason: str) -> None:
    _SKIPPED.append(label)
    print("    %-26s %-34s SKIP (%s)" % (label, "", reason))


def _build_offline(num_classes: int = 8):
    """Build the same topology without downloading ImageNet weights."""
    import torch.nn as nn
    from torchvision import models as tv

    model = tv.mobilenet_v2(weights=None)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    insert_cab(model, position=CAB_POSITION, reduction=8)
    return model


def check_topology(model, num_classes: int) -> None:
    print("\n[1] topology of get_model('mbv2_tl2', %d)" % num_classes)

    total = sum(p.numel() for p in model.parameters())
    _report("total parameters", total == EXPECTED_TOTAL_PARAMS,
            "%d (expected %d)" % (total, EXPECTED_TOTAL_PARAMS))

    n_features = len(model.features)
    _report("len(model.features)", n_features == EXPECTED_NUM_FEATURES,
            "%d (expected %d)" % (n_features, EXPECTED_NUM_FEATURES))

    block = model.features[CAB_POSITION]
    _report("features[%d]" % CAB_POSITION,
            isinstance(block, ColorAttentionBlock),
            type(block).__name__)

    cab_keys = {k for k in model.state_dict() if k.startswith("features.%d." % CAB_POSITION)}
    _report("CAB state_dict keys", cab_keys == EXPECTED_CAB_KEYS,
            ", ".join(sorted(cab_keys)))


def check_groups(model) -> None:
    print("\n[2] TL2 parameter groups")
    groups, _ = build_param_groups(model, lr=1e-4, tl_backbone_lr=5e-5,
                                   cab_position=CAB_POSITION)
    for name, group in zip(("head", "cab", "backbone"), groups):
        n = sum(p.numel() for p in group["params"])
        ok = (n == EXPECTED_GROUPS[name]) and (group["lr"] == EXPECTED_LRS[name])
        _report(name, ok, "%d params @ lr=%g" % (n, group["lr"]))

    opt = build_optimizer(model)
    _report("optimizer groups", len(opt.param_groups) == 3,
            ", ".join("%g" % g["lr"] for g in opt.param_groups))


def check_parity_with_train_py(model, offline: bool) -> None:
    """The package and the inline copy in train.py must define the same graph."""
    print("\n[3] parity with the inline copy in train.py")
    if offline:
        _skip("state_dict parity", "offline mode")
        return

    repo_root = str(Path(__file__).resolve().parent.parent)
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)

    try:
        train = importlib.import_module("train")
        inline = train.build_mobilenet_v2_color(8)
    except Exception as exc:                      # missing deps, no weights, ...
        _skip("state_dict parity", "%s: %s" % (type(exc).__name__, exc))
        return

    ours = model.state_dict()
    theirs = inline.state_dict()
    _report("state_dict keys", sorted(ours) == sorted(theirs),
            "%d keys" % len(ours))
    same_shapes = all(
        tuple(ours[k].shape) == tuple(theirs[k].shape)
        for k in ours if k in theirs
    )
    _report("state_dict shapes", same_shapes, "element-wise equal")


def check_registry(build_all: bool, offline: bool) -> None:
    from . import MODEL_REGISTRY, build_mobilenetv2_cab_tl2

    print("\n[4] registry")
    _report("mbv2_tl2 -> builder",
            MODEL_REGISTRY["mbv2_tl2"] is build_mobilenetv2_cab_tl2,
            MODEL_REGISTRY["mbv2_tl2"].__name__)

    if not build_all:
        _skip("build every key", "pass --all to include (downloads weights)")
        return

    ok, bad = [], []
    for key in sorted(MODEL_REGISTRY):
        if key in ("mbv3", "mobilevit") and offline:
            _skip(key, "offline mode")
            continue
        try:
            m = get_model(key, 8)
            ok.append("%s(%d)" % (key, sum(p.numel() for p in m.parameters())))
        except Exception as exc:
            bad.append("%s(%s)" % (key, type(exc).__name__))
    _report("build every key", not bad, ", ".join(ok) if ok else "")
    for item in bad:
        print("        failed: %s" % item)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m models",
        description="Verify the MobileNetV2-CAB-TL2 architecture.")
    parser.add_argument("--all", action="store_true",
                        help="also build every MODEL_REGISTRY entry")
    parser.add_argument("--offline", action="store_true",
                        help="skip pretrained-weight download where possible")
    parser.add_argument("--num-classes", type=int, default=8)
    args = parser.parse_args(argv)

    import torch
    import torchvision

    print("models / MobileNetV2-CAB-TL2 architecture self-check")
    print("  torch %s   torchvision %s" % (torch.__version__, torchvision.__version__))

    if args.offline:
        model = _build_offline(args.num_classes)
    else:
        model = get_model("mbv2_tl2", args.num_classes)

    check_topology(model, args.num_classes)
    check_groups(model)
    check_parity_with_train_py(model, args.offline)
    check_registry(args.all, args.offline)

    print("\nRESULT: %d passed, %d failed, %d skipped"
          % (len(_PASSED), len(_FAILED), len(_SKIPPED)))
    return 1 if _FAILED else 0


if __name__ == "__main__":
    raise SystemExit(main())
