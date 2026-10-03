# MobileNetV2-CAB-TL2 — Fine-Grained Termite Identification

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.22710889.svg)](https://doi.org/10.5281/zenodo.22710889)
[![License: MIT](https://img.shields.io/badge/code-MIT-yellow.svg)](LICENSE)
[![License: CC BY 4.0](https://img.shields.io/badge/data-CC%20BY%204.0-blue.svg)](LICENSE-DATA)

Code, trained weights and sample images supporting:

> Huang, H., & Wang, H. *A Lightweight Deep Learning Model Based on Improved MobileNetV2 for Fine-Grained Termite Identification with Shallow Channel Attention* (submitted to *Applied Sciences*).

**MobileNetV2-CAB-TL2** classifies the workers and soldiers of four termite species — eight species–caste categories in total. A **shallow channel attention block (CAB)** is inserted after the first inverted residual block of MobileNetV2: it subtracts the mean response across channels at each spatial location, so that the following channel recalibration depends on relative rather than absolute channel activations (a step standard SE-style attention does not take); global average pooling and a two-1×1-conv bottleneck then produce channel weights that are multiplied back into the feature map. CAB operates on learned feature maps, not on the RGB input, and adds only **128 parameters**. **Differential learning rates (TL2)** update the ImageNet-pretrained backbone (LR 5e-5) more slowly than the new CAB and classification head (LR 1e-4), with all parameters trainable.

## Results (5 random seeds, mean ± SD, held-out test set of 400 images)

| Model | Acc. (%) | Macro-F1 | Params (M) |
|---|---|---|---|
| ResNet18 | 96.25 ± 0.47 | 0.9631 ± 0.0047 | 11.180 |
| MobileNetV2 | 96.95 ± 0.21 | 0.9697 ± 0.0021 | 2.2341 |
| MobileNetV3-Large | 94.30 ± 0.62 | 0.9438 ± 0.0060 | 4.2122 |
| MobileViT-XXS | 94.75 ± 0.87 | 0.9481 ± 0.0088 | 0.9535 |
| **MobileNetV2-CAB-TL2** | **98.40 ± 0.38** | **0.9840 ± 0.0038** | **2.2342** |

Ablation against the baseline (manuscript Table 4): + CAB 97.42 ± 0.42% (p = 0.014), + DLR 97.20 ± 0.86% (p = 0.51), full model 98.40 ± 0.38% (p = 0.002; two-sided paired t-tests on per-seed accuracy). Mean single-image inference latency 4.37 ms (228.7 FPS) on an NVIDIA GeForce RTX 3050 Laptop GPU.

## Dataset

**4,000 RGB images**, exactly **500 per category**, collected from three sites in Huzhou, Zhejiang Province, China, June–November 2025. About 100 individuals per category were each photographed 5–6 times; images with motion blur, poor positioning or low quality were discarded on visual inspection.

| Code | Species | Caste |
|---|---|---|
| `Cf_S` / `Cf_W` | *Coptotermes formosanus* Shiraki | soldier / worker |
| `Rc_S` / `Rc_W` | *Reticulitermes chinensis* Snyder | soldier / worker |
| `Of_S` / `Of_W` | *Odontotermes formosanus* (Shiraki) | soldier / worker |
| `Mb_S` / `Mb_W` | *Macrotermes barneyi* Light | soldier / worker |

- **Split** (before any augmentation, at the individual level, ≈ 70 : 20 : 10): all images of one individual go to a single subset. The test set is fixed at exactly **50 images per category (400 total)** — every reported metric is computed on it.
- **Preprocessing**: validation/test resized to 256 × 256 and centre-cropped to 224 × 224; training images augmented (horizontal flip, ±15° rotation, random resized crop, brightness/contrast, blur, JPEG/noise corruptions, intra-class MixUp α = 0.4).
- **Imaging**: enclosed box, single-sided frosted PMMA plate pre-cooled at −10 °C, top LED ring + bottom LED strips, camera fixed 10 cm above the specimen at ≈ 3× zoom.
- **Identification**: species and caste assignments confirmed by Yongqiang Lu, Huzhou Termite Control Research Institute Co., Ltd.

`sample_data/` contains 5 preview images per category (8 × 5). The full reproduction dataset and the trained weights ship as GitHub Release assets; the full-resolution master copy is archived on Zenodo (DOI 10.5281/zenodo.22710889).

## Repository structure

```
MobileNetV2-CAB-TL2-Termite-Identification/
├── train.py            # training entry point — improved MobileNetV2 (CAB at features[2]),
│                       #   TL2 differential learning rates, all baseline/ablation variants
├── test.py             # evaluation — accuracy / Macro-P / Macro-R / Macro-F1, confusion matrix
├── verify_release.py   # checksum + structure verification of the released dataset package
├── models/             # the proposed model + reusable backbones (see models/README.md)
│   ├── __init__.py           # package exports: get_model, build_mobilenetv2_cab_tl2, ...
│   ├── __main__.py           # enables `python -m models` self-check
│   ├── factory.py            # get_model() registry; maps --model names to builders
│   ├── backbone.py           # MobileNetV2 / ResNet18 / EfficientNet builders + insert_cab()
│   ├── cab.py                # ColorAttentionBlock (CAB) — the proposed attention module
│   ├── eca.py                # ECA variant (--model mbv2_eca)
│   ├── tl2.py                # TL2 strategy: 3 LR groups, warm-up + cosine, early stopping
│   ├── mobilenet_v1.py       # MobileNetV1 baseline (--model mbv1)
│   ├── models_builder.py     # DenseNet builders (--model dense121 / dense161)
│   └── verify_architecture.py # asserts published numbers + parity with train.py
├── sample_data/        # 8 × 5 preview images
├── requirements.txt    # reference environment (manuscript Table 2: Python 3.10, PyTorch 2.0.1, CUDA 11.8)
├── LICENSE             # code — MIT
├── LICENSE-DATA        # dataset — CC BY 4.0
├── CITATION.cff        # "Cite this repository" metadata
└── .zenodo.json        # Zenodo archival metadata
```

The proposed **MobileNetV2-CAB-TL2** is assembled by `models/factory.py::get_model("mbv2_tl2", num_classes)`:
its topology (MobileNetV2 + ColorAttentionBlock at `features[2]`) lives in `models/backbone.py` +
`models/cab.py`, and its TL2 differential-learning-rate strategy lives in `models/tl2.py`.
`train.py` / `test.py` import this same `models` package, so a checkpoint trained by `train.py`
loads in `test.py` unchanged. Run `python -m models` to verify the published architecture numbers
(2,234,248 parameters; CAB at features[2]; TL2 groups of 10,248 / 128 / 2,223,872) and parity
with the inline copy inside `train.py`.

## Quickstart

```bash
# 1. environment
pip install -r requirements.txt

# 2. unpack the released dataset package and verify it
python verify_release.py --dataset ./TermiteData-256px

# 3. train the proposed model (seed 0; repeat with --seed 1..4)
python train.py --model mbv2_tl2 --data ./TermiteData-256px --seed 0

# 4. evaluate a trained checkpoint
python test.py --model mbv2_tl2 --data ./TermiteData-256px --weights outputs/best_mbv2_tl2_seed0.pth
```

Training configuration (manuscript §2.3.3): batch size 32, ≤ 80 epochs, AdamW (weight decay 1e-4), backbone LR 5e-5, CAB + classification head LR 1e-4, 5-epoch linear warm-up then cosine decay, cross-entropy with label smoothing ε = 0.1, early stopping patience 8 on validation accuracy, model selection by best validation accuracy, seeds 0–4.

## Data and code availability

The dataset and source code supporting the findings of this study are publicly available at https://github.com/hangjunw/MobileNetV2-CAB-TL2. The reproduction-resolution dataset and the trained weights ship with GitHub Release V1.1.0; the full-resolution master copy is archived on Zenodo.

## Citation

If you use the dataset or code, please cite:

```bibtex
@dataset{huang_wang_2026_termite,
  author       = {Huang, Hao and Wang, Hangjun},
  title        = {Termite image dataset and MobileNetV2-CAB-TL2 training code
                  for fine-grained termite identification},
  year         = 2026,
  publisher    = {Zenodo},
  version      = {V1.1.0},
  doi          = {10.5281/zenodo.22710889},
  url          = {https://doi.org/10.5281/zenodo.22710889}
}
```

GitHub's "Cite this repository" button uses [CITATION.cff](CITATION.cff).

## Licence

- **Code** — MIT, see [LICENSE](LICENSE)
- **Dataset** — Creative Commons Attribution 4.0 International, see [LICENSE-DATA](LICENSE-DATA)
