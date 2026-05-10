# CardioMark

**CardioMark: A Large-Scale Benchmark for Automated Vertebral Heart Score Estimation in Canine Radiographs**

> NeurIPS 2026 Evaluations and Datasets Track

---

## Overview

CardioMark is a benchmark dataset and evaluation suite for automated Vertebral Heart Score (VHS) estimation in canine lateral thoracic radiographs. It provides:

🔗 **Project Page:** [https://gen-ai-research.github.io/CardioMark/](https://gen-ai-research.github.io/CardioMark/)

- **21,465 radiographs** from **12,385 dogs** across **144 breeds**
- Six cardiac keypoint annotations per image (A–F), enabling geometry-aware VHS computation
- Rigorous inter-annotator agreement: Fleiss' κ = 0.88 across 9 experts on 300 images
- A multi-axis evaluation protocol revealing failure modes invisible to aggregate accuracy

### Key findings

| Finding | Result |
|---|---|
| Best overall accuracy | 89.81% (VectorGeoRefineNet, EfficientNet-B7) |
| Normal class collapse | 42–70% across all architectures |
| Near-boundary accuracy | **65–72%** vs. 99% far from threshold |
| Near-boundary flip rate | 14–17% under ±0.1 VU perturbation |
| Better keypoints ≠ better clinical accuracy | ConvNeXt best NME, worst Normal class |

---

## Dataset splits

| Split | Images | Normal | Borderline | Enlarged |
|---|---|---|---|---|
| Train | 15,036 | 855 | 3,600 | 10,571 |
| Val | 2,154 | 98 | 523 | 1,533 |
| Test | 4,275 | 185 | 1,012 | 3,077 |

Splits are **patient-level** — no dog appears in more than one split.

Ground truth VHS is computed from annotated keypoints via `calc_vhs(six_points)`, not from stored clinical measurements, ensuring a unified evaluation protocol.

---

## Repository structure

```
cardiomark/
├── data/
│   ├── dataset.py      # DogHeartDataset, train/eval transforms
│   └── sampling.py     # WeightedRandomSampler, class weight computation
├── models/
│   └── vgr_net.py      # VectorGeoRefineNet (proposed geometry-aware baseline)
├── utils/
│   ├── vhs.py          # calc_vhs, get_labels, keypoint metrics, calibration
│   ├── losses.py       # VHSAwareLoss, geometry loss, vector losses
│   └── evaluation.py   # validate(), evaluate_test()
├── train.py            # Training entry point
├── evaluate.py         # Standalone evaluation — produces all 5 experiment JSONs
├── extract_table1.py   # Aggregates eval results into paper tables
├── requirements.txt
└── README.md
```

---

## Installation

```bash
git clone https://github.com/your-org/cardiomark.git
cd cardiomark
pip install -r requirements.txt
```

Requires Python ≥ 3.10 and PyTorch ≥ 2.2 with CUDA.

---

## Data access

The CardioMark dataset will be released upon paper acceptance. To request early reviewer access, follow the instructions in the supplementary material submitted with the paper.

Expected directory layout:

```
/path/to/data/
  Train/
    Images/   *.png
    Labels/   *.mat   (field: six_points, shape 6×2 in pixel coords)
  Valid/
    Images/
    Labels/
  Test_Images/
    Images/
    Labels/
```

---

## Training

```bash
python train.py \
    --data_root /path/to/data \
    --gpu 0 \
    --epochs 2000 \
    --image_size 512 \
    --batch_size 64 \
    --accum_steps 4 \
    --lr 1.5e-5 \
    --seed 113 \
    --out_dir runs/vgr_effnet_b7
```

Resume from a checkpoint:

```bash
python train.py \
    --data_root /path/to/data \
    --checkpoint runs/vgr_effnet_b7/models/best_val_acc.pth \
    --out_dir runs/vgr_effnet_b7_resumed
```

Checkpoints saved:

| File | When saved |
|---|---|
| `best_val_loss.pth` | Lowest validation loss |
| `best_val_acc.pth`  | Highest validation accuracy |
| `best_val_f1.pth`   | Highest validation Macro-F1 |
| `epoch_latest.pth`  | Every epoch (overwritten) |

---

## Evaluation

Evaluate a checkpoint and produce all experiment JSONs:

```bash
python evaluate.py \
    --checkpoint runs/vgr_effnet_b7/models/best_val_acc.pth \
    --data_root  /path/to/data \
    --split      Test_Images \
    --out_dir    eval_results_vgr \
    --model_name "VectorGeoRefineNet (EfficientNet-B7)"
```

With post-hoc calibration:

```bash
python evaluate.py ... --calibrate
```

Cross-institution transfer (DogHeart dataset):

```bash
python evaluate.py \
    --checkpoint runs/vgr_effnet_b7/models/best_val_acc.pth \
    --data_root  /path/to/dogheart \
    --split      Test_Images \
    --out_dir    eval_results_dogheart
```

### Output files

| File | Contents |
|---|---|
| `exp1_per_class.json`  | Overall + per-class accuracy, F1, balanced accuracy |
| `exp2_boundary.json`   | Far / Mid / Near boundary accuracy and flip rates |
| `exp3_keypoints.json`  | NME, PCK@0.02, PCK@0.05, per-point pixel errors |
| `exp4_rawrr.json`      | Right-answer-wrong-reason analysis |
| `exp5_bland_altman.json` | Bias, LoA, MAE (clinical agreement) |
| `predictions.csv`      | Per-image predictions with boundary flags |

### Aggregate results into paper tables

```bash
python extract_table1.py eval_results_vgr/ eval_results_convnext/ ...

# or use glob
python extract_table1.py eval_results_*/
```

---

## Baselines

All baselines are evaluated with `evaluate.py` using the same GT protocol (`calc_vhs(keypoints)`).

| Model | Acc | Normal | Borderline | Enlarged | NME |
|---|---|---|---|---|---|
| EfficientNet-B7 Direct Reg. | 81.91% | – | – | – | 0.017 |
| ConvNeXt-tiny Direct Reg. | 82.90% | – | – | – | 0.018 |
| SVGR (HRNet-W32) | 85.35% | 54.59% | 65.91% | 93.60% | 0.024 |
| SVGR (ViT-Base) | 85.66% | 42.70% | 65.91% | 94.74% | 0.024 |
| SVGR (ConvNeXt-Base) | 89.61% | 62.70% | 76.68% | 95.48% | 0.019 |
| **SVGR (EfficientNet-B7) ★** | **89.81%** | **70.81%** | **75.89%** | **95.45%** | **0.009** |

★ = proposed strong baseline. SVGR = Structured Vector Geometry Refinement baseline.

---

## VHS formula

```
VHS = 6 × (|AB| + |CD|) / |EF|
```

Where:
- **A, B** — long cardiac axis endpoints
- **C, D** — short cardiac axis endpoints (CD ⊥ AB by protocol)
- **E, F** — vertebral reference segment endpoints

Clinical thresholds: Normal < 8.2 VU · Borderline 8.2–10.0 · Enlarged ≥ 10.0

---

## Annotation tool

VetMark reduces annotation time from over one minute to 10–30 seconds per image through model-assisted prediction and real-time VHS feedback. The tool enabled CardioMark's scale: at 1 min/image, 21,465 annotations would require ~358 person-hours. VetMark reduces this to ~89 hours.

---

## Citation

If you use CardioMark in your research, please cite:

```bibtex
@inproceedings{cardiomark2026,
  title     = {CardioMark: A Large-Scale Benchmark for Automated Vertebral Heart Score Estimation in Canine Radiographs},
  author    = {Anonymous},
  booktitle = {Advances in Neural Information Processing Systems (NeurIPS)},
  year      = {2026},
  note      = {Evaluations and Datasets Track}
}
```

---

## License

The CardioMark dataset is released under [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/).
Code is released under the MIT License.
