# Dog Re-Identification (ReID) – PetFace Setup, Regional Features & Training

This repository provides a complete pipeline for dog re-identification, including dataset preparation (PetFace), regional feature extraction from facial landmarks, training/evaluation scripts, and research-friendly backbone options.

## Table of Contents

- [Dog Re-Identification (ReID) – PetFace Setup, Regional Features \& Training](#dog-re-identification-reid--petface-setup-regional-features--training)
  - [Table of Contents](#table-of-contents)
  - [1. Overview](#1-overview)
  - [2. Repository Structure](#2-repository-structure)
  - [3. PetFace Dataset Setup](#3-petface-dataset-setup)
    - [3.1 Files Created](#31-files-created)
    - [3.2 Setup Steps](#32-setup-steps)
    - [3.3 Output \& Batch Format](#33-output--batch-format)
  - [4. Regional Feature Extraction](#4-regional-feature-extraction)
    - [4.1 Dataset Class](#41-dataset-class)
    - [4.2 Regions](#42-regions)
    - [4.3 Landmark JSON Format](#43-landmark-json-format)
  - [5. Training \& Evaluation](#5-training--evaluation)
    - [5.1 Quick Start](#51-quick-start)
    - [5.2 Hyperparameter Overrides](#52-hyperparameter-overrides)
    - [5.3 Hyperparameter Sweeps](#53-hyperparameter-sweeps)
  - [6. Research Backbones](#6-research-backbones)
  - [7. Troubleshooting](#7-troubleshooting)
  - [8. Next Steps](#8-next-steps)

---

## 1. Overview

- **Goal:** Dog re-identification using global and **regional** facial features.  
- **Dataset:** PetFace (dogs organized by identity folders).  
- **Regional model:** Uses landmark-driven crops (eyes, nose, mouth, ears, forehead) and an attention-based or fusion head.  
- **Strict data quality:** Landmarks are **required** to prevent noisy training.

> Example paths use SageMaker: `/home/sagemaker-user/LostPet/...`. Adjust for your environment.

---

## 2. Repository Structure
```
Data (images, CSV splits) ──▶ Datasets & Samplers ──▶ Processor (batches) ──▶ Model
                                                 └──▶ Loss (ID / Triplet / Center)
                                                            └──▶ Solver (optim & sched)
                                                                           └──▶ Train / Eval Loops
                                                                                      └──▶ Utils (logging, metrics)
```

- **config/**: Experiment configuration (backbone, hyper-params, data paths).  
- **datasets/**: Dataset definitions, split scripts, dataloaders, samplers.  
- **processor/**: Batch assembly, forward pass orchestration for training/eval.  
- **model/**: Backbone registry, fusion layers, final embedding heads.  
- **loss/**: ID (classification), triplet, optional center loss.  
- **solver/**: Optimizer and LR scheduler factory.  
- **train/**: Training entrypoints per setting (baseline, SWIN, DINOv3, regional, etc.).  
- **utils/**: Logging, metrics, timers.  
- **hp_grid_search/**: Sweep driver (`sweep.sh`) and launcher (`launch.py`).  
- **final_model/**: Frozen/best configs and quick tests for delivery.  
- **docs/**: How-tos and narrative docs.  
- **splits/**, **splits_petface/**, **splits_petface_valid/**, **splits_petface_test_10k/**: CSV split files.  
- **images/**: Optional sample images (gallery/query/train/val structure for demos).

---
Directory Map (Current)

The list below reflects the current **top-level** organization you shared (non-exhaustive for files).

```
config/
  config_training.py
  config_petface.py
  config_resnet50_petface.py
  config_dinov3{b,l,s}.py
  config_swin{,b_petface,l,swint}.py
  config_multilevel{,_advanced,_b,_swin}.py
  config_research.py

datasets/
  create_petface_splits.py
  create_valid_splits.py
  create_large_test_split.py
  filter_petface_valid_images.py
  make_dataloader_{petface,dogreid,regional}.py
  dog_face_regional{,_dataset}.py
  dog_multipose.py
  sampler.py
  __init__.py

docs/
  README.md
  PETFACE_SETUP.md
  REGIONAL_EXTRACTION_GUIDE.md
  RESEARCH_BACKBONES.md
  TRAINING_GUIDE.md
  SUMMARY.md
  demos/
    demo_albumentations.py
    example_petface_usage.py

final_model/
  config_regional.py
  test_regional_{dataloader,model,training_step}.py
  train_regional.py

hp_grid_search/
  launch.py
  sweep.sh

loss/
  id_loss.py
  triplet.py
  center.py
  make_loss.py
  __init__.py

model/
  backbones.py
  attention_fusion.py
  layers.py
  make_model.py
  __init__.py

processor/
  processor_{dogreid,regional}.py
  __init__.py

solver/
  lr_scheduler.py
  make_optimizer.py
  __init__.py

test/
  test_*.py   # baselines, regional, backbones, pipeline, etc.

train/
  train_{dogreid,petface,resnet50_petface}.py
  train_dinov3{b,l,s}.py
  train_multilevel{,_b,_swin}.py
  train_swin{,b_petface,l,swint}.py

utils/
  logger.py
  meter.py
  metrics.py
  __init__.py

splits*/  # various CSV split folders (train/val/test + query/gallery)
images/
  gallery/ | query/ | train/ | val/
```



**Training**

```bash
uv run python final_model/train_regional.py
```

---

## 3. PetFace Dataset Setup

### 3.1 Files Created

1) **Split generation** — `create_petface_splits.py`  
   - Splits by **dog ID** (7:1:2 train:val:test).  
   - Generates in `splits_petface/`:
     - `train.csv`
     - `val_query.csv` (first image per val ID)
     - `val_gallery.csv` (remaining val images)
     - `test_query.csv` (first image per test ID)
     - `test_gallery.csv` (remaining test images)

2) **Dataset config** — `config_petface.py`  
   - Paths, preprocessing, model, and hyperparameters.

3) **DataLoader** — `datasets/make_dataloader_petface.py`  
   - Five loaders: train, val query/gallery, test query/gallery (PK sampling in train).

4) **Example** — `example_petface_usage.py`  
   - Minimal usage of the PetFace dataloader.

### 3.2 Setup Steps

**Step 1 — Verify structure**
```
/home/sagemaker-user/LostPet/petface/
├── dog/
│   ├── 000001/
│   │   ├── 00.png
│   │   └── ...
│   ├── 000002/
│   └── ...
└── splits_petface/
    ├── train.csv
    ├── val_query.csv
    ├── val_gallery.csv
    ├── test_query.csv
    └── test_gallery.csv
```

**Step 2 — Test DataLoader**
```bash
python3 example_petface_usage.py
```

### 2.1 Output & Batch Format

Each training batch returns:
```python
imgs  : torch.Tensor [B, 3, H, W]
pids  : torch.Tensor [B]        # identity IDs
camids: torch.Tensor [B]        # all zeros for PetFace
paths : List[str] length B      # original image paths
```

**Relabeling:**

- Train: `relabel=True` → dense labels 0..N-1  
- Val/Test: `relabel=False` → original IDs (for query/gallery matching)

---

## 3. Regional Feature Extraction

> **Required:** Landmark JSONs for every image used. The loader raises errors if missing/invalid to ensure data quality.

### 3.1 Dataset Class

`datasets/dog_face_regional.py`:

- Loads images from CSVs.  
- Requires landmark JSONs for region bboxes.  
- Produces a dict with global image **+ 7** regional crops.

**Usage**
```python
from datasets.dog_face_regional import DogFaceRegionalDataset

dataset = DogFaceRegionalDataset(
    csv_path='splits_petface/train.csv',
    image_dir='/home/sagemaker-user/LostPet/petface/dog',
    landmarks_dir='/path/to/petface_landmarks_json_all',  # REQUIRED
    is_train=True
)
sample = dataset[0]
```

### 3.2 Regions

| Region    | Description               |
|-----------|---------------------------|
| left_eye  | Left eye                  |
| right_eye | Right eye                 |
| nose      | Nose and surrounding area |
| mouth     | Mouth and lower face      |
| left_ear  | Left ear                  |
| right_ear | Right ear                 |
| forehead  | Forehead / upper face     |

### 3.3 Landmark JSON Format

```json
{
  "image_path": "/home/.../dog/029364/02.png",
  "image_width": 224,
  "image_height": 224,
  "landmarks": [...],
  "region_bboxes": {
    "left_eye": {"x_min": 147, "y_min": 88, "x_max": 169, "y_max": 92, "width": 22, "height": 4},
    "right_eye": {...},
    "nose": {...},
    "mouth": {...},
    "left_ear": {...},
    "right_ear": {...},
    "forehead": {...}
  },
  "avg_confidence": 0.5893,
  "visible_landmarks": 42
}
```

---

## 4. Training & Evaluation

### 4.1 Quick Start

**Train (frozen backbone recommended first)**
```bash
uv run python launch.py   --output_dir ./outputs/exp_regional_default   --epochs 60   --freeze_backbone true
```

**Evaluate only**
```bash
uv run python launch.py   --output_dir ./outputs/exp_regional_default   --eval_only   --resume ./outputs/exp_regional_default/checkpoints/last.pth
```

### 4.2 Hyperparameter Overrides

`launch.py` forwards unknown flags to the config layer, so **any hyperparameter** can be overridden:

```bash
uv run python launch.py   --output_dir ./outputs/exp_tuned   --epochs 40   --base_lr 0.0003   --weight_decay 0.05   --triplet_margin 0.5   --bn_neck true   --image_size 256 256   --regional_size 128 128   --steps 30 50
```

### 4.3 Hyperparameter Sweeps

Use `hp_grid_search/sweep.sh` to run LR/margin/BN/epochs grids and optional Center Loss:

```bash
bash docs/sweep.sh   --gpu 0   --epochs 20,40,60   --lrs 0.0002,0.0003,0.0005   --margins 0.3,0.5,0.8   --bns true,false   --center-sweep true --center-ws 0.0005,0.001   --root-out ./outputs/sweeps_$(date +%Y%m%d_%H%M%S)   --extra "--weight_decay 0.05 --warmup_iters 500"
```

- Use `--dry-run` to print commands without running.  
- Per-run logs are saved under each run’s output directory.

---

## 6. Research Backbones

Backbones with different capacity / “research headroom”:

- **SWIN (hierarchical ViT)**  
  - `swin_base_patch4_window7_224` (1024D)  
  - `swin_large_patch4_window7_224` (1536D)  
  - `swin_large_patch4_window12_384` (1536D)  
  > Typically expects 224×224 input due to window attention. Use a SWIN-specific config when necessary.

- **DINOv3-B**  
  - `dinov3_vitb16` (768D) — strong baseline, efficient.

**Backbone quick test**
```bash
uv run python test_research_backbones.py
```


## 7. Troubleshooting

**Missing directories / files**
- Confirm `.../petface/dog` exists and contains per-identity folders.
- Ensure split CSVs are present under `splits_petface/`.

**Missing or invalid landmarks**
- Landmark JSONs are required. The loader raises explicit errors if:
  - Landmarks directory is missing.
  - JSON for an image is missing.
  - Region bboxes are invalid.

**Out-of-memory (OOM)**
- Reduce `--ims_per_batch`.
- Freeze backbone: `--freeze_backbone true`.
- Lower `--image_size` / `--regional_size`.

**Slow convergence / unstable losses**
- Adjust `--base_lr` (e.g., 1e-4 to 3e-4).  
- Use shorter schedules or earlier LR steps: `--steps 30 50`.  
- Inspect data with `docs/demos/`.

---

## 8. Next Steps

1. Generate PetFace splits and verify structure.  
2. Test the data loader (global and regional).  
3. Train a frozen-backbone baseline; monitor mAP/CMC on validation split.  
4. Add regional features (with strict landmark checks) and compare.  
5. Explore backbone variants and research ideas (losses, attention, augmentations).

---

**Further reading (see `docs/`):**
- `PETFACE_SETUP.md` – Dataset setup and verification  
- `REGIONAL_EXTRACTION_GUIDE.md` – Regional pipeline details  
- `TRAINING_GUIDE.md` – Training/evaluation workflow and tips  
- `RESEARCH_BACKBONES.md` – Backbone options and guidance  
- `SUMMARY.md` – High-level overview and results
