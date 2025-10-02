# PetFace Dataset Setup Guide

## Overview
This guide shows how to set up and use the PetFace dataset for dog re-identification.

## Files Created

### 1. Data Split Generation
- **`create_petface_splits.py`**: Creates train/val/test splits with proper query/gallery separation
  - Splits by dog ID (7:1:2 ratio)
  - Generates 5 CSV files in `splits_petface/`:
    - `train.csv` - All images from training dogs
    - `val_query.csv` - First image of each validation dog
    - `val_gallery.csv` - Remaining images from validation dogs
    - `test_query.csv` - First image of each test dog
    - `test_gallery.csv` - Remaining images from test dogs

### 2. Dataset Configuration
- **`config_petface.py`**: Configuration for PetFace training
  - Dataset paths (root: `/home/sagemaker-user/LostPet/petface`)
  - Image preprocessing settings
  - Model architecture settings
  - Training hyperparameters

### 3. DataLoader
- **`datasets/make_dataloader_petface.py`**: DataLoader implementation
  - Uses existing `DogMultiPose` dataset class (format compatible)
  - Creates 5 dataloaders:
    - Training (with PK sampling)
    - Val query + gallery (for validation during training)
    - Test query + gallery (for final evaluation)

### 4. Example Usage
- **`example_petface_usage.py`**: Shows how to use the dataloader

## Setup Steps

### Step 1: Generate Splits
On your SageMaker instance:
```bash
cd /home/sagemaker-user/LostPet
python3 create_petface_splits.py
```

This will create `splits_petface/` directory with all CSV files.

### Step 2: Verify Data Structure
Ensure your data is organized as:
```
/home/sagemaker-user/LostPet/petface/
├── dog/
│   ├── 000001/
│   │   ├── 00.png
│   │   ├── 01.png
│   │   └── ...
│   ├── 000002/
│   │   ├── 00.png
│   │   └── ...
│   └── ...
└── splits_petface/
    ├── train.csv
    ├── val_query.csv
    ├── val_gallery.csv
    ├── test_query.csv
    └── test_gallery.csv
```

### Step 3: Test DataLoader
```bash
python3 example_petface_usage.py
```

### Step 4: Use in Training
```python
from config_petface import cfg
from datasets.make_dataloader_petface import make_petface_dataloaders

# Load data
(train_loader, val_query, val_gallery, 
 test_query, test_gallery, num_classes) = make_petface_dataloaders(cfg)

# Training loop
for epoch in range(cfg.MAX_EPOCHS):
    # Train
    for imgs, pids, camids, paths in train_loader:
        # Your training code with triplet/ID loss
        pass
    
    # Validate (every N epochs)
    if epoch % cfg.EVAL_PERIOD == 0:
        # Evaluate on val_query vs val_gallery
        cmc, mAP = evaluate(model, val_query, val_gallery)
        print(f"Val mAP: {mAP:.1%}")

# Final test
cmc, mAP = evaluate(model, test_query, test_gallery)
print(f"Test mAP: {mAP:.1%}")
```

## Key Features

### 1. Proper Train/Val/Test Separation
- **Training**: Uses all images with PK sampling (P identities × K instances per batch)
- **Validation**: Separate query/gallery from val set (used during training)
- **Test**: Separate query/gallery from test set (used only for final evaluation)
- **No data leakage**: Dogs are split by ID, not by images

### 2. Output Format
Each batch returns:
```python
(imgs, pids, camids, paths)
```
- `imgs`: Tensor `[B, 3, H, W]` - Preprocessed images
- `pids`: Tensor `[B]` - Dog identity IDs
- `camids`: Tensor `[B]` - Camera IDs (all 0s for PetFace)
- `paths`: List `[B]` - Original image paths

### 3. Relabeling
- **Training**: `relabel=True` → PIDs are remapped to dense 0,1,2,...
- **Val/Test**: `relabel=False` → Original PIDs preserved (needed for query/gallery matching)

## Configuration

Edit `config_petface.py` to adjust:
- Image size: `IMAGE_SIZE = (256, 256)`
- Batch size: `IMS_PER_BATCH = 64` (P×K)
- Instances per ID: `NUM_INSTANCE = 4` (K)
- Model backbone: `BACKBONE = 'resnet50'`
- Learning rate: `BASE_LR = 3.5e-4`
- Loss weights: `ID_LOSS_WEIGHT`, `TRIPLET_LOSS_WEIGHT`

## Differences from Original Dataset

| Feature | Original (DogMultiPose) | PetFace |
|---------|------------------------|---------|
| Root dir | `.` (local) | `/home/sagemaker-user/LostPet/petface` |
| Images dir | `images/` | `dog/` (dog ID folders) |
| Camera IDs | May have real camids | All 0s (not applicable) |
| Validation | No val query/gallery | Proper val query/gallery |
| CSV structure | Same | Same |

## Next Steps

1. ✅ Generate splits with `create_petface_splits.py`
2. ✅ Test dataloader with `example_petface_usage.py`
3. 🔜 Train model using the dataloader
4. 🔜 Evaluate on validation set during training
5. 🔜 Final evaluation on test set

## Troubleshooting

**Issue**: "Directory does not exist"
- Check that `/home/sagemaker-user/LostPet/petface/dog` exists
- Verify dog ID folders are present

**Issue**: "No images found for dog ID"
- Check image file extensions (`.png`, `.jpg`, etc.)
- Verify images exist in dog ID folders

**Issue**: "Import error"
- Ensure you're in the correct directory
- Check that all required dependencies are installed

