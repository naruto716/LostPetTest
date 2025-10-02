# PetFace DataLoader - Quick Reference

## Current Setup (Ready for Raw Pipeline Evaluation)

### Files
- `datasets/make_dataloader_petface.py` - Main dataloader
- `config_petface.py` - Configuration
- `create_petface_splits.py` - Script to generate train/val/test splits

### Augmentations (Based on Amur Tiger ReID Best Practices)

**Training:**
- Resize to 256×256
- RandomRotation(±10°)
- ColorJitter (brightness/contrast/saturation/hue = 0.2)
- Pad(10) + RandomCrop (spatial augmentation)
- ToTensor + Normalize

**Test:**
- Resize to 256×256
- ToTensor + Normalize

**Disabled (for animal ReID):**
- ❌ RandomErasing
- ❌ HorizontalFlip

### Usage

**Training:**
```bash
# Basic training (uses validation set for checkpointing)
python train_petface.py

# With options
python train_petface.py --epochs 120 --output_dir ./output_petface

# Resume from checkpoint
python train_petface.py --resume ./output_petface/checkpoint_epoch_50.pth

# Training + final test evaluation
python train_petface.py --final_test
```

**Evaluation Only:**
```bash
# Evaluate on validation set
python train_petface.py --eval_only --eval_split val --resume ./output_petface/best_model.pth

# Evaluate on test set
python train_petface.py --eval_only --eval_split test --resume ./output_petface/best_model.pth
```

**Programmatic Usage:**
```python
from config_petface import cfg
from datasets.make_dataloader_petface import make_petface_dataloaders

# Load data (proper train/val/test separation)
(train_loader, 
 val_query_loader, val_gallery_loader,    # For validation during training
 test_query_loader, test_gallery_loader,  # For final evaluation
 num_classes) = make_petface_dataloaders(cfg)

# Training loop
for imgs, pids, camids, paths in train_loader:
    # imgs: [B, 3, 256, 256]
    # pids: [B] - identity IDs
    # camids: [B] - camera IDs (all 0s)
    # paths: [B] - image paths
    pass
```

### Configuration

Edit `config_petface.py`:
```python
# Model
BACKBONE = 'dinov3_vitl16'  # DINOv3-Large (proven winner)
EMBED_DIM = 768
FREEZE_BACKBONE = True      # Frozen backbone approach

# Data
IMAGE_SIZE = (256, 256)
IMS_PER_BATCH = 64          # P×K
NUM_INSTANCE = 4            # K instances per ID

# Augmentation
ROTATION_DEGREE = 10
BRIGHTNESS = 0.2
PADDING = 10
COORD_SAFE_MODE = False     # Set True for bbox tracking later

# Optimizer
OPTIMIZER_NAME = 'AdamW'
BASE_LR = 3e-4
WEIGHT_DECAY = 0.01
```

### Proper Train/Val/Test Separation

**Key Principle:** Test set should NEVER be used during training!

**Our Approach:**
1. **Training (70% of dogs)**
   - Used with PK sampling for learning
   
2. **Validation (10% of dogs)**
   - Split into query/gallery
   - Used DURING training for model selection and checkpointing
   - Best model selected based on val mAP
   
3. **Test (20% of dogs)**
   - Split into query/gallery
   - Used ONLY for final evaluation after training
   - Represents true unseen performance

**Why this matters:**
- Prevents data leakage
- Honest performance estimates
- Standard machine learning practice

### Splits Generated

Run on SageMaker:
```bash
cd /home/sagemaker-user/LostPet
python3 create_petface_splits.py
```

Creates (7:1:2 ratio by dog ID):
- `splits_petface/train.csv` - All images from 70% of dogs
- `splits_petface/val_query.csv` - First image of each dog (10% of dogs)
- `splits_petface/val_gallery.csv` - Remaining images (10% of dogs)
- `splits_petface/test_query.csv` - First image of each dog (20% of dogs)
- `splits_petface/test_gallery.csv` - Remaining images (20% of dogs)

## Future: Regional Pooling with Bboxes

When ready to add coordinate-based regional pooling:

1. **Simple approach**: Set `COORD_SAFE_MODE = True`
   - Disables rotation, crop, flip
   - Only uses resize (easy to scale coordinates)

2. **Advanced approach**: Use albumentations
   - Automatically tracks coordinates through all augmentations
   - See `docs/demos/demo_albumentations.py`

## Documentation

See `docs/` folder for:
- Demo scripts
- Detailed setup guides

