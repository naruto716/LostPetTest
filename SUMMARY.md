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

```python
from config_petface import cfg
from datasets.make_dataloader_petface import make_petface_dataloaders

# Load data
(train_loader, val_query, val_gallery, 
 test_query, test_gallery, num_classes) = make_petface_dataloaders(cfg)

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
IMAGE_SIZE = (256, 256)
IMS_PER_BATCH = 64          # P×K
NUM_INSTANCE = 4            # K instances per ID
ROTATION_DEGREE = 10
BRIGHTNESS = 0.2
PADDING = 10
COORD_SAFE_MODE = False     # Set True for bbox tracking later
```

### Splits Generated

Run on SageMaker:
```bash
cd /home/sagemaker-user/LostPet
python3 create_petface_splits.py
```

Creates:
- `splits_petface/train.csv` - 70% of dogs
- `splits_petface/val_query.csv` - 10% of dogs (query)
- `splits_petface/val_gallery.csv` - 10% of dogs (gallery)
- `splits_petface/test_query.csv` - 20% of dogs (query)
- `splits_petface/test_gallery.csv` - 20% of dogs (gallery)

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

