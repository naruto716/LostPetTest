# Documentation & Demos

This folder contains documentation and demo files for the project.

## Contents

### `/demos/`
Demo scripts for understanding concepts:
- `demo_albumentations.py` - Shows how albumentations handles coordinate transformations
- `example_petface_usage.py` - Example of using the petface dataloader

### Documentation Files
- `PETFACE_SETUP.md` - Complete setup guide for PetFace dataset

## Usage

Run demos with:
```bash
uv run python docs/demos/demo_albumentations.py
uv run python docs/demos/example_petface_usage.py
```

# Training logs： 

```
================================================================================
📊 VALIDATION RESULTS - Epoch 50
================================================================================
   mAP:      95.32%
   Rank-1:   99.00%
   Rank-5:   100.00%
   Rank-10:  100.00%
   Eval time: 5.8s
================================================================================

.
.
.

================================================================================
VALIDATION RESULTS - Epoch 120
================================================================================
   mAP:      95.22%
   Rank-1:   99.00%
   Rank-5:   100.00%
   Rank-10:  100.00%
   Eval time: 5.8s

================================================================================   
```
