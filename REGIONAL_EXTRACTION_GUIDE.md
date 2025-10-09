# Dog Face Regional Feature Extraction

## Overview

Regional feature extraction for dog faces using landmark-based bounding boxes. Extracts 7 facial regions (eyes, nose, mouth, ears, forehead) inspired by Amur Tiger ReID.

**⚠️ REQUIRES landmark JSONs - no fallbacks to prevent noisy training data**

## Architecture

### 1. Regional Dataset (`datasets/dog_face_regional.py`)

The `DogFaceRegionalDataset` class handles:
- Loading PetFace images from CSV files
- Extracting 7 facial regions using landmark JSONs (REQUIRED)
- Validating landmark availability before training
- Applying transforms to global and regional images

```python
# Basic usage
from datasets.dog_face_regional import DogFaceRegionalDataset

dataset = DogFaceRegionalDataset(
    csv_path='splits_petface/train.csv',
    image_dir='images',
    landmarks_dir='/path/to/petface_landmarks_json_all',  # REQUIRED
    is_train=True
)

# Get a sample
sample = dataset[0]
# Returns:
# {
#   'image': torch.Tensor,          # Full face image
#   'regions': {                    # Regional images
#     'left_eye': torch.Tensor,
#     'right_eye': torch.Tensor,
#     'nose': torch.Tensor,
#     'mouth': torch.Tensor,
#     'left_ear': torch.Tensor,
#     'right_ear': torch.Tensor,
#     'forehead': torch.Tensor
#   },
#   'pid': str,                     # Dog ID
#   'camid': int,                   # Camera ID
#   'img_path': str                 # Original image path
# }
```

### 2. Face Regions

The system extracts 7 key facial regions:

| Region | Description | Typical Size |
|--------|-------------|--------------|
| left_eye | Dog's left eye region | ~30×20 pixels |
| right_eye | Dog's right eye region | ~30×20 pixels |
| nose | Nose and surrounding area | ~40×30 pixels |
| mouth | Mouth and lower face | ~80×40 pixels |
| left_ear | Left ear region | ~50×70 pixels |
| right_ear | Right ear region | ~50×70 pixels |
| forehead | Forehead and upper face | ~100×40 pixels |

### 3. Strict Landmark Requirements

The dataset will raise exceptions if:
- Landmarks directory doesn't exist
- No landmark JSONs found for initial samples
- Missing landmark JSON for any accessed image
- Missing regions in a landmark JSON
- Invalid bounding box coordinates

This strict approach ensures:
- No training on inaccurate mock data
- Early detection of data issues
- Consistent high-quality training data

## Landmark JSON Format

The system expects landmark JSONs in this format:

```json
{
  "image_path": "/home/sagemaker-user/LostPet/PetFace/dog/029364/02.png",
  "image_width": 224,
  "image_height": 224,
  "landmarks": [...],
  "region_bboxes": {
    "left_eye": {
      "x_min": 147,
      "y_min": 88,
      "x_max": 169,
      "y_max": 92,
      "width": 22,
      "height": 4
    },
    "right_eye": {...},
    "nose": {...},
    "mouth": {...},
    "left_ear": {...},
    "right_ear": {...},
    "forehead": {...}
  },
  "avg_confidence": 0.5892922878265381,
  "visible_landmarks": 42
}
```

## Integration with Training Pipeline

### Step 1: Create Regional Model

```python
class DogFaceRegionalModel(nn.Module):
    def __init__(self, backbone='dinov2_vitb14', embed_dim=512):
        super().__init__()
        
        # Shared DINO backbone for all regions
        self.backbone = build_dino_backbone(backbone)
        
        # Fusion layer: concatenate all features and project
        # Input: (1 global + 7 regional) × dino_dim
        # Output: embed_dim
        dino_dim = 768  # for ViT-B
        self.fusion = nn.Sequential(
            nn.Linear(dino_dim * 8, embed_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(embed_dim * 2, embed_dim),
            nn.BatchNorm1d(embed_dim)
        )
        
    def forward(self, batch):
        # Extract global features
        global_feat = self.backbone(batch['image'])
        
        # Extract regional features
        regional_feats = []
        for region_name in ['left_eye', 'right_eye', 'nose', 
                           'mouth', 'left_ear', 'right_ear', 'forehead']:
            feat = self.backbone(batch['regions'][region_name])
            regional_feats.append(feat)
        
        # Concatenate and fuse
        all_feats = torch.cat([global_feat] + regional_feats, dim=1)
        fused_feat = self.fusion(all_feats)
        
        return fused_feat
```

### Step 2: Update Training Loop

```python
# In processor_dogreid.py, modify the training loop:

for batch in train_loader:
    # batch now contains:
    # - batch['image']: global images
    # - batch['regions']: dict of regional images
    # - batch['pid'], batch['camid']: labels
    
    # Forward pass
    features = model(batch)
    
    # Compute losses as usual
    # ...
```

## Testing the Implementation

### 1. Test Strict Requirements

```python
python test_regional_strict.py
```

This will:
- Verify that landmarks are required
- Show error messages when landmarks are missing
- Demonstrate proper usage

### 2. Test with Real Landmarks

When you have landmark JSONs available:

```python
dataset = DogFaceRegionalDataset(
    csv_path='splits_petface/train.csv',
    image_dir='images',
    landmarks_dir='/path/to/petface_landmarks_json_all'  # REQUIRED
)
```

### 3. Visualize a Sample

```python
# Visualize regions for a specific sample
dataset.visualize_sample(idx=0, save_path='sample_regions.png')
```

## Next Steps

1. **Copy landmark JSONs locally** or mount the sagemaker directory
2. **Test with real landmarks** to verify extraction accuracy
3. **Implement the regional model** based on the blueprint above
4. **Update the training pipeline** to use regional features
5. **Compare performance** with and without regional features

## Performance Expectations

Based on the Amur Tiger ReID results, regional features should provide:
- Better discrimination for similar-looking dogs
- Robustness to partial occlusions
- Improved mAP by 5-10% (typical improvement from regional approaches)

## Troubleshooting

### Missing Landmarks
- The system will raise a clear error message
- Check the landmark JSON filename format: `{dog_id}_{photo_id}.json`
- Ensure ALL training images have corresponding landmark JSONs

### Memory Issues
- Use smaller batch sizes when processing regional features
- Consider using a shared backbone instead of separate models per region
- Process regions sequentially instead of in parallel if needed

### Region Quality
- Verify landmark accuracy by visualizing samples
- Report any systematic issues with landmark generation
- Consider data augmentation specific to each region type
