"""
Simple demo showing how albumentations handles coordinate transformation.
Run this to understand the concept before integrating into dataloader.
"""

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# First, let's show the problem with torchvision
print("=" * 60)
print("PROBLEM: Manual coordinate tracking with torchvision")
print("=" * 60)

# Create dummy image and bbox
img = np.random.randint(0, 255, (400, 300, 3), dtype=np.uint8)  # H=400, W=300
bbox_orig = [50, 100, 200, 300]  # [x1, y1, x2, y2]

print(f"Original image size: {img.shape[:2]}")  # (H, W)
print(f"Original bbox: {bbox_orig}")

# With torchvision, if you resize:
from torchvision import transforms
import torch

# Method 1: torchvision (manual tracking needed)
print("\n--- torchvision approach ---")
img_pil = Image.fromarray(img)
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
])
img_transformed = transform(img_pil)
print(f"Transformed image size: {img_transformed.shape}")  # [C, H, W]

# YOU have to manually calculate new bbox:
scale_y = 256 / 400
scale_x = 256 / 300
bbox_new = [
    bbox_orig[0] * scale_x,  # x1
    bbox_orig[1] * scale_y,  # y1
    bbox_orig[2] * scale_x,  # x2
    bbox_orig[3] * scale_y,  # y2
]
print(f"Manually calculated bbox: {bbox_new}")

print("\n" + "=" * 60)
print("SOLUTION: Albumentations does this automatically")
print("=" * 60)

# Method 2: albumentations (automatic!)
try:
    import albumentations as A
    from albumentations.pytorch import ToTensorV2
    
    print("\n--- albumentations approach ---")
    
    # Define transforms WITH bbox tracking
    transform_albu = A.Compose([
        A.Resize(256, 256),
        ToTensorV2(),  # PyTorch tensor conversion
    ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['class_labels']))
    
    # Apply transform - bbox automatically adjusted!
    transformed = transform_albu(
        image=img, 
        bboxes=[bbox_orig],
        class_labels=[0]  # Dummy class (e.g., 0 for "body")
    )
    
    img_transformed = transformed['image']
    bbox_transformed = transformed['bboxes'][0]
    
    print(f"Transformed image size: {img_transformed.shape}")
    print(f"Automatically calculated bbox: {list(bbox_transformed)}")
    print("\n✅ No manual calculation needed!")
    
except ImportError:
    print("\n⚠️  albumentations not installed")
    print("Install with: pip install albumentations")

print("\n" + "=" * 60)
print("WHY THIS MATTERS FOR AUGMENTATIONS")
print("=" * 60)

try:
    import albumentations as A
    
    # Complex augmentation pipeline
    transform_complex = A.Compose([
        A.Resize(256, 256),
        A.Rotate(limit=10, p=1.0),              # Rotates image AND bbox!
        A.RandomCrop(200, 200, p=1.0),          # Crops image AND bbox!
        A.HorizontalFlip(p=1.0),                # Flips image AND bbox!
    ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['class_labels']))
    
    print("\nWith complex augmentations:")
    print("  - Resize")
    print("  - Rotate ±10°")  
    print("  - Random crop to 200x200")
    print("  - Horizontal flip")
    
    transformed = transform_complex(
        image=img,
        bboxes=[bbox_orig],
        class_labels=[0]
    )
    
    print(f"\nOriginal bbox: {bbox_orig}")
    print(f"After all augmentations: {list(transformed['bboxes'][0])}")
    print("\n✅ All coordinate transformations handled automatically!")
    
except ImportError:
    pass
except Exception as e:
    print(f"\n(Example transform failed - this is expected: {e})")

print("\n" + "=" * 60)
print("FORMATS SUPPORTED")
print("=" * 60)
print("""
Bounding boxes:
  - 'pascal_voc': [x_min, y_min, x_max, y_max]
  - 'coco': [x_min, y_min, width, height]
  - 'yolo': [x_center, y_center, width, height] (normalized)
  
Keypoints:
  - 'xy': (x, y)
  - 'yx': (y, x)
  - 'xya': (x, y, angle)
  - 'xys': (x, y, scale)
""")

print("\n" + "=" * 60)
print("NEXT STEPS")
print("=" * 60)
print("""
1. Install albumentations: pip install albumentations
2. Decide on annotation format (bbox? keypoints?)
3. Integrate into dataloader gradually
4. Test with visualization
""")

