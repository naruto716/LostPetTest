"""
Dataset loaders for multi-part ReID using bounding box annotations.
"""

import json
from pathlib import Path
from PIL import Image
import pandas as pd
from torch.utils.data import Dataset

class MultiPartReIDDataset(Dataset):
    def __init__(self, split_csv, img_dir, label_dir, regions=None, transform=None, crop_transform=None):
        self.split_df = pd.read_csv(split_csv)
        self.img_dir = Path(img_dir)
        self.label_dir = Path(label_dir)
        self.transform = transform
        self.crop_transform = crop_transform
        self.regions = regions  # List of region names to extract, or None for all

    def __len__(self):
        return len(self.split_df)

    def __getitem__(self, idx):
        row = self.split_df.iloc[idx]
        img_path = self.img_dir / row['path']
        pid = row['pid']
        img = Image.open(img_path).convert('RGB')
        label_path = self.label_dir / (img_path.parent.name + '_' + img_path.stem + '.json')
        with open(label_path) as f:
            label = json.load(f)
        region_bboxes = label.get('region_bboxes', {})
        crops = []
        # Use specified regions or all available
        region_names = self.regions if self.regions is not None else list(region_bboxes.keys())
        for region in region_names:
            if region in region_bboxes:
                bbox = region_bboxes[region]
                crop = img.crop((bbox['x_min'], bbox['y_min'], bbox['x_max'], bbox['y_max']))
                if self.crop_transform:
                    crop = self.crop_transform(crop)
                crops.append(crop)
        if self.transform:
            img = self.transform(img)
        return img, crops, pid

class PreCroppedMultiPartReIDDataset(Dataset):
    """
    Dataset for pre-cropped part images. Assumes crops are saved as separate files and split CSV lists all crops.
    Each row in the split CSV should have columns: 'path', 'pid', 'region' (region name for the crop).
    """
    def __init__(self, split_csv, crop_dir, transform=None):
        self.split_df = pd.read_csv(split_csv)
        self.crop_dir = Path(crop_dir)
        self.transform = transform

    def __len__(self):
        return len(self.split_df)

    def __getitem__(self, idx):
        row = self.split_df.iloc[idx]
        crop_path = self.crop_dir / row['path']
        pid = row['pid']
        region = row['region'] if 'region' in row else None
        crop = Image.open(crop_path).convert('RGB')
        if self.transform:
            crop = self.transform(crop)
        return crop, pid, region

# Example usage for MultiPartReIDDataset:
# dataset = MultiPartReIDDataset(
#     split_csv='subset_splits_petface/train_subset.csv',
#     img_dir='dog',
#     label_dir='dog_landmarks',
#     regions=['nose', 'mouth', 'right_ear'],  # or None for all
#     transform=your_img_transform,
#     crop_transform=your_crop_transform
# )
# img, crops, pid = dataset[0]

# Example usage for PreCroppedMultiPartReIDDataset:
# dataset = PreCroppedMultiPartReIDDataset(
#     split_csv='subset_splits_petface/train_crops.csv',
#     crop_dir='dog_crops',
#     transform=your_crop_transform
# )
# crop, pid, region = dataset[0]
