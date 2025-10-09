"""
Dog Face Regional Dataset with landmark-based region extraction.
Supports both full landmark JSONs and mock regions for testing.
"""

import os
import json
import numpy as np
from PIL import Image, ImageDraw
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T
from pathlib import Path
import pandas as pd
from typing import Dict, List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class DogFaceRegionalDataset(Dataset):
    """
    Dataset for Dog Face ReID with regional feature extraction.
    
    Loads face images and extracts 7 regions based on landmarks:
    - left_eye, right_eye, nose, mouth, left_ear, right_ear, forehead
    """
    
    def __init__(
        self, 
        csv_path: str,
        image_dir: str,
        landmarks_dir: str,
        transform_global: Optional[T.Compose] = None,
        transform_regional: Optional[T.Compose] = None,
        is_train: bool = True,
        region_size: Tuple[int, int] = (64, 64)
    ):
        """
        Args:
            csv_path: Path to train/val/test CSV file
            image_dir: Root directory for images
            landmarks_dir: Directory containing landmark JSONs (REQUIRED)
            transform_global: Transform for full face image
            transform_regional: Transform for regional images
            is_train: Whether this is training data
            region_size: Target size for regional images
        """
        self.image_dir = Path(image_dir)
        self.landmarks_dir = Path(landmarks_dir)
        self.is_train = is_train
        self.region_size = region_size
        
        # Validate landmarks directory
        if not self.landmarks_dir.exists():
            raise ValueError(f"Landmarks directory does not exist: {self.landmarks_dir}")
        
        # Load CSV data
        self.df = pd.read_csv(csv_path)
        
        # Define regions
        self.regions = ['left_eye', 'right_eye', 'nose', 'mouth', 
                       'left_ear', 'right_ear', 'forehead']
        
        # Transforms
        self.transform_global = transform_global or self._default_transform()
        self.transform_regional = transform_regional or self._default_transform(self.region_size)
        
        logger.info(f"Using landmarks from: {self.landmarks_dir}")
        
        # Validate that we have landmarks for at least some samples
        self._validate_landmarks_availability()
    
    def _default_transform(self, size=(224, 224)):
        """Default transform for images."""
        if self.is_train:
            return T.Compose([
                T.Resize(size),
                T.RandomHorizontalFlip(p=0.5),
                T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        else:
            return T.Compose([
                T.Resize(size),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
    
    def _validate_landmarks_availability(self):
        """Check if we have landmarks for at least some samples."""
        # Check first 10 samples
        samples_to_check = min(10, len(self.df))
        found_landmarks = 0
        
        for i in range(samples_to_check):
            row = self.df.iloc[i]
            parts = row['img_rel_path'].split('/')
            dog_id = parts[0]
            photo_id = parts[1].split('.')[0]
            
            if self._check_landmark_exists(dog_id, photo_id):
                found_landmarks += 1
        
        if found_landmarks == 0:
            raise ValueError(
                f"No landmark JSONs found for first {samples_to_check} samples. "
                f"Please ensure landmark JSONs are in {self.landmarks_dir} "
                f"with format: <dog_id>_<photo_id>.json"
            )
        
        logger.info(f"Found landmarks for {found_landmarks}/{samples_to_check} checked samples")
    
    def _check_landmark_exists(self, dog_id: str, photo_id: str) -> bool:
        """Check if landmark JSON exists for given image."""
        json_path = self.landmarks_dir / f"{dog_id}_{photo_id}.json"
        if not json_path.exists():
            # Try without leading zeros
            dog_id_int = str(int(dog_id))
            photo_id_int = str(int(photo_id))
            json_path = self.landmarks_dir / f"{dog_id_int}_{photo_id_int}.json"
        return json_path.exists()
    
    def __len__(self):
        return len(self.df)
    
    def _load_landmarks(self, dog_id: str, photo_id: str) -> Dict:
        """Load landmarks JSON for an image. Raises exception if not found."""
        json_path = self.landmarks_dir / f"{dog_id}_{photo_id}.json"
        if not json_path.exists():
            # Try without leading zeros
            dog_id_int = str(int(dog_id))
            photo_id_int = str(int(photo_id))
            json_path = self.landmarks_dir / f"{dog_id_int}_{photo_id_int}.json"
        
        if not json_path.exists():
            raise FileNotFoundError(
                f"Landmark JSON not found: {json_path}\n"
                f"Expected format: {dog_id}_{photo_id}.json or {dog_id_int}_{photo_id_int}.json"
            )
        
        with open(json_path, 'r') as f:
            return json.load(f)
    
    
    def _extract_regions_from_landmarks(self, image: Image.Image, landmarks: Dict) -> Dict:
        """Extract regions using actual landmarks."""
        regions = {}
        region_bboxes = landmarks.get('region_bboxes', {})
        
        if not region_bboxes:
            raise ValueError("No region_bboxes found in landmark JSON")
        
        missing_regions = []
        for region_name in self.regions:
            if region_name in region_bboxes:
                bbox = region_bboxes[region_name]
                x_min = max(0, bbox['x_min'])
                y_min = max(0, bbox['y_min'])
                x_max = min(image.width, bbox['x_max'])
                y_max = min(image.height, bbox['y_max'])
                
                # Validate bbox
                if x_max <= x_min or y_max <= y_min:
                    raise ValueError(f"Invalid bbox for {region_name}: {bbox}")
                
                region_img = image.crop((x_min, y_min, x_max, y_max))
                regions[region_name] = {
                    'image': region_img,
                    'bbox': (x_min, y_min, x_max, y_max)
                }
            else:
                missing_regions.append(region_name)
        
        if missing_regions:
            raise ValueError(f"Missing regions in landmarks: {missing_regions}")
        
        return regions
    
    def __getitem__(self, idx):
        """Get image and regional features."""
        row = self.df.iloc[idx]
        img_path = self.image_dir / row['img_rel_path']
        pid = row['pid']
        camid = row['camid']
        
        # Load full image
        image = Image.open(img_path).convert('RGB')
        
        # Extract dog_id and photo_id from path
        # Format: 011103/00.png -> dog_id=011103, photo_id=00
        parts = row['img_rel_path'].split('/')
        dog_id = parts[0]
        photo_id = parts[1].split('.')[0]
        
        # Get regions from landmarks (required)
        landmarks = self._load_landmarks(dog_id, photo_id)
        regions = self._extract_regions_from_landmarks(image, landmarks)
        
        # Apply transforms
        if self.transform_global:
            image_tensor = self.transform_global(image)
        else:
            image_tensor = T.ToTensor()(image)
        
        # Process regional images
        region_tensors = {}
        for region_name, region_data in regions.items():
            if self.transform_regional:
                region_tensor = self.transform_regional(region_data['image'])
            else:
                region_tensor = T.ToTensor()(region_data['image'])
            region_tensors[region_name] = region_tensor
        
        if self.is_train:
            return {
                'image': image_tensor,
                'regions': region_tensors,
                'pid': pid,
                'camid': camid,
                'img_path': str(img_path)
            }
        else:
            # For evaluation, return in format expected by inference
            return image_tensor, pid, camid, str(img_path)
    
    def visualize_sample(self, idx: int, save_path: Optional[str] = None):
        """Visualize a sample with its regions."""
        import matplotlib.pyplot as plt
        import matplotlib.patches as patches
        
        # Get raw data without transforms
        row = self.df.iloc[idx]
        img_path = self.image_dir / row['img_rel_path']
        image = Image.open(img_path).convert('RGB')
        
        # Get regions
        parts = row['img_rel_path'].split('/')
        dog_id = parts[0]
        photo_id = parts[1].split('.')[0]
        
        landmarks = self._load_landmarks(dog_id, photo_id)
        regions = self._extract_regions_from_landmarks(image, landmarks)
        
        # Create visualization
        fig, axes = plt.subplots(2, 4, figsize=(16, 8))
        axes = axes.flatten()
        
        # Show original with bboxes
        ax = axes[0]
        ax.imshow(image)
        ax.set_title(f"Original: {dog_id}/{photo_id}")
        ax.axis('off')
        
        # Draw bboxes
        colors = ['red', 'blue', 'green', 'yellow', 'purple', 'orange', 'cyan']
        for (region_name, region_data), color in zip(regions.items(), colors):
            x_min, y_min, x_max, y_max = region_data['bbox']
            rect = patches.Rectangle(
                (x_min, y_min), x_max - x_min, y_max - y_min,
                linewidth=2, edgecolor=color, facecolor='none'
            )
            ax.add_patch(rect)
            ax.text(x_min, y_min - 2, region_name, color=color, fontsize=8)
        
        # Show each region
        for i, region_name in enumerate(self.regions):
            ax = axes[i + 1]
            if region_name in regions:
                ax.imshow(regions[region_name]['image'])
                ax.set_title(region_name)
            ax.axis('off')
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved visualization to {save_path}")
        else:
            plt.show()


# Convenience functions for creating datasets
def make_regional_dataloader(csv_path, image_dir, landmarks_dir, batch_size=32, 
                           is_train=True, num_workers=4):
    """Create a regional dataloader."""
    dataset = DogFaceRegionalDataset(
        csv_path=csv_path,
        image_dir=image_dir,
        landmarks_dir=landmarks_dir,
        is_train=is_train
    )
    
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=is_train,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=is_train
    )
    
    return dataloader, dataset
