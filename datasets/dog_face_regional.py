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
    
    Loads face images and extracts 7 regions based on landmarks with bbox expansion.
    - left_eye, right_eye, nose, mouth, left_ear, right_ear, forehead
    """
    
    def __init__(
        self, 
        valid_json_path: str,
        image_dir: str,
        landmarks_dir: str,
        transform_global: Optional[T.Compose] = None,
        transform_regional: Optional[T.Compose] = None,
        is_train: bool = True,
        region_size: Tuple[int, int] = (224, 224)
    ):
        """
        Args:
            valid_json_path: Path to filtered valid images JSON (from filter script)
            image_dir: Root directory for images
            landmarks_dir: Directory containing landmark JSONs (REQUIRED)
            transform_global: Transform for full face image
            transform_regional: Transform for regional images
            is_train: Whether this is training data
            region_size: Target size for regional images (global image size)
        """
        self.image_dir = Path(image_dir)
        self.landmarks_dir = Path(landmarks_dir)
        self.is_train = is_train
        self.region_size = region_size
        
        # Validate landmarks directory
        if not self.landmarks_dir.exists():
            raise ValueError(f"Landmarks directory does not exist: {self.landmarks_dir}")
        
        # Load filtered valid images JSON
        with open(valid_json_path, 'r') as f:
            data = json.load(f)
        
        # Flatten to list of (dog_id, photo_id, pid_label) tuples
        self.samples = []
        dog_ids = sorted(data['dog_images'].keys())
        
        # Create pid mapping (dog_id -> integer label)
        self.dog_id_to_pid = {dog_id: i for i, dog_id in enumerate(dog_ids)}
        self.num_pids = len(dog_ids)
        
        for dog_id in dog_ids:
            photo_ids = data['dog_images'][dog_id]
            pid_label = self.dog_id_to_pid[dog_id]
            for photo_id in photo_ids:
                self.samples.append((dog_id, photo_id, pid_label))
        
        # Define regions
        self.regions = ['left_eye', 'right_eye', 'nose', 'mouth', 
                       'left_ear', 'right_ear', 'forehead']
        
        # Bbox expansion ratios (manually tuned)
        self.expansion_ratios = {
            'left_eye': 0.8,
            'right_eye': 0.8,
            'nose': 0.15,
            'mouth': 0.1,
            'left_ear': 0.3,
            'right_ear': 0.3,
            'forehead': 0.2
        }
        
        # Transforms
        self.transform_global = transform_global or self._default_transform()
        self.transform_regional = transform_regional or self._default_transform((64, 64))
        
        logger.info(f"Loaded {len(self.samples)} valid images from {self.num_pids} dogs")
        logger.info(f"Using landmarks from: {self.landmarks_dir}")
    
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
    
    def __len__(self):
        return len(self.samples)
    
    def expand_bbox(self, bbox: Dict, expansion_ratio: float, img_width: int, img_height: int) -> Dict:
        """
        Expand bbox by percentage while staying within image bounds.
        
        Args:
            bbox: Original bbox with x_min, y_min, x_max, y_max, width, height
            expansion_ratio: Percentage to expand (e.g., 0.2 = 20%)
            img_width, img_height: Image dimensions
        
        Returns:
            Expanded bbox dict
        """
        width = bbox['width']
        height = bbox['height']
        
        # Calculate expansion in pixels (split evenly on both sides)
        expand_w = int(width * expansion_ratio / 2)
        expand_h = int(height * expansion_ratio / 2)
        
        # Expand bbox
        x_min = max(0, bbox['x_min'] - expand_w)
        y_min = max(0, bbox['y_min'] - expand_h)
        x_max = min(img_width, bbox['x_max'] + expand_w)
        y_max = min(img_height, bbox['y_max'] + expand_h)
        
        return {
            'x_min': x_min,
            'y_min': y_min,
            'x_max': x_max,
            'y_max': y_max,
            'width': x_max - x_min,
            'height': y_max - y_min
        }
    
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
        """Extract regions using actual landmarks with bbox expansion."""
        regions = {}
        region_bboxes = landmarks.get('region_bboxes', {})
        
        if not region_bboxes:
            raise ValueError("No region_bboxes found in landmark JSON")
        
        img_width, img_height = image.size
        missing_regions = []
        
        for region_name in self.regions:
            if region_name in region_bboxes:
                orig_bbox = region_bboxes[region_name]
                
                # Expand bbox
                expansion_ratio = self.expansion_ratios.get(region_name, 0.2)
                expanded_bbox = self.expand_bbox(orig_bbox, expansion_ratio, img_width, img_height)
                
                x_min = expanded_bbox['x_min']
                y_min = expanded_bbox['y_min']
                x_max = expanded_bbox['x_max']
                y_max = expanded_bbox['y_max']
                
                # Validate bbox
                if x_max <= x_min or y_max <= y_min:
                    raise ValueError(f"Invalid bbox for {region_name}: {expanded_bbox}")
                
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
        dog_id, photo_id, pid = self.samples[idx]
        
        # Construct image path
        img_path = self.image_dir / dog_id / f"{photo_id}.png"
        
        # Load full image
        image = Image.open(img_path).convert('RGB')
        
        # Load landmarks
        landmark_path = self.landmarks_dir / f"{dog_id}_{photo_id}.json"
        with open(landmark_path, 'r') as f:
            landmarks = json.load(f)
        
        # Extract regions with expansion
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
                'camid': 0,  # Not used for PetFace
                'img_path': str(img_path)
            }
        else:
            # For evaluation, return in format expected by inference
            return image_tensor, region_tensors, pid, str(img_path)
    
    def visualize_sample(self, idx: int, save_path: Optional[str] = None):
        """Visualize a sample with its regions."""
        import matplotlib.pyplot as plt
        import matplotlib.patches as patches
        
        # Get raw data without transforms
        dog_id, photo_id, pid = self.samples[idx]
        img_path = self.image_dir / dog_id / f"{photo_id}.png"
        image = Image.open(img_path).convert('RGB')
        
        # Load landmarks
        landmark_path = self.landmarks_dir / f"{dog_id}_{photo_id}.json"
        with open(landmark_path, 'r') as f:
            landmarks = json.load(f)
        
        # Get regions with expansion
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
def make_regional_dataloader(valid_json_path, image_dir, landmarks_dir, batch_size=32, 
                           is_train=True, num_workers=4):
    """Create a regional dataloader."""
    dataset = DogFaceRegionalDataset(
        valid_json_path=valid_json_path,
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
