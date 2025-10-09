"""
Regional dataset for PetFace with landmark-based region extraction.
Extends DogMultiPose to add facial regions.
"""

import os
import json
from torch.utils.data import Dataset
from PIL import Image
from pathlib import Path


class DogFaceRegional(Dataset):
    """
    Regional ReID dataset for Dog faces.
    Extends the basic CSV loading with landmark-based region extraction.
    
    CSV columns: img_rel_path,pid,camid
    Returns: img, regions_dict, pid, camid, path
    """
    
    def __init__(
        self, 
        root, 
        split_csv, 
        landmarks_dir,
        images_dir="images", 
        transform_global=None,
        transform_regional=None,
        use_camid=False, 
        relabel=True
    ):
        """
        Args:
            root: Root directory
            split_csv: Path to CSV file relative to root
            landmarks_dir: Absolute path to landmarks directory
            images_dir: Directory name containing images
            transform_global: Transform for full face image
            transform_regional: Transform for regional images
            use_camid: Whether to use camera IDs
            relabel: Dense PID mapping (True for train, False for eval)
        """
        self.root = root
        self.images_dir = images_dir
        self.landmarks_dir = Path(landmarks_dir)
        self.transform_global = transform_global
        self.transform_regional = transform_regional
        self.use_camid = use_camid
        self.relabel = relabel
        
        # Validate landmarks directory
        if not self.landmarks_dir.exists():
            raise ValueError(f"Landmarks directory does not exist: {self.landmarks_dir}")
        
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
        
        # Load CSV data
        import csv
        csv_path = os.path.join(root, split_csv)
        data = []
        
        with open(csv_path, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                data.append({
                    'img_rel_path': row['img_rel_path'],
                    'pid': row['pid'],  # Keep as string (dog_id)
                    'camid': int(row['camid']) if 'camid' in row and use_camid else 0
                })
        
        # Apply dense PID mapping only if relabel=True (for training)
        if self.relabel:
            unique_pids = sorted(set(record['pid'] for record in data))
            self.pid_mapping = {pid: i for i, pid in enumerate(unique_pids)}
        else:
            self.pid_mapping = None
        
        # Build final data structures
        self.img_paths = []
        self.pids = []
        self.camids = []
        
        for record in data:
            img_path = os.path.join(root, images_dir, record['img_rel_path'])
            self.img_paths.append(img_path)
            
            # Apply PID mapping if relabeling
            if self.relabel:
                pid = self.pid_mapping[record['pid']]
            else:
                pid = record['pid']  # Keep as string for evaluation
            self.pids.append(pid)
            
            self.camids.append(record['camid'])
        
        print(f"Dataset: {len(self.img_paths)} images, {self.num_classes} identities")
    
    def __len__(self):
        return len(self.img_paths)
    
    def expand_bbox(self, bbox, expansion_ratio, img_width, img_height):
        """Expand bbox by percentage while staying within image bounds."""
        width = bbox['width']
        height = bbox['height']
        
        expand_w = int(width * expansion_ratio / 2)
        expand_h = int(height * expansion_ratio / 2)
        
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
    
    def _load_and_extract_regions(self, img_path):
        """Load image and extract regions from landmarks."""
        # Parse dog_id and photo_id from path
        # Path format: /root/images/011103/00.png
        path_parts = img_path.split('/')
        filename = path_parts[-1]  # 00.png
        dog_id = path_parts[-2]    # 011103
        photo_id = filename.split('.')[0]  # 00
        
        # Load image
        img = Image.open(img_path).convert('RGB')
        img_width, img_height = img.size
        
        # Load landmarks
        landmark_path = self.landmarks_dir / f"{dog_id}_{photo_id}.json"
        if not landmark_path.exists():
            raise FileNotFoundError(f"Landmark JSON not found: {landmark_path}")
        
        with open(landmark_path, 'r') as f:
            landmarks = json.load(f)
        
        # Extract regions with expansion
        region_bboxes = landmarks.get('region_bboxes', {})
        if not region_bboxes:
            raise ValueError(f"No region_bboxes in {landmark_path}")
        
        regions = {}
        for region_name in self.regions:
            if region_name not in region_bboxes:
                raise ValueError(f"Missing region {region_name} in {landmark_path}")
            
            orig_bbox = region_bboxes[region_name]
            expansion = self.expansion_ratios.get(region_name, 0.2)
            expanded_bbox = self.expand_bbox(orig_bbox, expansion, img_width, img_height)
            
            # Crop region
            region_img = img.crop((
                expanded_bbox['x_min'], expanded_bbox['y_min'],
                expanded_bbox['x_max'], expanded_bbox['y_max']
            ))
            regions[region_name] = region_img
        
        return img, regions
    
    def __getitem__(self, idx):
        """
        Returns:
            img: Full face image (transformed)
            regions: Dict of regional images (transformed)
            pid: Dog ID (int for train, str for eval)
            camid: Camera ID (int)
            path: Image file path (str)
        """
        path = self.img_paths[idx]
        pid = self.pids[idx]
        camid = self.camids[idx]
        
        # Load image and extract regions
        img, regions = self._load_and_extract_regions(path)
        
        # Apply transforms
        if self.transform_global:
            img = self.transform_global(img)
        
        if self.transform_regional:
            regions = {name: self.transform_regional(reg_img) 
                      for name, reg_img in regions.items()}
        
        return img, regions, pid, camid, path
    
    @property
    def num_classes(self):
        """Number of unique identities in this split"""
        return len(set(self.pids))
    
    @property 
    def num_identities(self):
        """Alias for num_classes"""
        return self.num_classes
    
    def get_pid_mapping(self):
        """Get mapping from original PID to dense PID"""
        if self.relabel:
            return self.pid_mapping
        else:
            return None

