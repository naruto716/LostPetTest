import os
import csv
from torch.utils.data import Dataset
from PIL import Image

class DogMultiPose(Dataset):
    """
    ReID-style dataset for Dog ReID.
    CSV columns: img_rel_path,pid,camid
    
    Args:
        root: Root directory containing images and splits
        split_csv: Path to CSV file relative to root (e.g., 'splits/train.csv')
        images_dir: Directory name containing images (default: 'images')
        transform: Optional transforms to apply to images
        use_camid: Whether to use camera IDs (default: False for dogs)
        relabel: Whether to apply dense PID mapping (True for train, False for test)
    """
    
    def __init__(self, root, split_csv, images_dir="images", transform=None, use_camid=False, relabel=True):
        self.root = root
        self.images_dir = images_dir
        self.transform = transform
        self.use_camid = use_camid
        self.relabel = relabel  # Whether to apply dense PID mapping

        # Load CSV data
        csv_path = os.path.join(root, split_csv)
        data = []
        
        with open(csv_path, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                data.append({
                    'img_rel_path': row['img_rel_path'],
                    'pid': int(row['pid']),
                    'camid': int(row['camid']) if 'camid' in row and use_camid else 0
                })
        
        # Apply dense PID mapping only if relabel=True (for training)
        if self.relabel:
            # Create dense PID mapping: original PIDs -> 0,1,2,...
            unique_pids = sorted(set(record['pid'] for record in data))
            self.pid_mapping = {pid: i for i, pid in enumerate(unique_pids)}
        else:
            # Keep original PIDs for evaluation (query/gallery consistency)
            self.pid_mapping = None
        
        # Build final data structures
        self.img_paths = []
        self.pids = []
        self.camids = []
        
        for record in data:
            # Build image path
            img_path = os.path.join(root, images_dir, record['img_rel_path'])
            self.img_paths.append(img_path)
            
            # Apply PID mapping if relabeling, otherwise keep original
            if self.relabel:
                pid = self.pid_mapping[record['pid']]  # Dense: 0,1,2,...
            else:
                pid = record['pid']  # Original: 100,200,300,...
            self.pids.append(pid)
            
            # Camera ID
            self.camids.append(record['camid'])
        
        print(f"Dataset: {len(self.img_paths)} images, {self.num_classes} identities")

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        """
        Returns:
            img: PIL Image or transformed tensor
            pid: Person/Dog ID (int)
            camid: Camera ID (int) 
            path: Image file path (str)
        """
        path = self.img_paths[idx]
        pid = self.pids[idx]
        camid = self.camids[idx]
        
        # Load and convert image
        img = Image.open(path).convert("RGB")
        
        # Apply transforms if provided
        if self.transform:
            img = self.transform(img)
            
        return img, pid, camid, path

    @property
    def num_classes(self):
        """Number of unique identities in this split"""
        return len(set(self.pids))
    
    @property 
    def num_identities(self):
        """Alias for num_classes"""
        return self.num_classes
        
    def get_pid_mapping(self):
        """Get mapping from original PID to dense PID (only available if relabel=True)"""
        if self.relabel:
            return self.pid_mapping
        else:
            return None  # No mapping when using original PIDs
