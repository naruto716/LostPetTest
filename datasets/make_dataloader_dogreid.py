import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from .dog_multipose import DogMultiPose
from .sampler import RandomIdentitySampler

class RandomErasing(object):
    """
    Random Erasing data augmentation.
    Randomly erases a rectangular region in the image to improve generalization.
    """
    def __init__(self, probability=0.5, sl=0.02, sh=0.4, r1=0.3, mean=(0.485, 0.456, 0.406)):
        self.probability = probability
        self.mean = mean
        self.sl = sl
        self.sh = sh
        self.r1 = r1
       
    def __call__(self, img):
        if torch.rand(1) < self.probability:
            return img
            
        for attempt in range(100):
            area = img.size()[1] * img.size()[2]
       
            target_area = torch.empty(1).uniform_(self.sl, self.sh).item() * area
            aspect_ratio = torch.empty(1).uniform_(self.r1, 1/self.r1).item()
       
            h = int(round((target_area * aspect_ratio) ** 0.5))
            w = int(round((target_area / aspect_ratio) ** 0.5))
       
            if w < img.size()[2] and h < img.size()[1]:
                x1 = torch.randint(0, img.size()[1] - h + 1, (1,)).item()
                y1 = torch.randint(0, img.size()[2] - w + 1, (1,)).item()
                if img.size()[0] == 3:
                    img[0, x1:x1+h, y1:y1+w] = self.mean[0]
                    img[1, x1:x1+h, y1:y1+w] = self.mean[1]
                    img[2, x1:x1+h, y1:y1+w] = self.mean[2]
                else:
                    img[0, x1:x1+h, y1:y1+w] = self.mean[0]
                return img
                
        return img

def build_transforms(cfg, is_train=True):
    """
    Build image preprocessing transforms for training or testing.
    
    Args:
        cfg: Configuration object with transform parameters
        is_train: Whether to build training (with augmentations) or test transforms
    """
    normalize = transforms.Normalize(
        mean=cfg.PIXEL_MEAN, 
        std=cfg.PIXEL_STD
    )
    
    size = cfg.IMAGE_SIZE
    
    if is_train:
        transform_list = [
            transforms.Resize(size),
            transforms.Pad(cfg.PADDING) if cfg.PADDING > 0 else None,
            transforms.RandomCrop(size) if cfg.PADDING > 0 else None,
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
            normalize,
        ]
        
        # Add Random Erasing if enabled
        if cfg.RE_PROB > 0:
            transform_list.append(RandomErasing(cfg.RE_PROB))
            
    else:
        transform_list = [
            transforms.Resize(size),
            transforms.ToTensor(),
            normalize,
        ]
    
    # Remove None values
    transform_list = [t for t in transform_list if t is not None]
    
    return transforms.Compose(transform_list)

class ReIDCollator:
    """
    Collate function for ReID datasets.
    Groups batch data into tensors and lists as needed.
    """
    def __call__(self, batch):
        imgs, pids, camids, paths = zip(*batch)
        
        return (
            torch.stack(imgs, 0),                    # Images as tensor
            torch.tensor(pids, dtype=torch.long),    # Person IDs as tensor
            torch.tensor(camids, dtype=torch.long),  # Camera IDs as tensor  
            list(paths)                              # Paths as list of strings
        )

def make_dataloaders(cfg=None):
    """Create train, query, and gallery dataloaders for Dog ReID"""
    # Use simple config if none provided
    if cfg is None:
        from config import cfg
    """
    Create train, query, and gallery dataloaders for Dog ReID.
    
    Args:
        cfg: Configuration object with dataset and training parameters
        
    Returns:
        train_loader: DataLoader for training
        query_loader: DataLoader for query set
        gallery_loader: DataLoader for gallery set  
        num_classes: Number of unique identities in training set
    """
    # Build transforms
    train_transforms = build_transforms(cfg, is_train=True)
    test_transforms = build_transforms(cfg, is_train=False)
    
    root = cfg.ROOT_DIR
    
    # Create datasets
    train_set = DogMultiPose(
        root=root,
        split_csv=cfg.TRAIN_SPLIT,
        images_dir=cfg.IMAGES_DIR,
        transform=train_transforms,
        use_camid=cfg.USE_CAMID,
        relabel=True  # Dense PIDs for training
    )
    
    query_set = DogMultiPose(
        root=root,
        split_csv=cfg.QUERY_SPLIT,
        images_dir=cfg.IMAGES_DIR,
        transform=test_transforms,
        use_camid=cfg.USE_CAMID,
        relabel=False  # Original PIDs for evaluation
    )
    
    gallery_set = DogMultiPose(
        root=root,
        split_csv=cfg.GALLERY_SPLIT,
        images_dir=cfg.IMAGES_DIR,
        transform=test_transforms,
        use_camid=cfg.USE_CAMID,
        relabel=False  # Original PIDs for evaluation
    )

    # Create identity-balanced sampler for training
    sampler = RandomIdentitySampler(
        data_source=train_set,
        batch_size=cfg.IMS_PER_BATCH,
        num_instances=cfg.NUM_INSTANCE
    )

    # Create dataloaders
    train_loader = DataLoader(
        train_set,
        batch_size=cfg.IMS_PER_BATCH,
        sampler=sampler,
        num_workers=cfg.NUM_WORKERS,
        pin_memory=True,
        drop_last=True,
        collate_fn=ReIDCollator()
    )

    query_loader = DataLoader(
        query_set,
        batch_size=128,  # Fixed batch size for evaluation
        shuffle=False,
        num_workers=cfg.NUM_WORKERS,
        pin_memory=True,
        collate_fn=ReIDCollator()
    )
    
    gallery_loader = DataLoader(
        gallery_set,
        batch_size=128,  # Fixed batch size for evaluation
        shuffle=False,
        num_workers=cfg.NUM_WORKERS,
        pin_memory=True,
        collate_fn=ReIDCollator()
    )

    return train_loader, query_loader, gallery_loader, train_set.num_classes
