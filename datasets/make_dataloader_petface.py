import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from .dog_multipose import DogMultiPose
from .sampler import RandomIdentitySampler

def build_transforms(cfg, is_train=True, coord_safe=False):
    """
    Build image preprocessing transforms for training or testing.
    
    Args:
        cfg: Configuration object with transform parameters
        is_train: Whether to build training (with augmentations) or test transforms
        coord_safe: If True, disable augmentations that change spatial coordinates
                    (useful for coordinate-based regional pooling)
    """
    normalize = transforms.Normalize(
        mean=cfg.PIXEL_MEAN, 
        std=cfg.PIXEL_STD
    )
    
    size = cfg.IMAGE_SIZE
    
    if is_train and not coord_safe:
        # Standard training augmentations (based on Amur Tiger ReID)
        # Note: They disabled horizontal flip and random erasing for animals
        transform_list = [
            transforms.Resize(size),
            transforms.RandomRotation(getattr(cfg, 'ROTATION_DEGREE', 10)),
            transforms.ColorJitter(
                brightness=getattr(cfg, 'BRIGHTNESS', 0.2),
                contrast=getattr(cfg, 'CONTRAST', 0.2),
                saturation=getattr(cfg, 'SATURATION', 0.2),
                hue=getattr(cfg, 'HUE', 0.2)
            ),
            transforms.Pad(cfg.PADDING) if cfg.PADDING > 0 else None,
            transforms.RandomCrop(size) if cfg.PADDING > 0 else None,
            transforms.ToTensor(),
            normalize,
        ]
    elif is_train and coord_safe:
        # Coordinate-safe training: only resize (predictable scaling)
        # Note: You'll need to scale coordinates by resize_factor
        transform_list = [
            transforms.Resize(size),
            transforms.ToTensor(),
            normalize,
        ]
    else:
        # Test transforms (always coordinate-safe)
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

def make_petface_dataloaders(cfg=None):
    """
    Create train, val query/gallery, and test query/gallery dataloaders for PetFace dataset.
    
    Args:
        cfg: Configuration object with dataset and training parameters
        
    Returns:
        train_loader: DataLoader for training with PK sampling
        val_query_loader: DataLoader for validation query set
        val_gallery_loader: DataLoader for validation gallery set
        test_query_loader: DataLoader for test query set (for final evaluation)
        test_gallery_loader: DataLoader for test gallery set (for final evaluation)
        num_classes: Number of unique identities in training set
    """
    # Use simple config if none provided
    if cfg is None:
        from config_petface import cfg
    
    # Build transforms
    coord_safe = getattr(cfg, 'COORD_SAFE_MODE', False)
    train_transforms = build_transforms(cfg, is_train=True, coord_safe=coord_safe)
    test_transforms = build_transforms(cfg, is_train=False)
    
    root = cfg.ROOT_DIR
    
    # Create training dataset
    train_set = DogMultiPose(
        root=root,
        split_csv=cfg.TRAIN_SPLIT,
        images_dir=cfg.IMAGES_DIR,
        transform=train_transforms,
        use_camid=cfg.USE_CAMID,
        relabel=True  # Dense PIDs for training
    )
    
    # Create VALIDATION query/gallery datasets
    val_query_set = DogMultiPose(
        root=root,
        split_csv=cfg.VAL_QUERY_SPLIT,
        images_dir=cfg.IMAGES_DIR,
        transform=test_transforms,
        use_camid=cfg.USE_CAMID,
        relabel=False  # Original PIDs for evaluation
    )
    
    val_gallery_set = DogMultiPose(
        root=root,
        split_csv=cfg.VAL_GALLERY_SPLIT,
        images_dir=cfg.IMAGES_DIR,
        transform=test_transforms,
        use_camid=cfg.USE_CAMID,
        relabel=False  # Original PIDs for evaluation
    )
    
    # Create TEST query/gallery datasets (for final evaluation)
    test_query_set = DogMultiPose(
        root=root,
        split_csv=cfg.TEST_QUERY_SPLIT,
        images_dir=cfg.IMAGES_DIR,
        transform=test_transforms,
        use_camid=cfg.USE_CAMID,
        relabel=False  # Original PIDs for evaluation
    )
    
    test_gallery_set = DogMultiPose(
        root=root,
        split_csv=cfg.TEST_GALLERY_SPLIT,
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

    val_query_loader = DataLoader(
        val_query_set,
        batch_size=cfg.TEST_BATCH_SIZE,
        shuffle=False,
        num_workers=cfg.NUM_WORKERS,
        pin_memory=True,
        collate_fn=ReIDCollator()
    )
    
    val_gallery_loader = DataLoader(
        val_gallery_set,
        batch_size=cfg.TEST_BATCH_SIZE,
        shuffle=False,
        num_workers=cfg.NUM_WORKERS,
        pin_memory=True,
        collate_fn=ReIDCollator()
    )
    
    test_query_loader = DataLoader(
        test_query_set,
        batch_size=cfg.TEST_BATCH_SIZE,
        shuffle=False,
        num_workers=cfg.NUM_WORKERS,
        pin_memory=True,
        collate_fn=ReIDCollator()
    )
    
    test_gallery_loader = DataLoader(
        test_gallery_set,
        batch_size=cfg.TEST_BATCH_SIZE,
        shuffle=False,
        num_workers=cfg.NUM_WORKERS,
        pin_memory=True,
        collate_fn=ReIDCollator()
    )

    return (
        train_loader, 
        val_query_loader, 
        val_gallery_loader,
        test_query_loader,
        test_gallery_loader,
        train_set.num_classes
    )

