"""
DataLoader factory for regional PetFace dataset.
Similar to make_dataloader_petface.py but handles regional features.
"""

import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from .dog_face_regional_dataset import DogFaceRegional
from .sampler import RandomIdentitySampler


def build_transforms(cfg, is_train=True):
    """Build image preprocessing transforms."""
    normalize = transforms.Normalize(
        mean=cfg.PIXEL_MEAN, 
        std=cfg.PIXEL_STD
    )
    
    size = cfg.IMAGE_SIZE
    
    if is_train:
        transform_list = [
            transforms.Resize(size),
            transforms.RandomRotation(getattr(cfg, 'ROTATION_DEGREE', 10)),
            transforms.ColorJitter(
                brightness=getattr(cfg, 'BRIGHTNESS', 0.2),
                contrast=getattr(cfg, 'CONTRAST', 0.2),
                saturation=getattr(cfg, 'SATURATION', 0.2),
                hue=0
            ),
            transforms.Pad(cfg.PADDING) if cfg.PADDING > 0 else None,
            transforms.RandomCrop(size) if cfg.PADDING > 0 else None,
            transforms.ToTensor(),
            normalize,
        ]
    else:
        transform_list = [
            transforms.Resize(size),
            transforms.ToTensor(),
            normalize,
        ]
    
    transform_list = [t for t in transform_list if t is not None]
    return transforms.Compose(transform_list)


class RegionalCollator:
    """
    Collate function for regional ReID datasets.
    Handles batching of images + regional features.
    """
    def __call__(self, batch):
        # batch is list of (img, regions, pid, camid, path)
        imgs, regions_list, pids, camids, paths = zip(*batch)
        
        # Stack global images
        imgs_tensor = torch.stack(imgs, 0)
        
        # Stack regional images
        # regions_list is list of dicts, need to convert to dict of tensors
        region_names = regions_list[0].keys()
        regions_batch = {}
        for region_name in region_names:
            region_tensors = [regions[region_name] for regions in regions_list]
            regions_batch[region_name] = torch.stack(region_tensors, 0)
        
        # PIDs and camids
        pids_tensor = torch.tensor(pids, dtype=torch.long)
        camids_tensor = torch.tensor(camids, dtype=torch.long)
        
        return (
            imgs_tensor,        # [B, 3, H, W]
            regions_batch,      # Dict of region_name -> [B, 3, H, W]
            pids_tensor,        # [B]
            camids_tensor,      # [B]
            list(paths)         # List of strings
        )


def make_regional_dataloaders(cfg=None, landmarks_dir=None):
    """
    Create train, val, test dataloaders for regional PetFace dataset.
    
    Args:
        cfg: Configuration object
        landmarks_dir: Path to landmarks directory (required)
        
    Returns:
        train_loader, val_query_loader, val_gallery_loader,
        test_query_loader, test_gallery_loader, num_classes
    """
    if cfg is None:
        from config_petface import cfg
    
    if landmarks_dir is None:
        raise ValueError("landmarks_dir is required for regional dataset")
    
    # Build transforms
    train_transforms_global = build_transforms(cfg, is_train=True)
    test_transforms_global = build_transforms(cfg, is_train=False)
    
    # Regional transforms (smaller size)
    regional_size = getattr(cfg, 'REGIONAL_SIZE', (64, 64))
    train_transforms_regional = transforms.Compose([
        transforms.Resize(regional_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=cfg.PIXEL_MEAN, std=cfg.PIXEL_STD)
    ])
    test_transforms_regional = transforms.Compose([
        transforms.Resize(regional_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=cfg.PIXEL_MEAN, std=cfg.PIXEL_STD)
    ])
    
    root = cfg.ROOT_DIR
    
    # Create training dataset
    train_set = DogFaceRegional(
        root=root,
        split_csv=cfg.TRAIN_SPLIT,
        landmarks_dir=landmarks_dir,
        images_dir=cfg.IMAGES_DIR,
        transform_global=train_transforms_global,
        transform_regional=train_transforms_regional,
        use_camid=cfg.USE_CAMID,
        relabel=True
    )
    
    # Create validation query/gallery
    val_query_set = DogFaceRegional(
        root=root,
        split_csv=cfg.VAL_QUERY_SPLIT,
        landmarks_dir=landmarks_dir,
        images_dir=cfg.IMAGES_DIR,
        transform_global=test_transforms_global,
        transform_regional=test_transforms_regional,
        use_camid=cfg.USE_CAMID,
        relabel=False
    )
    
    val_gallery_set = DogFaceRegional(
        root=root,
        split_csv=cfg.VAL_GALLERY_SPLIT,
        landmarks_dir=landmarks_dir,
        images_dir=cfg.IMAGES_DIR,
        transform_global=test_transforms_global,
        transform_regional=test_transforms_regional,
        use_camid=cfg.USE_CAMID,
        relabel=False
    )
    
    # Create test query/gallery
    test_query_set = DogFaceRegional(
        root=root,
        split_csv=cfg.TEST_QUERY_SPLIT,
        landmarks_dir=landmarks_dir,
        images_dir=cfg.IMAGES_DIR,
        transform_global=test_transforms_global,
        transform_regional=test_transforms_regional,
        use_camid=cfg.USE_CAMID,
        relabel=False
    )
    
    test_gallery_set = DogFaceRegional(
        root=root,
        split_csv=cfg.TEST_GALLERY_SPLIT,
        landmarks_dir=landmarks_dir,
        images_dir=cfg.IMAGES_DIR,
        transform_global=test_transforms_global,
        transform_regional=test_transforms_regional,
        use_camid=cfg.USE_CAMID,
        relabel=False
    )
    
    # Create identity-balanced sampler for training
    print(f"Creating PK sampler for {train_set.num_classes} identities...")
    import time
    sampler_start = time.time()
    sampler = RandomIdentitySampler(
        data_source=train_set,
        batch_size=cfg.IMS_PER_BATCH,
        num_instances=cfg.NUM_INSTANCE
    )
    print(f"✅ Sampler created in {time.time() - sampler_start:.1f}s")
    
    # Create dataloaders
    print("Creating DataLoaders...")
    loader_start = time.time()
    
    collator = RegionalCollator()
    
    train_loader = DataLoader(
        train_set,
        batch_size=cfg.IMS_PER_BATCH,
        sampler=sampler,
        num_workers=cfg.NUM_WORKERS,
        pin_memory=True,
        drop_last=True,
        collate_fn=collator
    )
    
    val_query_loader = DataLoader(
        val_query_set,
        batch_size=cfg.TEST_BATCH_SIZE,
        shuffle=False,
        num_workers=cfg.NUM_WORKERS,
        pin_memory=True,
        collate_fn=collator
    )
    
    val_gallery_loader = DataLoader(
        val_gallery_set,
        batch_size=cfg.TEST_BATCH_SIZE,
        shuffle=False,
        num_workers=cfg.NUM_WORKERS,
        pin_memory=True,
        collate_fn=collator
    )
    
    test_query_loader = DataLoader(
        test_query_set,
        batch_size=cfg.TEST_BATCH_SIZE,
        shuffle=False,
        num_workers=cfg.NUM_WORKERS,
        pin_memory=True,
        collate_fn=collator
    )
    
    test_gallery_loader = DataLoader(
        test_gallery_set,
        batch_size=cfg.TEST_BATCH_SIZE,
        shuffle=False,
        num_workers=cfg.NUM_WORKERS,
        pin_memory=True,
        collate_fn=collator
    )
    
    print(f"✅ All DataLoaders created in {time.time() - loader_start:.1f}s")
    
    return (
        train_loader, 
        val_query_loader, 
        val_gallery_loader,
        test_query_loader,
        test_gallery_loader,
        train_set.num_classes
    )

