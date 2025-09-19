"""
Optimizer setup for Dog ReID training.
Based on CLIP-ReID's approach with backbone freezing capability.
"""

import torch

def make_optimizer(cfg, model, center_criterion=None, freeze_backbone=False):
    """
    Create optimizer for Dog ReID training.
    
    Args:
        cfg: Config object with optimizer settings
        model: Model to optimize
        center_criterion: Optional center loss criterion
        freeze_backbone: Whether to freeze backbone parameters
    Returns:
        optimizer: Main optimizer
        optimizer_center: Center loss optimizer (if center_criterion provided)
    """
    params = []
    
    for key, value in model.named_parameters():
        if not value.requires_grad:
            continue
            
        # Skip backbone parameters if frozen
        if freeze_backbone and 'backbone' in key:
            print(f"Freezing backbone parameter: {key}")
            value.requires_grad = False
            continue
            
        lr = cfg.BASE_LR
        weight_decay = cfg.WEIGHT_DECAY
        
        # Different learning rates for different components
        if freeze_backbone:
            # When backbone is frozen, use normal LR for all other components
            lr = cfg.BASE_LR
        else:
            # When fine-tuning, use smaller LR for backbone
            if 'backbone' in key:
                lr = cfg.BASE_LR * 0.1  # 10x smaller for backbone
                print(f"Using reduced LR for backbone parameter: {key}")
        
        # Special handling for bias terms
        if "bias" in key:
            lr = lr * cfg.BIAS_LR_FACTOR
            weight_decay = cfg.WEIGHT_DECAY_BIAS
            
        # Optional: larger LR for classifier
        if cfg.LARGE_FC_LR and ("classifier" in key):
            lr = lr * 2
            print(f"Using 2x learning rate for classifier: {key}")

        params.append({"params": [value], "lr": lr, "weight_decay": weight_decay})

    # Create main optimizer
    if cfg.OPTIMIZER_NAME == 'SGD':
        optimizer = torch.optim.SGD(params, momentum=cfg.MOMENTUM)
    elif cfg.OPTIMIZER_NAME == 'AdamW':
        optimizer = torch.optim.AdamW(params, eps=1e-8)
    elif cfg.OPTIMIZER_NAME == 'Adam':
        optimizer = torch.optim.Adam(params, eps=1e-8)
    else:
        optimizer = getattr(torch.optim, cfg.OPTIMIZER_NAME)(params)
    
    # Center loss optimizer (if provided)
    optimizer_center = None
    if center_criterion is not None:
        optimizer_center = torch.optim.SGD(
            center_criterion.parameters(), 
            lr=cfg.CENTER_LR
        )
    
    print(f"✅ Optimizer created: {cfg.OPTIMIZER_NAME}")
    print(f"   Base LR: {cfg.BASE_LR}")
    print(f"   Weight Decay: {cfg.WEIGHT_DECAY}")
    print(f"   Backbone frozen: {freeze_backbone}")
    
    return optimizer, optimizer_center

def freeze_backbone(model):
    """Freeze backbone parameters."""
    for name, param in model.named_parameters():
        if 'backbone' in name:
            param.requires_grad = False
    print("🔒 Backbone parameters frozen")

def unfreeze_backbone(model):
    """Unfreeze backbone parameters.""" 
    for name, param in model.named_parameters():
        if 'backbone' in name:
            param.requires_grad = True
    print("🔓 Backbone parameters unfrozen")
