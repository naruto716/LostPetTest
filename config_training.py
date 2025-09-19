"""
Training configuration for Dog ReID.
Based on CLIP-ReID's config structure but simplified for our use case.
"""

class TrainingConfig:
    # Dataset paths
    ROOT_DIR = "."
    IMAGES_DIR = "images"
    TRAIN_SPLIT = "splits/train.csv"
    QUERY_SPLIT = "splits/query.csv"
    GALLERY_SPLIT = "splits/gallery.csv"
    VAL_SPLIT = "splits/val.csv"  # Combined query + gallery for evaluation
    USE_CAMID = False
    
    # Image preprocessing
    IMAGE_SIZE = (336, 336)    # Larger images for better features
    PIXEL_MEAN = [0.485, 0.456, 0.406]
    PIXEL_STD = [0.229, 0.224, 0.225]
    PADDING = 15
    RE_PROB = 0.5  # Random erasing probability
    
    # P×K Sampling (scaled for server)
    IMS_PER_BATCH = 128  # Total batch size (P×K)
    NUM_INSTANCE = 4     # K - instances per identity  
    NUM_WORKERS = 8      # Multiprocessing workers
    
    # Model architecture  
    BACKBONE = 'dinov3_vitl16'  # Proven winner from comparison
    EMBED_DIM = 768             # Feature embedding dimension
    PRETRAINED = True
    BN_NECK = True
    FREEZE_BACKBONE = True      # Start with frozen backbone approach
    
    # Optimizer settings (following CLIP-ReID)
    OPTIMIZER_NAME = 'AdamW'    # Best for transformers
    BASE_LR = 3e-4              # Base learning rate
    WEIGHT_DECAY = 0.01         # L2 regularization
    WEIGHT_DECAY_BIAS = 0.0     # No weight decay for bias
    BIAS_LR_FACTOR = 1.0        # Same LR for bias
    LARGE_FC_LR = False         # No special classifier LR
    MOMENTUM = 0.9              # For SGD (if used)
    
    # Learning rate schedule (WarmupMultiStepLR)
    MAX_EPOCHS = 60             # Total training epochs
    STEPS = [30, 50]            # LR decay milestones  
    GAMMA = 0.1                 # LR decay factor
    WARMUP_METHOD = 'linear'    # Linear warmup
    WARMUP_ITERS = 10          # Warmup iterations
    WARMUP_FACTOR = 0.1         # Initial warmup factor
    
    # Loss function weights
    ID_LOSS_WEIGHT = 1.0        # Identity classification loss
    TRIPLET_LOSS_WEIGHT = 1.0   # Triplet loss
    TRIPLET_MARGIN = 0.3        # Triplet margin
    LABEL_SMOOTHING = 0.1       # Label smoothing factor
    
    # Center loss (optional)
    USE_CENTER_LOSS = False     # Disable for small dataset
    CENTER_LOSS_WEIGHT = 0.0005
    CENTER_LR = 0.5
    
    # Training schedule
    LOG_PERIOD = 10             # Log every N iterations
    EVAL_PERIOD = 10            # Evaluate every N epochs
    CHECKPOINT_PERIOD = 20      # Save checkpoint every N epochs
    
    # Evaluation settings
    FEAT_NORM = True            # Normalize features during evaluation
    NECK_FEAT = 'after'         # Use post-BN features for evaluation
    
    # Output
    OUTPUT_DIR = "./outputs/dogreid_training"

# Create config instance
cfg = TrainingConfig()

# Print configuration summary
def print_config():
    """Print training configuration summary."""
    print("🎯 Dog ReID Training Configuration")
    print("=" * 50)
    print(f"Model: {cfg.BACKBONE} (frozen: {cfg.FREEZE_BACKBONE})")
    print(f"Batch size: {cfg.IMS_PER_BATCH} (P×K: P×{cfg.NUM_INSTANCE})")
    print(f"Image size: {cfg.IMAGE_SIZE}")
    print(f"Optimizer: {cfg.OPTIMIZER_NAME} (LR: {cfg.BASE_LR})")
    print(f"Epochs: {cfg.MAX_EPOCHS} (LR steps: {cfg.STEPS})")
    print(f"Loss weights: ID={cfg.ID_LOSS_WEIGHT}, Triplet={cfg.TRIPLET_LOSS_WEIGHT}")
    print(f"Output: {cfg.OUTPUT_DIR}")
    print("=" * 50)

if __name__ == '__main__':
    print_config()
