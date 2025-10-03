"""
SWIN-specific configuration using correct image sizes for SWIN models.
SWIN models work best with 224x224 inputs (their native resolution).
"""

class SwinConfig:
    """Configuration optimized for SWIN Transformer backbones."""
    
    # Dataset paths
    ROOT_DIR = "."
    IMAGES_DIR = "images"
    TRAIN_SPLIT = "splits/train.csv"
    QUERY_SPLIT = "splits/query.csv"
    GALLERY_SPLIT = "splits/gallery.csv"
    VAL_SPLIT = "splits/val.csv"
    USE_CAMID = False
    
    # Image preprocessing - SWIN native resolution
    IMAGE_SIZE = (224, 224)    # SWIN's native input size ✅
    PIXEL_MEAN = [0.485, 0.456, 0.406]
    PIXEL_STD = [0.229, 0.224, 0.225]
    PADDING = 10
    RE_PROB = 0.5
    
    # P×K Sampling
    IMS_PER_BATCH = 64
    NUM_INSTANCE = 4
    NUM_WORKERS = 4
    
    # Model architecture - SWIN research backbone 🎯
    BACKBONE = 'swin_large_patch4_window7_224'  # Great for research!
    EMBED_DIM = 768                             # Project 1536 -> 768
    PRETRAINED = True
    BN_NECK = True
    FREEZE_BACKBONE = True
    
    # Training settings
    OPTIMIZER_NAME = 'AdamW'
    BASE_LR = 3e-4
    WEIGHT_DECAY = 0.01
    WEIGHT_DECAY_BIAS = 0.0
    BIAS_LR_FACTOR = 1.0
    LARGE_FC_LR = False
    MOMENTUM = 0.9
    
    # Learning rate schedule
    MAX_EPOCHS = 30
    STEPS = [15, 25]
    GAMMA = 0.1
    WARMUP_METHOD = 'linear'
    WARMUP_ITERS = 5
    WARMUP_FACTOR = 0.1
    
    # Loss function weights
    ID_LOSS_WEIGHT = 1.0
    TRIPLET_LOSS_WEIGHT = 1.0
    TRIPLET_MARGIN = 0.3
    LABEL_SMOOTHING = 0.1
    
    # Center loss
    USE_CENTER_LOSS = False
    CENTER_LOSS_WEIGHT = 0.0005
    CENTER_LR = 0.5
    
    # Training schedule
    LOG_PERIOD = 5
    EVAL_PERIOD = 5
    CHECKPOINT_PERIOD = 10
    
    # Evaluation settings
    FEAT_NORM = True
    NECK_FEAT = 'after'
    
    # Output
    OUTPUT_DIR = "./outputs/swin_research"

# Create config instance
cfg = SwinConfig()

def print_swin_config():
    """Print SWIN configuration summary."""
    print("🌪️  SWIN Transformer Research Configuration")
    print("=" * 50)
    print(f"🎯 Backbone: {cfg.BACKBONE}")
    print(f"🖼️  Image size: {cfg.IMAGE_SIZE} (SWIN native resolution)")
    print(f"📊 Features: 1536D -> {cfg.EMBED_DIM}D (embedding)")
    print(f"🎯 Expected: ~80-90% mAP (research space!)")
    print(f"🏋️  Training: {cfg.MAX_EPOCHS} epochs")
    print(f"📁 Output: {cfg.OUTPUT_DIR}")
    print("=" * 50)
    print()
    print("🌪️  SWIN Advantages:")
    print("   ✅ Hierarchical attention (different from ViTs)")
    print("   ✅ Window-based self-attention")  
    print("   ✅ Strong performance but not 100%")
    print("   ✅ Great architecture for novel techniques")

if __name__ == '__main__':
    print_swin_config()
