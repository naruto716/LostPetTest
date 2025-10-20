"""
Research configuration using weaker backbones for room to improve.
Perfect for developing novel ReID techniques and demonstrating improvements.
"""

class ResearchConfig:
    """
    Configuration optimized for research rather than peak performance.
    Uses backbones that are strong but not dominant, giving room for novelty.
    """
    
    # Dataset paths (same as before)
    ROOT_DIR = "."
    IMAGES_DIR = "images"
    TRAIN_SPLIT = "splits/train.csv"
    QUERY_SPLIT = "splits/query.csv"
    GALLERY_SPLIT = "splits/gallery.csv"
    VAL_SPLIT = "splits/val.csv"
    USE_CAMID = False
    
    # Image preprocessing
    IMAGE_SIZE = (224, 224)    # Standard size for most research
    PIXEL_MEAN = [0.485, 0.456, 0.406]
    PIXEL_STD = [0.229, 0.224, 0.225]
    PADDING = 10
    RE_PROB = 0.5
    
    # P×K Sampling 
    IMS_PER_BATCH = 64      # Smaller batch for research experiments
    NUM_INSTANCE = 4        # K - instances per identity  
    NUM_WORKERS = 4         # Moderate workers
    
    # Model architecture - RESEARCH FRIENDLY! 🎯
    # Choose one of these backbones for research:
    
    # Option 1: SWIN-L (different architecture paradigm, ~80-90% expected performance)
    BACKBONE = 'swin_large_patch4_window7_224'
    EMBED_DIM = 768        # Project from 1536 -> 768
    
    # Option 2: DINOv3-B (smaller than your 100% DINOv3-L, ~85-95% expected performance)  
    # BACKBONE = 'dinov3_vitb16'
    # EMBED_DIM = 512        # Project from 768 -> 512
    
    # Option 3: SWIN-B (even more room for improvement, ~75-85% expected performance)
    # BACKBONE = 'swin_base_patch4_window7_224'  
    # EMBED_DIM = 512        # Project from 1024 -> 512
    
    PRETRAINED = True
    BN_NECK = True
    FREEZE_BACKBONE = True   # Start with frozen for faster experiments
    
    # Training settings (faster for research iterations)
    OPTIMIZER_NAME = 'AdamW'
    BASE_LR = 3e-4
    WEIGHT_DECAY = 0.01
    WEIGHT_DECAY_BIAS = 0.0
    BIAS_LR_FACTOR = 1.0
    LARGE_FC_LR = False
    MOMENTUM = 0.9
    
    # Learning rate schedule
    MAX_EPOCHS = 30         # Faster experiments
    STEPS = [15, 25]        # Earlier decay
    GAMMA = 0.1
    WARMUP_METHOD = 'linear'
    WARMUP_ITERS = 5        # Quick warmup
    WARMUP_FACTOR = 0.1
    
    # Loss function weights
    ID_LOSS_WEIGHT = 1.0
    TRIPLET_LOSS_WEIGHT = 1.0
    TRIPLET_MARGIN = 0.3
    LABEL_SMOOTHING = 0.1
    
    # Center loss (can experiment with this)
    USE_CENTER_LOSS = False
    CENTER_LOSS_WEIGHT = 0.0005
    CENTER_LR = 0.5
    
    # Training schedule (faster for research)
    LOG_PERIOD = 5
    EVAL_PERIOD = 5          # Evaluate more frequently
    CHECKPOINT_PERIOD = 10
    
    # Evaluation settings
    FEAT_NORM = True
    NECK_FEAT = 'after'
    
    # Output
    OUTPUT_DIR = "./outputs/research_baseline"

# Create config instance
cfg = ResearchConfig()

def print_research_config():
    """Print research configuration summary."""
    print("🔬 Dog ReID Research Configuration")
    print("=" * 50)
    print(f"🎯 Backbone: {cfg.BACKBONE}")
    print(f"   Expected Performance: 70-90% (room for improvement!)")
    print(f"📊 Image size: {cfg.IMAGE_SIZE}")
    print(f"🏋️  Batch size: {cfg.IMS_PER_BATCH} (P×K: P×{cfg.NUM_INSTANCE})")
    print(f"⚡ Epochs: {cfg.MAX_EPOCHS} (faster research iterations)")
    print(f"🎯 Strategy: Start with baseline, then add novel techniques")
    print(f"📁 Output: {cfg.OUTPUT_DIR}")
    print("=" * 50)
    print()
    print("💡 Research Ideas to Try:")
    print("   1. Different loss functions (focal, arcface, etc.)")
    print("   2. Advanced data augmentation")
    print("   3. Attention mechanisms")
    print("   4. Multi-scale features")
    print("   5. Self-supervised pretraining")
    print("   6. Knowledge distillation from DINOv3-L")
    print()
    print("📈 Expected Progression:")
    print(f"   Baseline ({cfg.BACKBONE}): ~80%")
    print("   + Your novel technique: ~90%+")
    print("   Upper bound (DINOv3-L): ~100%")

if __name__ == '__main__':
    print_research_config()
