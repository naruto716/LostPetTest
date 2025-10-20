"""
Configuration for SWIN-Tiny Research Training  
SMALLEST SWIN version - excellent for research!
"""

from config_training import TrainingConfig

class SWINTinyConfig(TrainingConfig):
    """SWIN-Tiny configuration - smallest SWIN for research"""
    
    # Model Architecture
    BACKBONE = 'swin_tiny_patch4_window7_224'  # 🎯 SMALLEST SWIN - perfect for research!
    EMBED_DIM = 768                            # Keep 768D (no projection needed)
    PRETRAINED = True
    BN_NECK = True
    
    # SWIN requires 224x224 images (hardcoded in architecture)
    IMAGE_SIZE = (224, 224)  # 🚨 Critical: SWIN is tied to this resolution
    
    # Training Strategy  
    FREEZE_BACKBONE = False  # 🔥 Fine-tune transformer for adaptation
    
    # Output
    OUTPUT_DIR = "./outputs/swint_research"
    
    # Optimization (careful tuning for tiny transformer)
    BASE_LR = 2e-4           # Moderate LR for tiny transformer
    WEIGHT_DECAY = 0.05      # Higher weight decay for transformers
    MAX_EPOCHS = 100         # Longer training for tiny model
    
    # Loss weights (emphasize metric learning for tiny model)
    ID_LOSS_WEIGHT = 1.0
    TRIPLET_LOSS_WEIGHT = 2.0    # High triplet weight - metric learning focus
    TRIPLET_MARGIN = 0.3
    LABEL_SMOOTHING = 0.1
    
    # Evaluation (more frequent for tiny model research)
    EVAL_PERIOD = 10
    CHECKPOINT_PERIOD = 10
    LOG_PERIOD = 10
    
    # Batch size (can be larger for tiny model)
    IMS_PER_BATCH = 128

cfg = SWINTinyConfig()
