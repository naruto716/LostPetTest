"""
Configuration for DINOv3-B Research Training
Perfect for research with room for improvement (~85-95% expected)
"""

from config_training import TrainingConfig

class DINOv3BConfig(TrainingConfig):
    """DINOv3-B configuration for research with improvement potential"""
    
    # Model Architecture
    BACKBONE = 'dinov3_vitb16'   # 🎯 Base version - room for improvement
    EMBED_DIM = 768              # Keep features at 768D (native DINOv3-B size)
    PRETRAINED = True
    BN_NECK = True
    
    # Training Strategy  
    FREEZE_BACKBONE = False      # Fine-tune for better adaptation
    
    # Output
    OUTPUT_DIR = "./outputs/dinov3b_research"
    
    # Optimization (can be more aggressive since backbone is smaller)
    BASE_LR = 3e-4               # Standard learning rate
    WEIGHT_DECAY = 0.01
    MAX_EPOCHS = 60              # Standard training length
    
    # Loss weights (balanced approach)
    ID_LOSS_WEIGHT = 1.0
    TRIPLET_LOSS_WEIGHT = 1.0
    TRIPLET_MARGIN = 0.3
    LABEL_SMOOTHING = 0.1
    
    # Evaluation
    EVAL_PERIOD = 10
    CHECKPOINT_PERIOD = 10
    LOG_PERIOD = 10

cfg = DINOv3BConfig()
