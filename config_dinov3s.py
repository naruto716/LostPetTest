"""
Configuration for DINOv3-Small Research Training
SMALLEST version - maximum room for improvement!
"""

from config_training import TrainingConfig

class DINOv3SmallConfig(TrainingConfig):
    """DINOv3-Small configuration - smallest DINO for research"""
    
    # Model Architecture
    BACKBONE = 'dinov3_vits16'   # 🎯 SMALLEST DINOv3 - lots of improvement room!
    EMBED_DIM = 768              # Project 384 -> 768 for standard dim
    PRETRAINED = True
    BN_NECK = True
    
    # Training Strategy  
    FREEZE_BACKBONE = False      # Fine-tune for adaptation
    
    # Output
    OUTPUT_DIR = "./outputs/dinov3s_research"
    
    # Optimization (can be more aggressive)
    BASE_LR = 5e-4               # Higher LR for smaller model
    WEIGHT_DECAY = 0.01
    MAX_EPOCHS = 80              # Longer training for smaller model
    
    # Loss weights (emphasize learning)
    ID_LOSS_WEIGHT = 1.0
    TRIPLET_LOSS_WEIGHT = 1.5    # Higher triplet weight for metric learning
    TRIPLET_MARGIN = 0.3
    LABEL_SMOOTHING = 0.1
    
    # Evaluation
    EVAL_PERIOD = 10
    CHECKPOINT_PERIOD = 10
    LOG_PERIOD = 10
    
    # Batch size (can be larger for smaller model)
    IMS_PER_BATCH = 128

cfg = DINOv3SmallConfig()
