"""
Configuration for DINOv3-L Baseline Training
Large model for establishing strong baseline performance
"""

from config_training import TrainingConfig

class DINOv3LConfig(TrainingConfig):
    """DINOv3-L configuration for baseline"""
    
    # Model Architecture
    BACKBONE = 'dinov3_vitl16'   # 🎯 Large version - strong baseline
    EMBED_DIM = 1024             # Native DINOv3-L size
    PRETRAINED = True
    BN_NECK = True
    
    # Training Strategy  
    FREEZE_BACKBONE = False      # Fine-tune for better adaptation
    
    # Output
    OUTPUT_DIR = "./outputs/dinov3l_baseline"
    
    # Optimization
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

cfg = DINOv3LConfig()

