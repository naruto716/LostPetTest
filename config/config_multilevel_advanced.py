"""
Configuration for Advanced Multi-Level DINOv3 Research
🧪 ADVANCED: Different fusion architectures for experimentation
"""

from config_training import TrainingConfig

class AdvancedMultiLevelConfig(TrainingConfig):
    """Advanced multi-level configuration with different fusion strategies"""
    
    # 🧪 RESEARCH: Multi-level backbone
    BACKBONE = 'dinov3_vits16_multilevel'  # 5×384 = 1920D multi-scale features!
    EMBED_DIM = 768
    PRETRAINED = True
    BN_NECK = True
    
    # 🔬 FUSION ARCHITECTURE CHOICE
    FUSION_ARCHITECTURE = 'tiger'  # Options: 'tiger', 'bottleneck', 'deep'
    
    # Training Strategy  
    FREEZE_BACKBONE = False
    
    # Output
    OUTPUT_DIR = "./outputs/multilevel_advanced"
    
    # Optimization
    BASE_LR = 3e-4
    WEIGHT_DECAY = 0.02
    MAX_EPOCHS = 100
    
    # Loss weights
    ID_LOSS_WEIGHT = 1.0
    TRIPLET_LOSS_WEIGHT = 2.0
    TRIPLET_MARGIN = 0.5
    LABEL_SMOOTHING = 0.1
    
    # Evaluation
    EVAL_PERIOD = 10
    CHECKPOINT_PERIOD = 10
    LOG_PERIOD = 10
    
    # Batch size
    IMS_PER_BATCH = 96

cfg = AdvancedMultiLevelConfig()
