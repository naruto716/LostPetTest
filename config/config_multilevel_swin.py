"""
Configuration for Multi-Level SWIN Research
🧪 NOVEL APPROACH: TRUE hierarchical multi-scale features!
SWIN's hierarchical downsampling = perfect for Tiger ReID regional pooling adaptation
"""

from config_training import TrainingConfig

class MultiLevelSWINConfig(TrainingConfig):
    """Multi-level SWIN configuration - TRUE multi-scale like CNNs!"""
    
    # 🧪 RESEARCH: Multi-level SWIN backbone (IMPROVED!)
    BACKBONE = 'swin_tiny_patch4_window7_224_multilevel'  # 🎯 384+768 = 1152D (skip low-level noise!)
    EMBED_DIM = 768                                       # Project 1152 -> 768 for consistency
    PRETRAINED = True
    BN_NECK = True
    
    # SWIN requires 224x224 images
    IMAGE_SIZE = (224, 224)  # 🚨 Critical: SWIN architecture requirement
    
    # Training Strategy  
    FREEZE_BACKBONE = False                # Fine-tune the whole network
    
    # Output
    OUTPUT_DIR = "./outputs/multilevel_swin"
    
    # Optimization (transformers need careful tuning)
    BASE_LR = 3e-4                         # Good for SWIN fine-tuning
    WEIGHT_DECAY = 0.03                    # Higher for transformer regularization
    MAX_EPOCHS = 100                       # Longer for multi-level feature learning
    
    # Loss weights (emphasize metric learning for multi-scale)
    ID_LOSS_WEIGHT = 1.0
    TRIPLET_LOSS_WEIGHT = 2.0              # Strong emphasis on multi-scale metric learning
    TRIPLET_MARGIN = 0.3                   # Standard margin
    LABEL_SMOOTHING = 0.1
    
    # Evaluation
    EVAL_PERIOD = 10
    CHECKPOINT_PERIOD = 10
    LOG_PERIOD = 10
    
    # Batch size (conservative for multi-level)
    IMS_PER_BATCH = 96

cfg = MultiLevelSWINConfig()
