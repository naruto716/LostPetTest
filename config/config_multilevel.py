"""
Configuration for Multi-Level DINOv3-S Research
🧪 NOVEL APPROACH: Inspired by Amur Tiger ReID regional pooling!
Extracts features from multiple transformer layers for rich multi-scale representation.
"""

from config_training import TrainingConfig

class MultiLevelConfig(TrainingConfig):
    """Multi-level DINOv3-S configuration - Novel transformer regional pooling!"""
    
    # 🧪 RESEARCH: Multi-level backbone
    BACKBONE = 'dinov3_vits16_multilevel'  # 🎯 384 * 5 = 1920D multi-scale features!
    EMBED_DIM = 768                        # Project 1920 -> 768 for manageable size
    PRETRAINED = True
    BN_NECK = True
    
    # Training Strategy  
    FREEZE_BACKBONE = False                # Fine-tune for adaptation
    
    # Output
    OUTPUT_DIR = "./outputs/multilevel_research"
    
    # Optimization (careful with high-dimensional features)
    BASE_LR = 3e-4                         # Moderate LR for multi-level features
    WEIGHT_DECAY = 0.02                    # Higher weight decay for regularization
    MAX_EPOCHS = 100                       # Longer training for complex features
    
    # Loss weights (emphasize metric learning with rich features)
    ID_LOSS_WEIGHT = 1.0
    TRIPLET_LOSS_WEIGHT = 2.0              # Strong emphasis on metric learning
    TRIPLET_MARGIN = 0.5                   # Larger margin for rich features
    LABEL_SMOOTHING = 0.1
    
    # Evaluation
    EVAL_PERIOD = 10
    CHECKPOINT_PERIOD = 10
    LOG_PERIOD = 10
    
    # Batch size (might need adjustment for high-dim features)
    IMS_PER_BATCH = 96                     # Conservative for 1920D -> 768D processing

cfg = MultiLevelConfig()
