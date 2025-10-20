"""
Configuration for Multi-Level DINOv3-B Research
🧪 NOVEL APPROACH: DINOv3-Base with multi-level features!
Better than Small version - more capacity while still having improvement room.
"""

from config_training import TrainingConfig

class MultiLevelBConfig(TrainingConfig):
    """Multi-level DINOv3-B configuration - Novel transformer regional pooling!"""
    
    # 🧪 RESEARCH: Multi-level backbone
    BACKBONE = 'dinov3_vitb16_multilevel'  # 🎯 768 * 5 = 3840D multi-scale features!
    EMBED_DIM = 768                        # Project 3840 -> 768 for manageable size
    PRETRAINED = True
    BN_NECK = True
    
    # Training Strategy  
    FREEZE_BACKBONE = False                # Fine-tune for adaptation
    
    # Output
    OUTPUT_DIR = "./outputs/multilevel_dinov3b"
    
    # Optimization (careful with high-dimensional features)
    BASE_LR = 2e-4                         # Lower LR for larger model
    WEIGHT_DECAY = 0.02                    # Higher weight decay for regularization
    MAX_EPOCHS = 80                        # Longer training for complex features
    
    # Loss weights (emphasize metric learning with rich features)
    ID_LOSS_WEIGHT = 1.0
    TRIPLET_LOSS_WEIGHT = 1.5              # Strong emphasis on metric learning
    TRIPLET_MARGIN = 0.3                   # Standard margin
    LABEL_SMOOTHING = 0.1
    
    # Evaluation
    EVAL_PERIOD = 10
    CHECKPOINT_PERIOD = 10
    LOG_PERIOD = 10
    
    # Batch size (smaller due to larger model)
    IMS_PER_BATCH = 64                     # Conservative for 3840D -> 768D processing

cfg = MultiLevelBConfig()
