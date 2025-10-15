"""
Configuration for DINOv3-L Training on PetFace Dataset
Larger model for higher performance
"""

from config_petface import PetFaceConfig


class DINOv3LPetFaceConfig(PetFaceConfig):
    """DINOv3-L configuration for PetFace dataset"""
    
    # Model Architecture
    BACKBONE = 'dinov3_vitl16'   # DINOv3-L: 1024-dim features
    EMBED_DIM = 1024             # Keep native dimension (no projection)
    PRETRAINED = True
    BN_NECK = True
    
    # Training Strategy  
    FREEZE_BACKBONE = False      # Fine-tune for dog adaptation
    
    # Output
    OUTPUT_DIR = "./outputs/petface_dinov3l_baseline"
    
    # Optimization (same as DINOv3-B baseline)
    BASE_LR = 3e-4
    WEIGHT_DECAY = 0.01
    MAX_EPOCHS = 120
    
    # Batch size (smaller for larger model)
    IMS_PER_BATCH = 32           # Reduced from 64 for DINOv3-L
    NUM_INSTANCE = 4
    TEST_BATCH_SIZE = 64
    
    # Evaluation
    EVAL_PERIOD = 1              # Validate every epoch
    CHECKPOINT_PERIOD = 10
    LOG_PERIOD = 50


cfg = DINOv3LPetFaceConfig()

