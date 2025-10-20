"""
Configuration for SWIN-Base Training on PetFace Dataset
Different architecture family for comparison with DINOv3
"""

from config_petface import PetFaceConfig


class SWINBPetFaceConfig(PetFaceConfig):
    """SWIN-Base configuration for PetFace dataset"""
    
    # Model Architecture
    BACKBONE = 'swin_base_patch4_window7_224'  # SWIN-Base: 1024-dim features
    EMBED_DIM = 768                            # Project to 768D for consistency
    PRETRAINED = True
    BN_NECK = True
    
    # SWIN requires 224x224 images (architecture constraint)
    IMAGE_SIZE = (224, 224)
    
    # Training Strategy  
    FREEZE_BACKBONE = False  # Fine-tune for dog adaptation
    
    # Output
    OUTPUT_DIR = "./outputs/petface_swinb_baseline"
    
    # Optimization (transformers need careful tuning)
    BASE_LR = 1e-4           # Lower LR for SWIN fine-tuning
    WEIGHT_DECAY = 0.05      # Higher weight decay for transformers
    MAX_EPOCHS = 120         # Match DINOv3 baseline
    
    # Loss weights
    ID_LOSS_WEIGHT = 1.0
    TRIPLET_LOSS_WEIGHT = 1.0
    TRIPLET_MARGIN = 0.3
    LABEL_SMOOTHING = 0.1
    
    # Evaluation
    EVAL_PERIOD = 1          # Validate every epoch
    CHECKPOINT_PERIOD = 10
    LOG_PERIOD = 50
    
    # Batch size (SWIN can handle decent batch sizes)
    IMS_PER_BATCH = 64
    NUM_INSTANCE = 4
    TEST_BATCH_SIZE = 128


cfg = SWINBPetFaceConfig()

