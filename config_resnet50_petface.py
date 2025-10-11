"""
Configuration for ResNet50 Training on PetFace Dataset
Classic CNN architecture for comparison with transformers
"""

from config_petface import PetFaceConfig


class ResNet50PetFaceConfig(PetFaceConfig):
    """ResNet50 configuration for PetFace dataset"""
    
    # Model Architecture
    BACKBONE = 'resnet50'        # Classic CNN: 2048-dim features
    EMBED_DIM = 768              # Project to 768D for consistency
    PRETRAINED = True
    BN_NECK = True
    
    # Image size (ResNet is flexible, but 256x256 is standard)
    IMAGE_SIZE = (256, 256)
    
    # Training Strategy  
    FREEZE_BACKBONE = False  # Fine-tune for dog adaptation
    
    # Output
    OUTPUT_DIR = "./outputs/petface_resnet50_baseline"
    
    # Optimization (CNNs can handle higher learning rates than transformers)
    BASE_LR = 3e-4           # Standard LR (same as DINOv3)
    WEIGHT_DECAY = 0.01      # Standard weight decay for CNNs
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
    
    # Batch size (CNNs can handle large batches)
    IMS_PER_BATCH = 128      # Larger batch for CNN
    NUM_INSTANCE = 4
    TEST_BATCH_SIZE = 256    # CNNs are memory efficient


cfg = ResNet50PetFaceConfig()

