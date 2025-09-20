"""
Configuration for SWIN-L Research Training
Different architecture family - excellent for research!
"""

from config_training import TrainingConfig

class SWINLConfig(TrainingConfig):
    """SWIN-L configuration for research with different architecture"""
    
    # Model Architecture
    BACKBONE = 'swin_large_patch4_window7_224'  # 🎯 SWIN-L transformer
    EMBED_DIM = 768                             # Project 1536 -> 768 for consistency
    PRETRAINED = True
    BN_NECK = True
    
    # SWIN requires 224x224 images (hardcoded in architecture)
    IMAGE_SIZE = (224, 224)  # 🚨 Critical: SWIN is tied to this resolution
    
    # Training Strategy  
    FREEZE_BACKBONE = False  # 🔥 Fine-tune transformer for adaptation
    
    # Output
    OUTPUT_DIR = "./outputs/swinl_research"
    
    # Optimization (transformers need careful tuning)
    BASE_LR = 1e-4           # Lower LR for transformer fine-tuning
    WEIGHT_DECAY = 0.05      # Higher weight decay for transformers
    MAX_EPOCHS = 80          # Longer training for transformer adaptation
    
    # Loss weights (can be more aggressive with triplet for metric learning)
    ID_LOSS_WEIGHT = 1.0
    TRIPLET_LOSS_WEIGHT = 1.5    # Emphasize metric learning
    TRIPLET_MARGIN = 0.3
    LABEL_SMOOTHING = 0.1
    
    # Evaluation (more frequent for research)
    EVAL_PERIOD = 10
    CHECKPOINT_PERIOD = 10
    LOG_PERIOD = 10
    
    # Batch size (might need smaller due to 224x224 and fine-tuning)
    IMS_PER_BATCH = 96       # Slightly smaller batch for stability

cfg = SWINLConfig()
