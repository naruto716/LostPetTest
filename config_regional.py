"""
Configuration for Regional Dog ReID with DINOv3-L.
Frozen backbone, train fusion layers only.
"""

from config_training import TrainingConfig


class RegionalConfig(TrainingConfig):
    """
    Regional feature extraction config - v3 with Attention Fusion:
    - DINOv3-L backbone (more capacity for 8 regions)
    - 128x128 regional crops (vs 64x64 before)
    - Attention-based fusion (learns to ignore bad landmarks!)
    - Stronger regularization to prevent overfitting
    """
    
    # Model Architecture
    BACKBONE = 'dinov3_vitl16'   # DINOv3-L: 1024-dim per region (more capacity for 8 regions)
    EMBED_DIM = 1024             # Final embedding after fusion (match backbone)
    PRETRAINED = True
    BN_NECK = True
    USE_ATTENTION = True         # 🌟 NEW: Use attention fusion instead of simple concat
    
    # Training Strategy
    FREEZE_BACKBONE = False      # Fine-tune backbone for better performance
    
    # Dataset Paths - Using filtered valid splits
    ROOT_DIR = "/home/sagemaker-user/LostPet/LostPetTest"
    IMAGES_DIR = "/home/sagemaker-user/LostPet/PetFace/dog"
    LANDMARKS_DIR = "/home/sagemaker-user/LostPet/dogface_landmark_estimation_hrcnn/petface_landmarks_json_all"
    
    TRAIN_SPLIT = "splits_petface_valid/train.csv"
    VAL_QUERY_SPLIT = "splits_petface_valid/val_query.csv"
    VAL_GALLERY_SPLIT = "splits_petface_valid/val_gallery.csv"
    TEST_QUERY_SPLIT = "splits_petface_valid/test_query.csv"
    TEST_GALLERY_SPLIT = "splits_petface_valid/test_gallery.csv"
    
    # Output
    OUTPUT_DIR = "./outputs/regional_dinov3l_attention_v3"
    
    # Optimization
    BASE_LR = 3e-4               # Standard learning rate (same as baseline)
    WEIGHT_DECAY = 0.05          # ⬆️ Increased from 0.01 to reduce overfitting
    MAX_EPOCHS = 60              # ⬇️ Reduced - model peaks early anyway
    
    # Batch size - adjust based on GPU memory
    # Regional model uses 8x forward passes per batch (1 global + 7 regions)
    # Reduced due to larger backbone (DINOv3-L) and larger regional crops (128x128)
    IMS_PER_BATCH = 32           # 8 identities × 4 images
    NUM_INSTANCE = 4             # K = 4 images per identity
    TEST_BATCH_SIZE = 32         # Reduced for DINOv3-L
    
    # Loss weights (same as baseline)
    ID_LOSS_WEIGHT = 1.0
    TRIPLET_LOSS_WEIGHT = 1.0
    TRIPLET_MARGIN = 0.3
    LABEL_SMOOTHING = 0.1
    
    # Learning rate schedule (adjusted for shorter training)
    STEPS = [30, 50]             # Reduce LR earlier (since we train for 60 epochs)
    GAMMA = 0.1                  # LR reduction factor
    WARMUP_FACTOR = 0.01
    WARMUP_ITERS = 10
    WARMUP_METHOD = 'linear'
    
    # Dropout for regularization
    DROPOUT = 0.3                # Add dropout to fusion layer
    
    # Evaluation
    EVAL_PERIOD = 5              # Evaluate every 5 epochs
    CHECKPOINT_PERIOD = 10
    LOG_PERIOD = 20
    
    # Data augmentation
    PIXEL_MEAN = [0.485, 0.456, 0.406]
    PIXEL_STD = [0.229, 0.224, 0.225]
    IMAGE_SIZE = (224, 224)
    REGIONAL_SIZE = (128, 128)   # ⬆️ Increased from 64x64 - more details for regions
    PADDING = 10
    ROTATION_DEGREE = 10
    BRIGHTNESS = 0.2
    CONTRAST = 0.2
    SATURATION = 0.2
    
    # Other
    NUM_WORKERS = 4
    USE_CAMID = False


cfg = RegionalConfig()

