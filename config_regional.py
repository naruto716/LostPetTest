"""
Configuration for Regional Dog ReID with DINOv3-L.
Frozen backbone, train fusion layers only.
"""

from config_training import TrainingConfig


class RegionalConfig(TrainingConfig):
    """
    Regional feature extraction config - v3 with Attention Fusion (A100):
    - DINOv3-L backbone (1024-dim, more capacity than baseline)
    - 128x128 regional crops (vs 64x64 before - more details)
    - Attention-based fusion (learns to ignore bad landmarks!)
    - Optimized for A100 80GB: larger batch size, stronger model
    - Stronger regularization to prevent overfitting
    """
    
    # Model Architecture
    BACKBONE = 'dinov3_vitl16'   # DINOv3-L: 1024-dim per region (for A100)
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
    OUTPUT_DIR = "./outputs/regional_dinov3l_attention_a100"
    
    # Optimization
    BASE_LR = 3e-4               # Standard learning rate (same as baseline)
    WEIGHT_DECAY = 0.05          # ⬆️ Increased from 0.01 to reduce overfitting
    MAX_EPOCHS = 60              # ⬇️ Reduced - model peaks early anyway
    
    # Batch size - A100 80GB can handle much larger batches
    # Regional model uses 8x forward passes per batch (1 global + 7 regions)
    # DINOv3-L with 128x128 regional crops on A100
    IMS_PER_BATCH = 64           # 16 identities × 4 images (A100 can handle this!)
    NUM_INSTANCE = 4             # K = 4 images per identity
    TEST_BATCH_SIZE = 64         # A100 has plenty of memory
    
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
    EVAL_PERIOD = 1              # Evaluate EVERY epoch (for plotting convergence curves)
    CHECKPOINT_PERIOD = 5       # Save checkpoint every 10 epochs
    LOG_PERIOD = 20              # Log training every 20 iterations
    
    # Data augmentation
    PIXEL_MEAN = [0.485, 0.456, 0.406]
    PIXEL_STD = [0.229, 0.224, 0.225]
    IMAGE_SIZE = (224, 224)
    REGIONAL_SIZE = (128, 128)   # ⬆️ Increased from 64x64 - more details for regions
    PADDING = 10
    ROTATION_DEGREE = 15     # Increased to prevent overfitting
    BRIGHTNESS = 0.3         # Increased
    CONTRAST = 0.3           # Increased
    SATURATION = 0.3         # Increased
    
    # Other
    NUM_WORKERS = 4
    USE_CAMID = False


cfg = RegionalConfig()

