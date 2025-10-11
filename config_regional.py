"""
Configuration for Regional Dog ReID with DINOv3-L.
Frozen backbone, train fusion layers only.
"""

from config_training import TrainingConfig


class RegionalConfig(TrainingConfig):
    """Regional feature extraction config with frozen DINOv3-L"""
    
    # Model Architecture
    BACKBONE = 'dinov3_vitl16'   # DINOv3-L: 1024-dim per region
    EMBED_DIM = 768              # Final embedding after fusion
    PRETRAINED = True
    BN_NECK = True
    
    # Training Strategy
    FREEZE_BACKBONE = True       # Keep DINOv3 frozen, only train fusion
    
    # Dataset Paths - Using filtered valid splits
    ROOT_DIR = "/home/sagemaker-user/src/LostPetTest"
    IMAGES_DIR = "/home/sagemaker-user/src/Mine/dog"
    LANDMARKS_DIR = "/home/sagemaker-user/src/Mine/dog_landmarks"
    
    TRAIN_SPLIT = "splits_petface_valid/train.csv"
    VAL_QUERY_SPLIT = "splits_petface_valid/val_query.csv"
    VAL_GALLERY_SPLIT = "splits_petface_valid/val_gallery.csv"
    TEST_QUERY_SPLIT = "splits_petface_valid/test_query.csv"
    TEST_GALLERY_SPLIT = "splits_petface_valid/test_gallery.csv"
    
    # Output
    OUTPUT_DIR = "./outputs/regional_dinov3l"
    
    # Optimization
    BASE_LR = 3e-4               # Standard learning rate for fusion layer
    WEIGHT_DECAY = 0.01
    MAX_EPOCHS = 60
    
    # Batch size - adjust based on GPU memory
    # Regional model uses 8x forward passes per batch (1 global + 7 regions)
    # Dataset has ~3 images per dog, so K must be small
    IMS_PER_BATCH = 32           # 8 identities × 4 images
    NUM_INSTANCE = 4             # K = 4 images per identity (realistic for ~3 avg)
    TEST_BATCH_SIZE = 64
    
    # Loss weights
    ID_LOSS_WEIGHT = 1.0
    TRIPLET_LOSS_WEIGHT = 1.0
    TRIPLET_MARGIN = 0.3
    LABEL_SMOOTHING = 0.1
    
    # Learning rate schedule
    STEPS = [30, 50]             # Reduce LR at these epochs
    GAMMA = 0.1                  # LR reduction factor
    WARMUP_FACTOR = 0.01
    WARMUP_ITERS = 10
    WARMUP_METHOD = 'linear'
    
    # Evaluation
    EVAL_PERIOD = 5              # Evaluate every 5 epochs
    CHECKPOINT_PERIOD = 10
    LOG_PERIOD = 20
    
    # Data augmentation
    PIXEL_MEAN = [0.485, 0.456, 0.406]
    PIXEL_STD = [0.229, 0.224, 0.225]
    IMAGE_SIZE = (224, 224)
    REGIONAL_SIZE = (64, 64)     # Size for regional images
    PADDING = 10
    ROTATION_DEGREE = 10
    BRIGHTNESS = 0.2
    CONTRAST = 0.2
    SATURATION = 0.2
    
    # Other
    NUM_WORKERS = 4
    USE_CAMID = False


cfg = RegionalConfig()

