"""
Configuration for PetFace dataset training.
"""

class PetFaceConfig:
    # Dataset paths
    ROOT_DIR = "/home/sagemaker-user/LostPet/petface"  # Where images are stored
    IMAGES_DIR = "dog"  # Subdirectory containing dog ID folders
    
    # CSV splits (absolute paths - kept in code repo for version control)
    # NOTE: Using subset by default for quick experimentation (~2K training images)
    # To use full dataset, change splits_petface_subset -> splits_petface
    TRAIN_SPLIT = "/home/sagemaker-user/LostPet/LostPetTest/splits_petface_subset/train.csv"
    VAL_QUERY_SPLIT = "/home/sagemaker-user/LostPet/LostPetTest/splits_petface_subset/val_query.csv"
    VAL_GALLERY_SPLIT = "/home/sagemaker-user/LostPet/LostPetTest/splits_petface_subset/val_gallery.csv"
    TEST_QUERY_SPLIT = "/home/sagemaker-user/LostPet/LostPetTest/splits_petface_subset/test_query.csv"
    TEST_GALLERY_SPLIT = "/home/sagemaker-user/LostPet/LostPetTest/splits_petface_subset/test_gallery.csv"
    USE_CAMID = False  # No camera IDs in petface dataset
    
    # Image preprocessing
    IMAGE_SIZE = (256, 256)    # Square images for petface
    PIXEL_MEAN = [0.485, 0.456, 0.406]
    PIXEL_STD = [0.229, 0.224, 0.225]
    PADDING = 10
    
    # Augmentation settings (based on Amur Tiger ReID best practices)
    ROTATION_DEGREE = 10       # Random rotation ±10 degrees
    BRIGHTNESS = 0.2           # ColorJitter brightness
    CONTRAST = 0.2             # ColorJitter contrast
    SATURATION = 0.2           # ColorJitter saturation
    HUE = 0.2                  # ColorJitter hue
    
    COORD_SAFE_MODE = False    # Set to True for coordinate-based regional pooling
    
    # P×K Sampling
    IMS_PER_BATCH = 64      # Total batch size (P×K)
    NUM_INSTANCE = 4        # K - instances per identity  
    NUM_WORKERS = 0         # No multiprocessing (avoids hangs, slightly slower)
    TEST_BATCH_SIZE = 128   # Batch size for evaluation
    
    # Model architecture  
    BACKBONE = 'dinov3_vitl16'  # DINOv3-Large (proven winner from comparison)
    EMBED_DIM = 768             # Feature embedding dimension
    PRETRAINED = True
    BN_NECK = True
    FREEZE_BACKBONE = True      # Freeze backbone (following old dataset approach)
    
    # Optimizer settings (following old dataset DINOv3 config)
    OPTIMIZER_NAME = 'AdamW'    # Best for transformers
    BASE_LR = 3e-4              # Base learning rate
    WEIGHT_DECAY = 0.01         # L2 regularization
    WEIGHT_DECAY_BIAS = 0.0
    BIAS_LR_FACTOR = 1.0    # Same LR for bias (like old dataset)
    LARGE_FC_LR = False
    MOMENTUM = 0.9          # For SGD (if used)
    
    # Learning rate schedule
    MAX_EPOCHS = 120
    STEPS = [40, 70]        # LR decay milestones  
    GAMMA = 0.1             # LR decay factor
    WARMUP_METHOD = 'linear'
    WARMUP_ITERS = 10
    WARMUP_FACTOR = 0.01
    
    # Loss function weights
    ID_LOSS_WEIGHT = 1.0
    TRIPLET_LOSS_WEIGHT = 1.0
    TRIPLET_MARGIN = 0.3
    LABEL_SMOOTHING = 0.1
    
    # Center loss
    USE_CENTER_LOSS = True
    CENTER_LOSS_WEIGHT = 0.0005
    CENTER_LR = 0.5
    
    # Logging and checkpointing
    OUTPUT_DIR = "./output_petface"
    LOG_PERIOD = 10         # Log every N iterations (more frequent for subset)
    CHECKPOINT_PERIOD = 10  # Save checkpoint every N epochs
    EVAL_PERIOD = 5         # Evaluate every N epochs
    
    # Device
    DEVICE = "cuda"

# Create a singleton instance
cfg = PetFaceConfig()

