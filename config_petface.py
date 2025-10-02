"""
Configuration for PetFace dataset training.
"""

class PetFaceConfig:
    # Dataset paths
    ROOT_DIR = "/home/sagemaker-user/LostPet/petface"
    IMAGES_DIR = "dog"  # Subdirectory containing dog ID folders
    TRAIN_SPLIT = "splits_petface/train.csv"
    VAL_QUERY_SPLIT = "splits_petface/val_query.csv"
    VAL_GALLERY_SPLIT = "splits_petface/val_gallery.csv"
    TEST_QUERY_SPLIT = "splits_petface/test_query.csv"
    TEST_GALLERY_SPLIT = "splits_petface/test_gallery.csv"
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
    NUM_WORKERS = 8         # Multiprocessing workers
    TEST_BATCH_SIZE = 128   # Batch size for evaluation
    
    # Model architecture  
    BACKBONE = 'resnet50'   # Can be changed to any supported backbone
    EMBED_DIM = 2048        # Feature embedding dimension
    PRETRAINED = True
    BN_NECK = True
    FREEZE_BACKBONE = False
    
    # Optimizer settings
    OPTIMIZER_NAME = 'Adam'
    BASE_LR = 3.5e-4        # Base learning rate
    WEIGHT_DECAY = 5e-4     # L2 regularization
    WEIGHT_DECAY_BIAS = 0.0
    BIAS_LR_FACTOR = 2.0    # Bias learning rate multiplier
    LARGE_FC_LR = False
    MOMENTUM = 0.9          # For SGD
    
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
    LOG_PERIOD = 50         # Log every N iterations
    CHECKPOINT_PERIOD = 10  # Save checkpoint every N epochs
    EVAL_PERIOD = 5         # Evaluate every N epochs
    
    # Device
    DEVICE = "cuda"

# Create a singleton instance
cfg = PetFaceConfig()

