"""
Configuration for Regional Dog ReID with DINOv3-L.
Frozen backbone, train fusion layers only.
"""

from config_training import TrainingConfig


class RegionalConfig(TrainingConfig):
    """Regional feature extraction config with fine-tuned DINOv3-B (fair comparison to baseline)"""
    
    # Model Architecture
    BACKBONE = 'dinov3_vitb16'   # DINOv3-B: 768-dim per region (same as baseline)
    EMBED_DIM = 768              # Final embedding after fusion
    PRETRAINED = True
    BN_NECK = True
    
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
    OUTPUT_DIR = "./outputs/regional_dinov3b_finetuned"
    
    # Optimization
    BASE_LR = 3e-4               # Standard learning rate (same as baseline)
    WEIGHT_DECAY = 0.01
    MAX_EPOCHS = 120             # Match baseline training length
    
    # Batch size - adjust based on GPU memory
    # Regional model uses 8x forward passes per batch (1 global + 7 regions)
    # Try to match baseline (64) but may need to reduce if OOM
    IMS_PER_BATCH = 64           # 16 identities × 4 images (match baseline)
    NUM_INSTANCE = 4             # K = 4 images per identity
    TEST_BATCH_SIZE = 64
    
    # Loss weights (same as baseline)
    ID_LOSS_WEIGHT = 1.0
    TRIPLET_LOSS_WEIGHT = 1.0
    TRIPLET_MARGIN = 0.3
    LABEL_SMOOTHING = 0.1
    
    # Learning rate schedule (match baseline)
    STEPS = [40, 70]             # Reduce LR at these epochs (same as baseline)
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

