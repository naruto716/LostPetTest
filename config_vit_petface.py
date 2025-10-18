"""
Configuration for ViT Training on PetFace Dataset
Similar setup as SWIN-B baseline, but using ViT
"""

from config_petface import PetFaceConfig

class ViTPetFaceConfig(PetFaceConfig):
    """ViT Tiny configuration for PetFace dataset"""

    # ---------------- Model Architecture ----------------
    BACKBONE = 'vit_tiny_patch16_224'  # TIMM ViT Tiny
    EMBED_DIM = 192                     # ViT Tiny CLS token dim
    PRETRAINED = True                   # Use pretrained weights
    BN_NECK = True                      # BatchNorm neck

    # Input
    IMAGE_SIZE = (224, 224)             # ViT patch size requirement

    # ---------------- Training Strategy ----------------
    FREEZE_BACKBONE = False             # Fine-tune backbone
    BASE_LR = 1e-4                      # Learning rate
    WEIGHT_DECAY = 0.05                 # Weight decay
    MAX_EPOCHS = 120

    # Loss weights
    ID_LOSS_WEIGHT = 1.0
    TRIPLET_LOSS_WEIGHT = 1.0
    TRIPLET_MARGIN = 0.3
    LABEL_SMOOTHING = 0.1

    # ---------------- Batch Settings ----------------
    IMS_PER_BATCH = 64                  # Batch size per GPU
    NUM_INSTANCE = 4                    # Instances per identity
    TEST_BATCH_SIZE = 128

    # ---------------- Output & Logging ----------------
    OUTPUT_DIR = "./outputs/petface_vit_baseline"
    EVAL_PERIOD = 1
    CHECKPOINT_PERIOD = 10
    LOG_PERIOD = 50

    # ---------------- Local Dataset Paths ----------------
    ROOT_DIR = r""
    IMAGES_DIR = r"../dog"
    TRAIN_SPLIT = r"splits_petface_valid/train.csv"
    VAL_QUERY_SPLIT = r"splits_petface_valid/val_query.csv"
    VAL_GALLERY_SPLIT = r"splits_petface_valid/val_gallery.csv"
    TEST_QUERY_SPLIT = r"splits_petface_valid/test_query.csv"
    TEST_GALLERY_SPLIT = r"splits_petface_valid/test_gallery.csv"

cfg = ViTPetFaceConfig()
