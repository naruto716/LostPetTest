"""
Simple config for Dog ReID.
No over-engineering - just the hyperparameters we actually need.
"""

class Config:
    # P×K Sampling hyperparameters (scaled up for server hardware!)
    IMS_PER_BATCH = 128     # Total batch size (P×K) - 2x larger for your server 🚀
    NUM_INSTANCE = 4        # K - instances per identity
    NUM_WORKERS = 8         # Multiprocessing workers (you have 48 vCPUs!)
    
    # Dataset paths
    ROOT_DIR = "."
    IMAGES_DIR = "images"
    TRAIN_SPLIT = "splits/train.csv"
    QUERY_SPLIT = "splits/query.csv"
    GALLERY_SPLIT = "splits/gallery.csv"
    USE_CAMID = False
    
    # Image preprocessing (larger images for better feature extraction)
    IMAGE_SIZE = (336, 336)    # Larger than 224 - server can handle it! 
    PIXEL_MEAN = [0.485, 0.456, 0.406]
    PIXEL_STD = [0.229, 0.224, 0.225]
    PADDING = 15               # Slightly more padding for larger images
    RE_PROB = 0.5             # Random erasing probability
    
    # Model architecture (scaled up for server!)
    BACKBONE = 'dinov2_vitl14'  # ViT-L/14 instead of ViT-B/14 - much more powerful! 🚀
    EMBED_DIM = 768            # Larger embedding (1024->768 to prevent overfitting)
    PRETRAINED = True          # Use pretrained weights
    BN_NECK = True            # Use BN-neck (important for ReID)

# Simple config object
cfg = Config()