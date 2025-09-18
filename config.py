"""
Simple config for the hyperparameters we actually have right now.
No over-engineering - just the basics we need for dataset loading.
"""

class Config:
    # P×K Sampling hyperparameters
    IMS_PER_BATCH = 64      # Total batch size (P×K)
    NUM_INSTANCE = 4        # K - instances per identity
    NUM_WORKERS = 0         # Number of data loading workers (0 for now to avoid multiprocessing issues)
    
    # Dataset paths  
    ROOT_DIR = "."
    IMAGES_DIR = "images"
    TRAIN_SPLIT = "splits/train.csv"
    QUERY_SPLIT = "splits/query.csv"
    GALLERY_SPLIT = "splits/gallery.csv"
    USE_CAMID = False
    
    # Basic image preprocessing
    IMAGE_SIZE = (224, 224)
    PIXEL_MEAN = [0.485, 0.456, 0.406]
    PIXEL_STD = [0.229, 0.224, 0.225]
    PADDING = 10
    RE_PROB = 0.5  # Random erasing probability

# Simple config object
cfg = Config()
