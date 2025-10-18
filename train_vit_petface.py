"""
============
📊 VALIDATION RESULTS - Epoch 32
================================================================================
   mAP:      85.93%
   Rank-1:   87.00%
   Rank-5:   94.50%
   Rank-10:  96.67%
   Eval time: 80.3s
================================================================================
"""

import sys
import os
import logging
import torch

# -----------------------------
# Force CPU mode before anything else
# -----------------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
USE_CUDA = DEVICE.type == "cuda"

# Monkey-patch torch.cuda.synchronize to avoid errors on CPU-only PyTorch
if not USE_CUDA:
    torch.cuda.synchronize = lambda: None

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import ViT config
from config_vit_petface import cfg

# Override device in cfg
cfg.DEVICE = DEVICE

# -----------------------------
# Setup logging safely
# -----------------------------
def safe_info(logger, msg):
    """Log safely on Windows if GBK codec cannot encode emoji."""
    try:
        logger.info(msg)
    except UnicodeEncodeError:
        logger.info(msg.encode("ascii", "ignore").decode("ascii"))

logger = logging.getLogger("petface")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s: %(message)s")
if not USE_CUDA:
    safe_info(logger, "⚠️ CUDA not available, forcing CPU training")

# -----------------------------
# Import training module after CPU patch
# -----------------------------
import train_petface
train_petface.cfg = cfg
from train_petface import main

# -----------------------------
# Launch training
# -----------------------------
if __name__ == "__main__":
    print("="*70)
    print("🚀 STARTING ViT Tiny TRAINING ON PETFACE DATASET")
    print("="*70)
    print(f"Output Dir: {cfg.OUTPUT_DIR}")
    print(f"Backbone:   {cfg.BACKBONE}")
    print(f"Feat Dim:   {cfg.EMBED_DIM}")
    print(f"Fine-tune:  {not cfg.FREEZE_BACKBONE}")
    print(f"Image Size: {cfg.IMAGE_SIZE}")
    print(f"Dataset:    PetFace")
    print(f"Device:     {cfg.DEVICE}")
    print("="*70)

    main()
