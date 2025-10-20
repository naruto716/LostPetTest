#!/usr/bin/env python3
"""
SWIN-Base Training Launcher for PetFace Dataset
Different architecture family - comparison with DINOv3 baseline
"""

import sys
import os

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import SWIN-B PetFace config
from config_swinb_petface import cfg

# Temporarily override the cfg in the training module
import train_petface
train_petface.cfg = cfg

# Import and run main
from train_petface import main

if __name__ == "__main__":
    print("="*70)
    print("Starting SWIN-Base Training on PetFace Dataset")
    print("="*70)
    print(f"Output:     {cfg.OUTPUT_DIR}")
    print(f"Backbone:   {cfg.BACKBONE}")
    print(f"Feat Dim:   {cfg.EMBED_DIM}")
    print(f"Fine-tune:  {not cfg.FREEZE_BACKBONE}")
    print(f"Image Size: {cfg.IMAGE_SIZE} (SWIN requirement)")
    print(f"Dataset:    PetFace 30k dogs")
    print(f"Target:     Compare with DINOv3-B baseline (87% mAP)")
    print("="*70)
    
    main()

