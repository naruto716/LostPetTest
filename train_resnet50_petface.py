#!/usr/bin/env python3
"""
ResNet50 Training Launcher for PetFace Dataset
Classic CNN architecture - comparison with transformer models
"""

import sys
import os

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import ResNet50 PetFace config
from config_resnet50_petface import cfg

# Temporarily override the cfg in the training module
import train_petface
train_petface.cfg = cfg

# Import and run main
from train_petface import main

if __name__ == "__main__":
    print("="*70)
    print("Starting ResNet50 Training on PetFace Dataset")
    print("="*70)
    print(f"Output:     {cfg.OUTPUT_DIR}")
    print(f"Backbone:   {cfg.BACKBONE}")
    print(f"Feat Dim:   {cfg.EMBED_DIM}")
    print(f"Fine-tune:  {not cfg.FREEZE_BACKBONE}")
    print(f"Image Size: {cfg.IMAGE_SIZE}")
    print(f"Dataset:    PetFace 30k dogs")
    print(f"Target:     Compare with DINOv3-B baseline (87% mAP)")
    print("="*70)
    
    main()

