#!/usr/bin/env python3
"""
DINOv3-L Training Launcher for PetFace Dataset
Larger model for higher performance baseline
"""

import sys
import os

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import DINOv3-L PetFace config
from config_dinov3l_petface import cfg

# Temporarily override the cfg in the training module
import train_petface
train_petface.cfg = cfg

# Import and run main
from train_petface import main

if __name__ == "__main__":
    print("="*70)
    print("Starting DINOv3-L Training on PetFace Dataset")
    print("="*70)
    print(f"Output:     {cfg.OUTPUT_DIR}")
    print(f"Backbone:   {cfg.BACKBONE}")
    print(f"Feat Dim:   {cfg.EMBED_DIM}")
    print(f"Fine-tune:  {not cfg.FREEZE_BACKBONE}")
    print(f"Dataset:    PetFace 30k dogs")
    print(f"Target:     Beat DINOv3-B baseline (87% mAP)")
    print("="*70)
    
    main()

