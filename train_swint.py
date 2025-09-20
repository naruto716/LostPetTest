#!/usr/bin/env python3
"""
SWIN-Tiny Training Launcher
SMALLEST SWIN - perfect research baseline!
"""

import sys
import os

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import SWIN-Tiny config
from config_swint import cfg

# Temporarily override the cfg in the training module
import train_dogreid
train_dogreid.cfg = cfg

# Import and run main
from train_dogreid import main

if __name__ == "__main__":
    print("🏗️ Starting SWIN-TINY Research Training")
    print(f"📁 Output: {cfg.OUTPUT_DIR}")
    print(f"🧠 Backbone: {cfg.BACKBONE} (768D features)")
    print(f"🔧 Transformer Fine-tuning: {not cfg.FREEZE_BACKBONE}")
    print(f"📐 Image Size: {cfg.IMAGE_SIZE} (SWIN requirement)")
    print(f"📊 Expected Performance: 50-80% mAP (EXCELLENT for research!)")
    print(f"🎯 Perfect for: Novel architecture techniques!")
    print("=" * 70)
    
    main()
