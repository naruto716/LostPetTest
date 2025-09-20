#!/usr/bin/env python3
"""
SWIN-L Training Launcher
Different architecture family - perfect for research comparison!
"""

import sys
import os

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import SWIN-L config
from config_swinl import cfg

# Temporarily override the cfg in the training module
import train_dogreid
train_dogreid.cfg = cfg

# Import and run main
from train_dogreid import main

if __name__ == "__main__":
    print("🏗️ Starting SWIN-L Research Training")
    print(f"📁 Output: {cfg.OUTPUT_DIR}")
    print(f"🧠 Backbone: {cfg.BACKBONE}")
    print(f"🔧 Transformer Fine-tuning: {not cfg.FREEZE_BACKBONE}")
    print(f"📐 Image Size: {cfg.IMAGE_SIZE} (SWIN requirement)")
    print(f"📊 Expected Performance: Different architecture family!")
    print(f"🎯 Perfect for: Architecture comparisons & novel techniques")
    print("=" * 70)
    
    main()
