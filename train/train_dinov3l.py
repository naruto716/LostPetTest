#!/usr/bin/env python3
"""
DINOv3-L Baseline Training Launcher
Establish strong baseline with DINOv3-Large
"""

import sys
import os

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import DINOv3-L config
from config_dinov3l import cfg

# Temporarily override the cfg in the training module
import train_dogreid
train_dogreid.cfg = cfg

# Import and run main
from train_dogreid import main

if __name__ == "__main__":
    print("🎯 Starting DINOv3-L Baseline Training")
    print(f"📁 Output: {cfg.OUTPUT_DIR}")
    print(f"🧠 Backbone: {cfg.BACKBONE}")
    print(f"📏 Embedding Dim: {cfg.EMBED_DIM}")
    print(f"🔧 Backbone Frozen: {cfg.FREEZE_BACKBONE}")
    print(f"📊 Target: Strong baseline performance")
    print("=" * 60)
    
    main()

