#!/usr/bin/env python3
"""
DINOv3-B Training Launcher
Quick way to train with DINOv3-B for research
"""

import sys
import os

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import DINOv3-B config
from config_dinov3b import cfg

# Temporarily override the cfg in the training module
import train_dogreid
train_dogreid.cfg = cfg

# Import and run main
from train_dogreid import main

if __name__ == "__main__":
    print("🎯 Starting DINOv3-B Research Training")
    print(f"📁 Output: {cfg.OUTPUT_DIR}")
    print(f"🧠 Backbone: {cfg.BACKBONE}")
    print(f"🔧 Backbone Frozen: {cfg.FREEZE_BACKBONE}")
    print(f"📊 Expected Performance: 85-95% mAP (room for improvement!)")
    print("=" * 60)
    
    main()
