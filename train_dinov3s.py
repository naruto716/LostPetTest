#!/usr/bin/env python3
"""
DINOv3-Small Training Launcher
SMALLEST DINOv3 - maximum improvement potential!
"""

import sys
import os

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import DINOv3-Small config
from config_dinov3s import cfg

# Temporarily override the cfg in the training module
import train_dogreid
train_dogreid.cfg = cfg

# Import and run main
from train_dogreid import main

if __name__ == "__main__":
    print("🎯 Starting DINOv3-SMALL Research Training")
    print(f"📁 Output: {cfg.OUTPUT_DIR}")
    print(f"🧠 Backbone: {cfg.BACKBONE} (384D features)")
    print(f"🔧 Fine-tuning: {not cfg.FREEZE_BACKBONE}")
    print(f"📊 Expected Performance: 60-85% mAP (LOTS of improvement room!)")
    print(f"🎯 Perfect for: Maximum research potential!")
    print("=" * 70)
    
    main()
