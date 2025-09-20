#!/usr/bin/env python3
"""
Multi-Level DINOv3-B Training Launcher
🧪 NOVEL RESEARCH: DINOv3-Base with multi-level features!
Regional pooling adapted for Vision Transformers - using Base model for better capacity
"""

import sys
import os

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import multi-level config
from config_multilevel_b import cfg

# Temporarily override the cfg in the training module
import train_dogreid
train_dogreid.cfg = cfg

# Import and run main
from train_dogreid import main

if __name__ == "__main__":
    print("🧪 Starting Multi-Level DINOv3-B Research Training")
    print("=" * 70)
    print("🎯 NOVEL APPROACH: Transformer Regional Pooling")
    print("📚 Inspired by: Amur Tiger ReID (ICCV 2019)")
    print("🔬 Innovation: Multi-level Vision Transformer features")
    print("🧠 Model: DINOv3-Base (768D × 5 layers = 3840D)")
    print("=" * 70)
    print(f"📁 Output: {cfg.OUTPUT_DIR}")
    print(f"🧠 Backbone: {cfg.BACKBONE}")
    print(f"📊 Feature Flow: 768D × 5 layers → 3840D → {cfg.EMBED_DIM}D")
    print(f"🔧 Layers: [3, 6, 9, 12, final] multi-scale extraction")
    print(f"🎯 Expected: Novel technique with substantial improvement potential!")
    print("=" * 70)
    
    main()
