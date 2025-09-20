#!/usr/bin/env python3
"""
Multi-Level DINOv3 Training Launcher
🧪 NOVEL RESEARCH: Regional pooling adapted for Vision Transformers!
Inspired by Amur Tiger ReID paper - extracts multi-scale transformer features
"""

import sys
import os

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import multi-level config
from config_multilevel import cfg

# Temporarily override the cfg in the training module
import train_dogreid
train_dogreid.cfg = cfg

# Import and run main
from train_dogreid import main

if __name__ == "__main__":
    print("🧪 Starting Multi-Level DINOv3 Research Training")
    print("=" * 70)
    print("🎯 NOVEL APPROACH: Transformer Regional Pooling")
    print("📚 Inspired by: Amur Tiger ReID (ICCV 2019)")
    print("🔬 Innovation: Multi-level Vision Transformer features")
    print("=" * 70)
    print(f"📁 Output: {cfg.OUTPUT_DIR}")
    print(f"🧠 Backbone: {cfg.BACKBONE}")
    print(f"📊 Feature Flow: 384D × 5 layers → 1920D → {cfg.EMBED_DIM}D")
    print(f"🔧 Layers: [3, 6, 9, 12, final] multi-scale extraction")
    print(f"🎯 Expected: Novel technique with clear improvement potential!")
    print("=" * 70)
    
    main()
