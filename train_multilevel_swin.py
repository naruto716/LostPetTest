#!/usr/bin/env python3
"""
Multi-Level SWIN Training Launcher
🧪 NOVEL RESEARCH: TRUE hierarchical multi-scale features!
Perfect adaptation of Tiger ReID regional pooling to hierarchical transformers
"""

import sys
import os

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import multi-level SWIN config
from config_multilevel_swin import cfg

# Temporarily override the cfg in the training module
import train_dogreid
train_dogreid.cfg = cfg

# Import and run main
from train_dogreid import main

if __name__ == "__main__":
    print("🧪 Starting Multi-Level SWIN Research Training")
    print("=" * 70)
    print("🎯 NOVEL APPROACH: Hierarchical Multi-Scale Features")
    print("📚 Inspired by: Amur Tiger ReID (ICCV 2019)")
    print("🔬 Innovation: SWIN hierarchical stages → TRUE multi-scale")
    print("🏗️ Architecture: Different stages, different resolutions!")
    print("=" * 70)
    print(f"📁 Output: {cfg.OUTPUT_DIR}")
    print(f"🧠 Backbone: {cfg.BACKBONE}")
    print(f"📊 IMPROVED Feature Flow (Skip low-level noise!):")
    print(f"   Stage 3: [H/16×W/16, 384] → Global pool → [384] (Mid-level)")
    print(f"   Stage 4: [H/32×W/32, 768] → Global pool → [768] (High-level)")
    print(f"   Concat: [384+768=1152] → Fusion → [{cfg.EMBED_DIM}]")
    print(f"📐 Image Size: {cfg.IMAGE_SIZE} (SWIN requirement)")
    print(f"🎯 Expected: TRUE multi-scale diversity like CNNs!")
    print("=" * 70)
    
    main()
