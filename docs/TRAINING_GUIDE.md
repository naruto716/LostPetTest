# 🚀 Dog ReID Training Guide

Complete training pipeline based on **CLIP-ReID's proven structure** for state-of-the-art dog re-identification.

## 🎯 Quick Start

### **1. Test the Pipeline (Recommended First Step)**
```bash
uv run python test_training_pipeline.py
```
This verifies all components work together before full training.

### **2. Start Training (Conservative Approach)**
```bash
# Frozen backbone (recommended start)
uv run python train_dogreid.py --freeze_backbone

# Output will be in: ./outputs/dogreid_training/
```

### **3. Monitor Training**
```bash
# Check logs
tail -f ./outputs/dogreid_training/train.log

# Look for:
# - Loss decreasing over epochs
# - mAP improving during evaluation
# - No gradient explosions/vanishing
```

---

## 📊 Training Modes

### **🛡️ Mode 1: Frozen Backbone (Recommended Start)**
```bash
uv run python train_dogreid.py --freeze_backbone
```
**What it does:**
- Freezes DINOv3 backbone weights  
- Only trains embedding + BN-neck + classifier
- Safe approach, can't break DINOv3's features
- Fast training, good baseline

**When to use:** Always start here!

### **🚀 Mode 2: Fine-tuning (Advanced)**
```bash
uv run python train_dogreid.py
```
**What it does:**
- Fine-tunes entire model including backbone
- Uses 10x smaller LR for backbone
- Higher potential performance but riskier

**When to use:** If frozen approach gives mAP < 70%

---

## 🔧 Advanced Options

### **Custom Training Duration**
```bash
uv run python train_dogreid.py --epochs 100
```

### **Custom Output Directory**
```bash
uv run python train_dogreid.py --output_dir ./my_experiment
```

### **Resume Training**
```bash
uv run python train_dogreid.py --resume ./outputs/dogreid_training/checkpoint_epoch_20.pth
```

### **Evaluation Only**
```bash
uv run python train_dogreid.py --eval_only --resume ./outputs/dogreid_training/best_model.pth
```

---

## 📈 Understanding the Output

### **Training Logs**
```
Epoch[1] Iteration[10/25] Loss: 4.234, Acc: 0.125, Base Lr: 3.00e-05
   Loss components: id_loss: 3.456, triplet_loss: 0.778, total_loss: 4.234
```

**What to look for:**
- ✅ **Loss decreasing**: 4.234 → 3.890 → 3.456 over epochs
- ✅ **Accuracy increasing**: 0.125 → 0.250 → 0.375
- ✅ **Stable training**: No NaN or sudden spikes

### **Evaluation Results**
```
Validation Results - Epoch: 10
mAP: 45.2%
CMC curve, Rank-1  :78.3%
CMC curve, Rank-5  :89.1%
CMC curve, Rank-10 :93.4%
🏆 New best model saved: mAP=45.2% at epoch 10
```

**Performance expectations:**
- 🥉 **Good**: mAP > 60%, Rank-1 > 80%
- 🥈 **Very good**: mAP > 75%, Rank-1 > 90%  
- 🥇 **Excellent**: mAP > 85%, Rank-1 > 95%

---

## 🎯 Training Strategy

### **Phase 1: Baseline (Frozen Backbone)**
```bash
uv run python train_dogreid.py --freeze_backbone --epochs 60
```
- ⏱️ **Time**: ~2-3 hours on your A10G
- 🎯 **Goal**: Establish baseline performance
- 📊 **Expected**: mAP 50-70% if DINOv3 features are good

### **Phase 2: Fine-tuning (If Needed)**
```bash
# Only if Phase 1 gives mAP < 70%
uv run python train_dogreid.py --epochs 60
```
- ⏱️ **Time**: ~4-5 hours (backbone learning)
- 🎯 **Goal**: Adapt DINOv3 to your dogs
- 📊 **Expected**: +5-15% mAP improvement

### **Phase 3: Optimization**
- Adjust learning rates in `config_training.py`
- Try different loss weights
- Experiment with data augmentation

---

## 🔍 Evaluation

### **Standalone Evaluation**
```bash
uv run python test_dogreid.py --checkpoint ./outputs/dogreid_training/best_model.pth
```

### **Understanding Metrics**

**mAP (mean Average Precision):**
- Most important metric for ReID
- Higher = better overall retrieval performance
- Range: 0-100%

**CMC (Cumulative Matching Characteristics):**
- Rank-1: % of queries where correct dog is #1 match
- Rank-5: % of queries where correct dog is in top 5
- Rank-10: % of queries where correct dog is in top 10

---

## 🐛 Troubleshooting

### **Out of Memory**
```bash
# Reduce batch size in config_training.py
IMS_PER_BATCH = 64  # Instead of 128
```

### **No Improvement**
```bash
# Check if data is loaded correctly
uv run python test_training_pipeline.py

# Try different learning rate
BASE_LR = 1e-4  # Instead of 3e-4
```

### **Loss Exploding**
```bash
# Use smaller learning rate
BASE_LR = 1e-5
```

### **Too Slow**
```bash
# Use smaller images
IMAGE_SIZE = (224, 224)  # Instead of (336, 336)
```

---

## 📁 Output Files

```
outputs/dogreid_training/
├── train.log                 # Training logs
├── best_model.pth           # Best model (highest mAP)
├── checkpoint_epoch_20.pth  # Periodic checkpoints
├── checkpoint_epoch_40.pth
├── checkpoint_epoch_60.pth
└── evaluation_results.pth   # Detailed evaluation results
```

---

## 🎓 Pro Tips

1. **Always start with frozen backbone** - it's safer and faster
2. **Monitor gradient norms** - should be stable, not exploding
3. **Save multiple checkpoints** - in case you need to backtrack
4. **Use your server's power** - the current config is optimized for 4x A10G
5. **Trust the process** - CLIP-ReID's approach is battle-tested

---

## 🏆 Success Criteria

**Your training is successful if:**
- ✅ Loss steadily decreases over epochs
- ✅ mAP reaches 60%+ (good) or 75%+ (excellent)
- ✅ No NaN/inf values in losses
- ✅ Gradients flowing properly (visible in logs)
- ✅ Best model saved with reasonable performance

**Ready to train? Start with the pipeline test!** 🚀

```bash
uv run python test_training_pipeline.py
```
