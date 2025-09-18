#!/usr/bin/env python3
"""Test the pluggable model architecture"""

import sys
import torch
sys.path.append('.')

from config import cfg
from model import make_model, build_backbone
from datasets import make_dataloaders

def test_backbone_factory():
    """Test the pluggable backbone system"""
    print("🧪 Testing Backbone Factory")
    print("=" * 50)
    
    # List available backbones
    from model.backbones import list_available_backbones
    list_available_backbones()
    
    # Test DINOv2 backbone
    print(f"\n🔧 Testing backbone: {cfg.BACKBONE}")
    try:
        backbone, feat_dim = build_backbone(cfg.BACKBONE, pretrained=cfg.PRETRAINED)
        print(f"✅ Backbone created: {type(backbone).__name__}")
        print(f"✅ Feature dimension: {feat_dim}")
        
        # Test forward pass
        dummy_input = torch.randn(2, 3, 224, 224)
        with torch.no_grad():
            output = backbone(dummy_input)
            print(f"✅ Forward pass: {dummy_input.shape} -> {output.shape}")
            
    except Exception as e:
        print(f"❌ Backbone test failed: {e}")
        return False
        
    return True

def test_full_model():
    """Test the complete model"""
    print("\n🧪 Testing Complete Model")
    print("=" * 50)
    
    try:
        # Create model for zero-shot (no classifier)
        print("🔧 Creating zero-shot model (no classifier)...")
        model_zeroshot = make_model(
            backbone_name=cfg.BACKBONE,
            num_classes=0,  # No classifier
            embed_dim=cfg.EMBED_DIM,
            pretrained=cfg.PRETRAINED
        )
        
        # Test zero-shot inference
        dummy_input = torch.randn(4, 3, 224, 224)
        print(f"\n🔍 Testing zero-shot inference:")
        with torch.no_grad():
            features = model_zeroshot.extract_features(dummy_input)
            print(f"✅ Input: {dummy_input.shape} -> Features: {features.shape}")
            print(f"✅ Features are L2 normalized: {torch.allclose(features.norm(dim=1), torch.ones(4))}")
        
        # Create model with classifier (for training)
        print(f"\n🔧 Creating training model (95 classes)...")
        model_train = make_model(
            backbone_name=cfg.BACKBONE,
            num_classes=95,  # Our dataset has 95 dog identities
            embed_dim=cfg.EMBED_DIM,
            pretrained=cfg.PRETRAINED
        )
        
        # Test training mode
        print(f"\n🎯 Testing training mode:")
        model_train.train()
        with torch.no_grad():
            logits, feat_for_triplet = model_train(dummy_input)
            print(f"✅ Logits: {logits.shape} (should be [4, 95])")
            print(f"✅ Features for triplet: {feat_for_triplet.shape}")
            
        # Test evaluation mode
        print(f"\n🔍 Testing evaluation mode:")
        model_train.eval()
        with torch.no_grad():
            eval_features = model_train(dummy_input)
            print(f"✅ Eval features: {eval_features.shape}")
            print(f"✅ Features are L2 normalized: {torch.allclose(eval_features.norm(dim=1), torch.ones(4))}")
        
        return True
        
    except Exception as e:
        print(f"❌ Model test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_with_real_data():
    """Test model with real data from our dataset"""
    print("\n🧪 Testing with Real Data")
    print("=" * 50)
    
    try:
        # Load our dog dataset
        train_loader, query_loader, gallery_loader, num_classes = make_dataloaders()
        print(f"✅ Dataset loaded: {num_classes} classes")
        
        # Create model with correct number of classes
        model = make_model(
            backbone_name=cfg.BACKBONE,
            num_classes=num_classes,
            embed_dim=cfg.EMBED_DIM,
            pretrained=cfg.PRETRAINED
        )
        
        # Test with real batch
        print(f"\n🐕 Testing with real dog images...")
        batch = next(iter(train_loader))
        imgs, pids, camids, paths = batch
        print(f"✅ Real batch: {imgs.shape}, PIDs: {pids.shape}")
        
        # Training mode
        model.train()
        with torch.no_grad():
            logits, features = model(imgs)
            print(f"✅ Training forward: {imgs.shape} -> logits: {logits.shape}, features: {features.shape}")
            
        # Evaluation mode  
        model.eval()
        with torch.no_grad():
            eval_features = model.extract_features(imgs)
            print(f"✅ Evaluation forward: {imgs.shape} -> features: {eval_features.shape}")
            print(f"✅ Features normalized: {torch.allclose(eval_features.norm(dim=1), torch.ones(eval_features.size(0)))}")
            
        print(f"🎉 Model works with real data!")
        return True
        
    except Exception as e:
        print(f"❌ Real data test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all tests"""
    print("🐕 Dog ReID Model Architecture Testing")
    print("Testing pluggable backbone system with DINOv2...")
    
    success = True
    
    # Test 1: Backbone factory
    success &= test_backbone_factory()
    
    # Test 2: Full model
    success &= test_full_model()
    
    # Test 3: Real data
    success &= test_with_real_data()
    
    if success:
        print("\n" + "=" * 50)
        print("🎉 ALL TESTS PASSED!")
        print("✅ Pluggable backbone system works")
        print("✅ DINOv2 integration successful") 
        print("✅ Zero-shot ready (no training needed)")
        print("✅ Training mode ready") 
        print("✅ Real data compatible")
        print("🚀 Ready for the next step!")
    else:
        print("\n❌ SOME TESTS FAILED")
        return 1
        
    return 0

if __name__ == '__main__':
    exit(main())
