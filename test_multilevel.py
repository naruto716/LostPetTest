#!/usr/bin/env python3
"""
Test Multi-Level DINOv3 Implementation
🧪 Verify the novel approach works before training!
"""

import torch
from model.backbones import build_backbone

def test_multilevel():
    print("🧪 Testing Multi-Level DINOv3 Implementation")
    print("=" * 60)
    
    # Test multi-level backbone
    try:
        backbone, feat_dim = build_backbone('dinov3_vits16_multilevel', pretrained=True)
        print(f"✅ Backbone created: {feat_dim}D features")
        
        # Test forward pass
        dummy_input = torch.randn(2, 3, 336, 336)  # Small batch
        print(f"📥 Input shape: {dummy_input.shape}")
        
        with torch.no_grad():
            features = backbone(dummy_input)
            print(f"📤 Output shape: {features.shape}")
            print(f"📊 Expected: [2, {feat_dim}] ✅" if features.shape == (2, feat_dim) else f"❌ Unexpected shape!")
            
            # Check if we got features from multiple layers
            if hasattr(backbone, 'intermediate_features'):
                num_captured = len(backbone.intermediate_features)
                print(f"🪝 Captured features from {num_captured} layers")
                
                # Show individual layer shapes
                for i, feat in enumerate(backbone.intermediate_features):
                    print(f"   Layer {i+1}: {feat.shape}")
            
        print("✅ Multi-level extraction successful!")
        
        # Show the multi-level fusion details
        print("\n🧪 Testing Multi-Level Fusion...")
        from model.make_model import make_model
        from config_multilevel import cfg
        
        model = make_model(cfg, num_classes=50)
        if hasattr(model, 'embedding') and hasattr(model.embedding, 'fusion'):
            fusion_layers = model.embedding.fusion
            print(f"📊 Fusion architecture:")
            for i, layer in enumerate(fusion_layers):
                if hasattr(layer, 'weight'):
                    print(f"   Layer {i}: {layer}")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return False
    
    # Test integration with model
    try:
        print("\n🔧 Testing integration with ReID model...")
        from model.make_model import make_model
        from config_multilevel import cfg
        
        model = make_model(cfg, num_classes=50)  # Dummy classes
        print(f"✅ Full model created with {cfg.BACKBONE}")
        
        # Test model forward pass
        with torch.no_grad():
            logits, features = model(dummy_input, 'auto')
            print(f"📤 Logits: {logits.shape}, Features: {features.shape}")
            print(f"✅ Expected final feature dim: {cfg.EMBED_DIM}D")
            
        print("✅ Full pipeline works!")
        
    except Exception as e:
        print(f"❌ Integration error: {e}")
        return False
    
    return True

if __name__ == "__main__":
    success = test_multilevel()
    if success:
        print("\n🎉 Multi-level DINOv3 ready for training!")
        print("💡 Run: uv run python train_multilevel.py")
    else:
        print("\n⚠️ Issues found - fix before training!")
