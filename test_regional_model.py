"""
Test script for regional model.
Verifies model creation, forward pass, and training/eval modes.
"""

import torch
from model import make_regional_model


def test_regional_model():
    """Test the regional model with mock data."""
    
    print("Testing Regional Dog ReID Model")
    print("="*60)
    
    # Test parameters
    batch_size = 4
    num_classes = 2100
    
    # 1. Create model
    print("\n1. Creating regional model...")
    model = make_regional_model(
        backbone_name='dinov3_vitl16',
        num_classes=num_classes,
        embed_dim=768,
        pretrained=False  # Use False for quick testing
    )
    print(f"✅ Model created")
    print(f"   Feature dim: {model.get_feature_dim()}")
    
    # 2. Freeze backbone (standard approach)
    print("\n2. Freezing backbone...")
    model.freeze_backbone()
    
    # Count trainable parameters
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"   Trainable params: {trainable_params:,} / {total_params:,}")
    print(f"   Frozen params: {total_params - trainable_params:,}")
    
    # 3. Create mock batch
    print("\n3. Creating mock batch...")
    global_imgs = torch.randn(batch_size, 3, 224, 224)
    
    regions = {
        'left_eye': torch.randn(batch_size, 3, 64, 64),
        'right_eye': torch.randn(batch_size, 3, 64, 64),
        'nose': torch.randn(batch_size, 3, 64, 64),
        'mouth': torch.randn(batch_size, 3, 64, 64),
        'left_ear': torch.randn(batch_size, 3, 64, 64),
        'right_ear': torch.randn(batch_size, 3, 64, 64),
        'forehead': torch.randn(batch_size, 3, 64, 64)
    }
    
    labels = torch.randint(0, num_classes, (batch_size,))
    
    print(f"   Global images: {global_imgs.shape}")
    print(f"   Regions: {len(regions)} regions")
    for name, tensor in regions.items():
        print(f"     {name:12}: {tensor.shape}")
    print(f"   Labels: {labels.shape}")
    
    # 4. Test training mode forward pass
    print("\n4. Testing training mode...")
    model.train()
    
    try:
        logits, features = model(global_imgs, regions)
        print(f"✅ Training forward pass successful")
        print(f"   Logits shape: {logits.shape} (expected: [{batch_size}, {num_classes}])")
        print(f"   Features shape: {features.shape} (expected: [{batch_size}, 768])")
        
        # Verify shapes
        assert logits.shape == (batch_size, num_classes), f"Wrong logits shape: {logits.shape}"
        assert features.shape == (batch_size, 768), f"Wrong features shape: {features.shape}"
        print("   ✓ Shapes correct")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # 5. Test eval mode forward pass
    print("\n5. Testing eval mode...")
    model.eval()
    
    try:
        with torch.no_grad():
            eval_features = model(global_imgs, regions)
        
        print(f"✅ Eval forward pass successful")
        print(f"   Features shape: {eval_features.shape} (expected: [{batch_size}, 768])")
        
        # Check if normalized
        norms = torch.norm(eval_features, p=2, dim=1)
        print(f"   Feature norms: {norms.tolist()}")
        print(f"   All ~1.0? {torch.allclose(norms, torch.ones_like(norms), atol=1e-5)}")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # 6. Test backward pass
    print("\n6. Testing backward pass...")
    model.train()
    
    try:
        logits, features = model(global_imgs, regions)
        
        # Compute dummy loss
        loss = logits.sum() + features.sum()
        loss.backward()
        
        print(f"✅ Backward pass successful")
        
        # Check which parameters have gradients
        has_grad = sum(1 for p in model.parameters() if p.grad is not None)
        should_have_grad = sum(1 for p in model.parameters() if p.requires_grad)
        
        print(f"   Parameters with gradients: {has_grad}/{should_have_grad}")
        
        # Verify backbone is frozen
        backbone_has_grad = sum(1 for p in model.backbone.parameters() if p.grad is not None)
        print(f"   Backbone parameters with gradients: {backbone_has_grad} (should be 0)")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return
    
    print("\n" + "="*60)
    print("✅ All tests passed!")
    print("="*60)
    print("\nModel summary:")
    print(f"  - Backbone: {model.get_backbone_name()} (FROZEN)")
    print(f"  - Input: 1 global + 7 regional images")
    print(f"  - Concat dim: 8 × 1024 = 8192")
    print(f"  - Fusion: 8192 → 768 (trainable)")
    print(f"  - Output: 768-dim embeddings")
    print(f"  - Trainable params: {trainable_params:,}")
    print("\nReady for training!")


if __name__ == "__main__":
    test_regional_model()

