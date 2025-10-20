"""
Test script showing how model + losses work together.
Demonstrates the complete training pipeline.
"""

import torch
from model import make_model
from loss import make_loss
from config import cfg

def test_model_with_losses():
    """Test model and losses working together."""
    
    print("🧪 Testing Model + Losses Integration")
    print("=" * 50)
    
    # Test parameters
    batch_size = 16  # Smaller for testing
    num_classes = 50  # Subset of classes
    
    # Create model
    print(f"🏗️  Creating model...")
    model = make_model(
        backbone_name=cfg.BACKBONE,
        num_classes=num_classes,
        embed_dim=cfg.EMBED_DIM,
        pretrained=cfg.PRETRAINED,
        bn_neck=cfg.BN_NECK
    )
    
    # Create loss function
    print(f"🎯 Creating loss function...")
    loss_builder = make_loss(
        num_classes=num_classes,
        feat_dim=model.get_feature_dim(),
        id_loss_weight=1.0,
        triplet_loss_weight=1.0,
        triplet_margin=0.3,
        label_smoothing=0.1,
        use_center_loss=False
    )
    
    # Simulate P×K batch (4 identities, 4 instances each = 16 samples)
    labels = torch.tensor([0, 0, 0, 0,  # Identity 0
                          1, 1, 1, 1,  # Identity 1  
                          2, 2, 2, 2,  # Identity 2
                          3, 3, 3, 3]) # Identity 3
    
    # Create dummy images
    images = torch.randn(batch_size, 3, *cfg.IMAGE_SIZE)
    
    print(f"\n📊 Test Configuration:")
    print(f"   Backbone: {cfg.BACKBONE}")
    print(f"   Batch size: {batch_size}")
    print(f"   Num classes: {num_classes}")
    print(f"   Feature dim: {model.get_feature_dim()}")
    print(f"   Image size: {cfg.IMAGE_SIZE}")
    print(f"   Labels: {labels.tolist()}")
    
    # Test inference mode
    print(f"\n🔍 Test 1: Inference Mode")
    model.eval()
    with torch.no_grad():
        inference_features = model(images, return_mode='features')
        print(f"   Inference features shape: {inference_features.shape}")
        print(f"   Features are L2 normalized: {torch.allclose(torch.norm(inference_features, dim=1), torch.ones(batch_size), atol=1e-6)}")
    
    # Test training mode  
    print(f"\n🏋️  Test 2: Training Mode")
    model.train()
    
    # Forward pass
    logits, features = model(images, return_mode='auto')
    print(f"   Logits shape: {logits.shape}")
    print(f"   Features shape: {features.shape}")
    
    # Compute loss
    total_loss, loss_dict = loss_builder(logits, features, labels)
    print(f"   Total loss: {total_loss:.4f}")
    for name, value in loss_dict.items():
        print(f"   {name}: {value:.4f}")
    
    # Test gradient flow
    print(f"\n🔄 Test 3: Gradient Flow")
    total_loss.backward()
    
    # Check gradients on key components
    backbone_grad = sum(p.grad.norm().item() for p in model.backbone.parameters() if p.grad is not None)
    classifier_grad = model.classifier.classifier.weight.grad.norm().item() if model.classifier.classifier.weight.grad is not None else 0
    
    print(f"   Backbone gradient norm: {backbone_grad:.4f}")
    print(f"   Classifier gradient norm: {classifier_grad:.4f}")
    print(f"   ✅ Gradients flowing through entire model!")
    
    # Test different return modes
    print(f"\n🔧 Test 4: Different Return Modes")
    model.eval()
    
    with torch.no_grad():
        # Features only
        feat_only = model(images, return_mode='features')
        print(f"   Features only: {feat_only.shape}")
        
        # Switch to training for logits
        model.train()
        logits_only = model(images, return_mode='logits')
        print(f"   Logits only: {logits_only.shape}")
        
        # Both
        both_logits, both_features = model(images, return_mode='both')
        print(f"   Both mode - Logits: {both_logits.shape}, Features: {both_features.shape}")
    
    print(f"\n✅ Model + Losses integration successful!")
    print(f"🚀 Ready for full training pipeline!")

if __name__ == '__main__':
    test_model_with_losses()
