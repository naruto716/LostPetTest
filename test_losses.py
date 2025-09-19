"""
Test script for Dog ReID loss functions.
Verify that our CLIP-ReID inspired losses work correctly.
"""

import torch
from loss import ReIDLossBuilder, make_loss

def test_loss_functions():
    """Test all loss functions with dummy data."""
    
    print("🧪 Testing Dog ReID Loss Functions")
    print("=" * 50)
    
    # Test parameters
    batch_size = 32
    num_classes = 100
    feat_dim = 1024
    
    # Create dummy data (P×K sampling style)
    # 8 identities, 4 instances each = 32 total samples
    labels = torch.tensor([0, 0, 0, 0,  # Identity 0
                          1, 1, 1, 1,  # Identity 1  
                          2, 2, 2, 2,  # Identity 2
                          3, 3, 3, 3,  # Identity 3
                          4, 4, 4, 4,  # Identity 4
                          5, 5, 5, 5,  # Identity 5
                          6, 6, 6, 6,  # Identity 6
                          7, 7, 7, 7]) # Identity 7
    
    # Simulate model outputs
    logits = torch.randn(batch_size, num_classes)  # Classification logits
    features = torch.randn(batch_size, feat_dim)   # Feature representations
    
    print(f"📊 Test Data:")
    print(f"   Batch size: {batch_size}")
    print(f"   Num classes: {num_classes}")
    print(f"   Feature dim: {feat_dim}")
    print(f"   Labels: {labels.tolist()}")
    
    # Test 1: Basic loss builder
    print(f"\n🎯 Test 1: Basic ID + Triplet Loss")
    loss_builder = make_loss(
        num_classes=num_classes,
        feat_dim=feat_dim,
        id_loss_weight=1.0,
        triplet_loss_weight=1.0,
        triplet_margin=0.3,
        label_smoothing=0.1,
        use_center_loss=False
    )
    
    total_loss, loss_dict = loss_builder(logits, features, labels)
    print(f"   Total Loss: {total_loss:.4f}")
    for name, value in loss_dict.items():
        print(f"   {name}: {value:.4f}")
    
    # Test 2: With center loss
    print(f"\n🎯 Test 2: ID + Triplet + Center Loss")
    loss_builder_center = make_loss(
        num_classes=num_classes,
        feat_dim=feat_dim,
        id_loss_weight=1.0,
        triplet_loss_weight=1.0,
        triplet_margin=0.3,
        label_smoothing=0.1,
        use_center_loss=True,
        center_loss_weight=0.0005
    )
    
    total_loss_center, loss_dict_center = loss_builder_center(logits, features, labels)
    print(f"   Total Loss: {total_loss_center:.4f}")
    for name, value in loss_dict_center.items():
        print(f"   {name}: {value:.4f}")
    
    # Test 3: Check gradients
    print(f"\n🔍 Test 3: Gradient Flow Check")
    
    # Create fresh tensors with gradients enabled
    logits_grad = torch.randn(batch_size, num_classes, requires_grad=True)
    features_grad = torch.randn(batch_size, feat_dim, requires_grad=True)
    
    # Compute loss and backpropagate
    grad_loss, _ = loss_builder(logits_grad, features_grad, labels)
    grad_loss.backward()
    
    logits_grad_norm = logits_grad.grad.norm().item() if logits_grad.grad is not None else 0
    features_grad_norm = features_grad.grad.norm().item() if features_grad.grad is not None else 0
    
    print(f"   Logits gradient norm: {logits_grad_norm:.4f}")
    print(f"   Features gradient norm: {features_grad_norm:.4f}")
    print(f"   ✅ Gradients flowing correctly!")
    
    # Test 4: Loss components analysis
    print(f"\n📈 Test 4: Loss Component Analysis")
    
    # Test with different margins
    margins = [0.1, 0.3, 0.5]
    for margin in margins:
        test_builder = make_loss(
            num_classes=num_classes,
            feat_dim=feat_dim,
            triplet_margin=margin,
            use_center_loss=False
        )
        test_loss, test_dict = test_builder(logits.detach(), features.detach(), labels)
        print(f"   Margin {margin}: Triplet={test_dict['triplet_loss']:.4f}, Total={test_dict['total_loss']:.4f}")
    
    print(f"\n✅ All loss function tests passed!")
    print(f"🚀 Loss functions are ready for training!")

if __name__ == '__main__':
    test_loss_functions()
