"""
Test the new research-friendly backbones: SWIN-L and DINOv3-B.
These should give you more room for improvement compared to DINOv3-L.
"""

import torch
from model import make_model
from config_training import cfg

def test_research_backbones():
    """Test the new backbones designed for research (less dominant than DINOv3-L)"""
    
    print("🔬 Testing Research-Friendly Backbones")
    print("=" * 60)
    print("Goal: Find backbones with room for improvement/novelty")
    print()
    
    # Test backbones in order of expected performance (low to high)
    research_backbones = [
        ('resnet50', 'Classic CNN baseline'),
        ('dinov2_vitb14', 'DINOv2 Base - good foundation'),
        ('dinov3_vitb16', '🎯 DINOv3 Base - promising for research'),
        ('swin_base_patch4_window7_224', '🎯 SWIN-B - hierarchical attention'),
        ('swin_large_patch4_window7_224', '🎯 SWIN-L - strong but not dominant'),
        ('dinov2_vitl14', 'DINOv2 Large - your previous success'),
        ('dinov3_vitl16', '🏆 DINOv3 Large - your 100% performance'),
    ]
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    print()
    
    results = []
    
    for backbone_name, description in research_backbones:
        print(f"📊 Testing: {backbone_name}")
        print(f"   Description: {description}")
        
        try:
            # Create model
            model = make_model(
                backbone_name=backbone_name,
                num_classes=95,
                embed_dim=cfg.EMBED_DIM,
                pretrained=True,
                bn_neck=True
            )
            
            model.to(device)
            model.eval()
            
            # Test forward pass with appropriate input size for each backbone
            batch_size = 8
            
            # SWIN models need 224x224, others can handle 336x336
            if backbone_name.startswith('swin'):
                input_size = 224
                print(f"   Using SWIN native size: {input_size}×{input_size}")
            else:
                input_size = 336
            
            dummy_input = torch.randn(batch_size, 3, input_size, input_size).to(device)
            
            with torch.no_grad():
                # Test feature extraction
                features = model(dummy_input, 'features')
                
                # Test training mode
                model.train()
                logits, train_features = model(dummy_input, 'auto')
            
            # Memory usage
            memory_mb = torch.cuda.max_memory_allocated() / (1024**2) if torch.cuda.is_available() else 0
            
            print(f"   ✅ Success! Features: {features.shape}, Memory: {memory_mb:.1f}MB")
            
            results.append({
                'backbone': backbone_name,
                'description': description,
                'feature_dim': features.shape[1],
                'memory_mb': memory_mb,
                'status': 'success'
            })
            
            # Clear memory
            del model
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
            
        except Exception as e:
            print(f"   ❌ Failed: {str(e)}")
            results.append({
                'backbone': backbone_name,
                'description': description,
                'status': 'failed',
                'error': str(e)
            })
        
        print()
    
    # Summary
    print("📈 Research Backbone Summary")
    print("=" * 60)
    
    successful_backbones = [r for r in results if r['status'] == 'success']
    
    for result in successful_backbones:
        marker = "🎯" if "🎯" in result['description'] else "  "
        print(f"{marker} {result['backbone']:30s} | {result['feature_dim']:4d}D | {result['memory_mb']:6.1f}MB")
    
    print()
    print("🎯 Recommended for Research:")
    print("   1. dinov3_vitb16 - Good performance but not perfect (flexible input size)")
    print("   2. swin_large_patch4_window7_224 - Different architecture paradigm (224×224 only)")
    print("   3. swin_base_patch4_window7_224 - Even more room for improvement (224×224 only)")
    print()
    print("📝 Note: SWIN models require 224×224 input due to window-based attention")
    print()
    print("🏆 Avoid for research (too dominant):")
    print("   - dinov3_vitl16 (your current 100% performance)")
    print()
    print("💡 Research Strategy:")
    print("   1. Start with SWIN-L or DINOv3-B")
    print("   2. Expect 70-90% performance (room for improvement)")
    print("   3. Develop novel techniques to bridge the gap")
    print("   4. Compare against DINOv3-L as upper bound")

if __name__ == "__main__":
    test_research_backbones()
