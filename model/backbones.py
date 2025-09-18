"""
Pluggable backbone architectures for Dog ReID.
Supports DINOv2, with easy extension to ResNet, Swin, etc.
"""

import torch
import torch.nn as nn
import timm
from typing import Tuple

def build_backbone(name: str, pretrained: bool = True) -> Tuple[nn.Module, int]:
    """
    Factory function to build different backbone architectures.
    
    Args:
        name: Backbone name (e.g., 'dinov2_vitb14', 'resnet50', 'vit_base_patch16_224')
        pretrained: Whether to use pretrained weights
        
    Returns:
        backbone: The backbone network
        feat_dim: Output feature dimension
    """
    if name.startswith('dinov3'):
        return build_dinov3_backbone(name, pretrained)
    elif name.startswith('dinov2'):
        return build_dinov2_backbone(name, pretrained)
    elif name.startswith('resnet'):
        return build_resnet_backbone(name, pretrained)
    elif name.startswith('vit'):
        return build_vit_backbone(name, pretrained)
    else:
        raise ValueError(f"Unsupported backbone: {name}")

def build_dinov2_backbone(name: str, pretrained: bool = True) -> Tuple[nn.Module, int]:
    """
    Build DINOv2 backbone from torch.hub.
    Based on your successful Tiger ReID implementation.
    
    Optimized for high-end server hardware (4x A10G, 96GB VRAM).
    No fallbacks - let failures be explicit for proper debugging.
    """
    print(f"🔧 Building DINOv2 backbone: {name}")
    
    # Load directly from torch.hub - no try/catch fallback
    model = torch.hub.load('facebookresearch/dinov2', name, pretrained=pretrained, trust_repo=True)
    
    # Get feature dimension with expanded model support
    if hasattr(model, 'embed_dim'):
        feat_dim = model.embed_dim
    elif hasattr(model, 'num_features'):
        feat_dim = model.num_features
    else:
        # Extended DINOv2 variant dimensions
        if 'vits14' in name:
            feat_dim = 384   # Small
        elif 'vitb14' in name:
            feat_dim = 768   # Base
        elif 'vitl14' in name:
            feat_dim = 1024  # Large (recommended for your server!)
        elif 'vitg14' in name:
            feat_dim = 1536  # Giant (if you're feeling bold)
        else:
            feat_dim = 768   # Default to Base
            
    print(f"🚀 DINOv2 loaded successfully - Feature dim: {feat_dim} (Server-optimized!)")
    return model, feat_dim

def build_dinov3_backbone(name: str, pretrained: bool = True) -> Tuple[nn.Module, int]:
    """
    Build DINOv3 backbone - the latest and greatest from Facebook!
    
    DINOv3 improvements:
    - 100x larger training dataset (1.7B images)
    - Up to 7B parameters 
    - Enhanced training strategies
    - State-of-the-art performance across CV tasks
    
    Released August 2025 - cutting edge!
    """
    print(f"🔧 Building DINOv3 backbone: {name} (CUTTING EDGE!)")
    
    # Try torch.hub first (most convenient)
    try:
        model = torch.hub.load('facebookresearch/dinov3', name, pretrained=pretrained, trust_repo=True)
        print(f"✅ DINOv3 loaded via torch.hub")
    except Exception as e:
        print(f"⚠️  torch.hub failed: {e}")
        print(f"🔄 Trying alternative loading method...")
        
        # Alternative: Load from HuggingFace transformers
        try:
            from transformers import AutoModel
            hf_model_map = {
                'dinov3_small': 'facebook/dinov3-small',
                'dinov3_base': 'facebook/dinov3-base', 
                'dinov3_large': 'facebook/dinov3-large',
                'dinov3_giant': 'facebook/dinov3-giant'
            }
            
            if name in hf_model_map:
                model = AutoModel.from_pretrained(hf_model_map[name])
                print(f"✅ DINOv3 loaded via HuggingFace transformers")
            else:
                raise ValueError(f"Unknown DINOv3 variant: {name}")
                
        except ImportError:
            raise RuntimeError("DINOv3 requires 'transformers' library. Install with: pip install transformers")
        except Exception as e2:
            raise RuntimeError(f"Failed to load DINOv3: torch.hub failed ({e}), transformers failed ({e2})")
    
    # Get feature dimensions for DINOv3 variants
    if hasattr(model, 'embed_dim'):
        feat_dim = model.embed_dim
    elif hasattr(model, 'config') and hasattr(model.config, 'hidden_size'):
        feat_dim = model.config.hidden_size  # HuggingFace format
    else:
        # DINOv3 typical dimensions (estimated based on model size)
        if 'small' in name:
            feat_dim = 384      # Small
        elif 'base' in name:
            feat_dim = 768      # Base  
        elif 'large' in name:
            feat_dim = 1024     # Large
        elif 'giant' in name:
            feat_dim = 1536     # Giant (7B params!)
        else:
            feat_dim = 1024     # Default to Large
            
    print(f"🚀 DINOv3 loaded successfully - Feature dim: {feat_dim} (NEXT-GEN!)")
    print(f"   📊 Model trained on 1.7B images (100x more than DINOv2)")
    print(f"   💪 Enhanced training strategies & data augmentation")
    
    return model, feat_dim

def build_resnet_backbone(name: str, pretrained: bool = True) -> Tuple[nn.Module, int]:
    """Build ResNet backbone using timm."""
    print(f"🔧 Building ResNet backbone: {name}")
    
    model = timm.create_model(name, pretrained=pretrained, num_classes=0, global_pool='avg')
    feat_dim = model.num_features
    
    print(f"✅ ResNet loaded - Feature dim: {feat_dim}")
    return model, feat_dim

def build_vit_backbone(name: str, pretrained: bool = True) -> Tuple[nn.Module, int]:
    """Build Vision Transformer backbone using timm."""
    print(f"🔧 Building ViT backbone: {name}")
    
    model = timm.create_model(name, pretrained=pretrained, num_classes=0)
    feat_dim = model.num_features if hasattr(model, 'num_features') else model.embed_dim
    
    print(f"✅ ViT loaded - Feature dim: {feat_dim}")
    return model, feat_dim

# Registry for easy backbone access (optimized for server hardware)
BACKBONE_REGISTRY = {
    # DINOv3 variants (CUTTING EDGE - Aug 2025!) 🔥
    'dinov3_small': 384,    # Small - 100x more training data than DINOv2!
    'dinov3_base': 768,     # Base - enhanced training strategies  
    'dinov3_large': 1024,   # Large - perfect for your server 🚀
    'dinov3_giant': 1536,   # Giant - 7B parameters! 💪
    
    # DINOv2 variants (your proven choice - still excellent!)
    'dinov2_vits14': 384,   # Small - for quick experiments
    'dinov2_vitb14': 768,   # Base - your original success
    'dinov2_vitl14': 1024,  # Large - recommended for your server! 🚀
    'dinov2_vitg14': 1536,  # Giant - if you want max power 💪
    
    # Standard backbones (for comparison)
    'resnet50': 2048,
    'resnet101': 2048,
    'vit_base_patch16_224': 768,
    'vit_large_patch16_224': 1024,
    'vit_huge_patch14_224': 1280,  # Added for completeness
}

def list_available_backbones():
    """List all available backbone architectures."""
    print("Available backbones:")
    for name, feat_dim in BACKBONE_REGISTRY.items():
        print(f"  - {name:25s} (feat_dim: {feat_dim})")
    return list(BACKBONE_REGISTRY.keys())
