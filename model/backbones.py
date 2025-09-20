"""
Pluggable backbone architectures for Dog ReID.
Supports DINOv2, with easy extension to ResNet, Swin, etc.
"""

import os
from pathlib import Path
from typing import Tuple

import torch
import torch.nn as nn
import timm

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
    elif name.startswith('swin'):
        return build_swin_backbone(name, pretrained)
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


def _load_hf_token() -> str:
    """Return the Hugging Face token from env, login cache, or local .env."""
    token = os.getenv("HF_TOKEN")
    if token:
        return token.strip()

    try:
        from huggingface_hub import get_token
    except ImportError:
        token = None
    else:
        token = get_token()
        if token:
            return token

    env_path = Path(__file__).resolve().parents[1] / ".env"
    if env_path.exists():
        for line in env_path.read_text().splitlines():
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            key, sep, value = stripped.partition("=")
            if key.strip() == "HF_TOKEN" and sep:
                return value.strip().strip('"').strip("'")
    return ""


def _from_pretrained_with_token(loader, hf_name: str, token: str):
    """Call a Hugging Face loader with the provided token if necessary."""
    if not token:
        return loader(hf_name)

    try:
        return loader(hf_name, token=token)
    except TypeError:
        return loader(hf_name, use_auth_token=token)


class DINOv3BackboneAdapter(nn.Module):
    """Thin wrapper to expose Hugging Face DINOv3 outputs as feature vectors."""

    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        outputs = self.model(pixel_values=x)

        # For ReID: Use CLS token (better for global image representation)
        # CLS token is specifically designed for classification/retrieval tasks
        if isinstance(outputs, tuple):
            last_hidden_state = outputs[0]
        else:
            last_hidden_state = getattr(outputs, "last_hidden_state", None)

        if last_hidden_state is None:
            raise RuntimeError("DINOv3 model did not return hidden states")

        return last_hidden_state[:, 0]  # CLS token [B, hidden_dim]


def build_dinov3_backbone(name: str, pretrained: bool = True) -> Tuple[nn.Module, int]:
    """
    Build a DINOv3 backbone using Hugging Face Transformers.

    Args:
        name: Either a canonical alias (``dinov3_vitl16``) or a Hugging Face repo id.
        pretrained: Load pretrained weights when True, otherwise initialize from config.
    """
    try:
        from transformers import AutoConfig, AutoModel
    except ImportError as exc:
        raise RuntimeError("DINOv3 backbones require the 'transformers' package") from exc

    official_aliases = {
        'dinov3_vits16': 'facebook/dinov3-vits16-pretrain-lvd1689m',
        'dinov3_vits16plus': 'facebook/dinov3-vits16plus-pretrain-lvd1689m',
        'dinov3_vitb16': 'facebook/dinov3-vitb16-pretrain-lvd1689m',
        'dinov3_vitl16': 'facebook/dinov3-vitl16-pretrain-lvd1689m',
        'dinov3_vith16plus': 'facebook/dinov3-vith16plus-pretrain-lvd1689m',
        'dinov3_vit7b16': 'facebook/dinov3-vit7b16-pretrain-lvd1689m',
    }

    hf_name = official_aliases.get(name, name)
    if '/' not in hf_name:
        valid = ", ".join(sorted(official_aliases))
        raise ValueError(f"Unknown DINOv3 variant '{name}'. Use one of: {valid}")

    printable = f"{name} ({hf_name})" if name != hf_name else hf_name
    print(f"🔧 Building DINOv3 backbone via Hugging Face: {printable}")

    hf_token = _load_hf_token()

    try:
        config = _from_pretrained_with_token(AutoConfig.from_pretrained, hf_name, hf_token)
    except Exception as exc:
        if not hf_token:
            raise RuntimeError(
                "Access to gated DINOv3 checkpoints requires an HF access token. "
                "Set HF_TOKEN in your environment or .env file."
            ) from exc
        raise

    if pretrained:
        model = _from_pretrained_with_token(AutoModel.from_pretrained, hf_name, hf_token)
    else:
        model = AutoModel.from_config(config)

    wrapped_model = DINOv3BackboneAdapter(model)

    # Determine feature dimensionality from config when possible.
    if hasattr(config, 'hidden_size'):
        feat_dim = config.hidden_size
    elif hasattr(config, 'embed_dim'):
        feat_dim = config.embed_dim
    elif hasattr(config, 'hidden_sizes') and config.hidden_sizes:
        feat_dim = config.hidden_sizes[-1]
    else:
        raise RuntimeError("Unable to infer feature dimension for DINOv3 model")

    print(f"🚀 DINOv3 model '{hf_name}' loaded - feature dim: {feat_dim}")
    return wrapped_model, feat_dim

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

def build_swin_backbone(name: str, pretrained: bool = True) -> Tuple[nn.Module, int]:
    """
    Build SWIN Transformer backbone using timm.
    Perfect for research - good performance but not as dominant as DINOv3-L!
    """
    print(f"🔧 Building SWIN backbone: {name}")
    
    try:
        import timm
    except ImportError:
        raise ImportError("Please install timm: pip install timm")
    
    # Map our names to timm model names
    swin_model_map = {
        'swin_tiny_patch4_window7_224': 'swin_tiny_patch4_window7_224',     # 🎯 Tiny - Perfect for research!
        'swin_small_patch4_window7_224': 'swin_small_patch4_window7_224',   # Small - Good middle ground
        'swin_base_patch4_window7_224': 'swin_base_patch4_window7_224',     # Base - Stronger
        'swin_large_patch4_window7_224': 'swin_large_patch4_window7_224',
        'swin_large_patch4_window12_384': 'swin_large_patch4_window12_384', 
    }
    
    timm_name = swin_model_map.get(name, name)
    
    # Create model without classification head
    # Note: SWIN models need specific input sizes (224x224)
    model = timm.create_model(
        timm_name, 
        pretrained=pretrained, 
        num_classes=0,  # Remove classification head
        global_pool='avg'  # Use global average pooling for flexible input sizes
    )
    
    # Get feature dimension
    if hasattr(model, 'num_features'):
        feat_dim = model.num_features
    elif hasattr(model, 'embed_dim'):
        feat_dim = model.embed_dim
    else:
        # Default SWIN-L feature dimension
        feat_dim = 1536
    
    print(f"🚀 SWIN loaded successfully - Feature dim: {feat_dim}")
    return model, feat_dim

# Registry for easy backbone access (optimized for server hardware)
BACKBONE_REGISTRY = {
    # DINOv3 variants (Hugging Face canonical names)
    'dinov3_vits16': 384,
    'dinov3_vits16plus': 384,  # hidden size resolved at runtime
    'dinov3_vitb16': 768,      # 🎯 Base - Good for research (less dominant than Large)
    'dinov3_vitl16': 1024,     # Large - Your current powerful choice
    'dinov3_vith16plus': 1280,
    'dinov3_vit7b16': 4096,

    # DINOv2 variants (your proven choice - still excellent!)
    'dinov2_vits14': 384,   # Small - for quick experiments
    'dinov2_vitb14': 768,   # Base - your original success
    'dinov2_vitl14': 1024,  # Large - recommended for your server! 🚀
    'dinov2_vitg14': 1536,  # Giant - if you want max power 💪
    
    # SWIN Transformers (great for research - balanced performance) 🎯
    'swin_tiny_patch4_window7_224': 768,      # 🎯 SWIN-Tiny - Perfect research baseline!
    'swin_small_patch4_window7_224': 768,     # SWIN-Small - Still reasonable
    'swin_base_patch4_window7_224': 1024,     # SWIN-B - Good research baseline
    'swin_large_patch4_window7_224': 1536,    # SWIN-L - Strong but not dominant
    'swin_large_patch4_window12_384': 1536,   # SWIN-L 384 - Higher resolution
    
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
