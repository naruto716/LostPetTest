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
    """Return the Hugging Face token from env or a local .env file."""
    token = os.getenv("HF_TOKEN")
    if token:
        return token.strip()

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

        # Prefer pooled representation when available.
        pooler_output = getattr(outputs, "pooler_output", None)
        if pooler_output is not None:
            return pooler_output

        # Fallback to CLS token / first position of hidden states.
        if isinstance(outputs, tuple):
            last_hidden_state = outputs[0]
        else:
            last_hidden_state = getattr(outputs, "last_hidden_state", None)

        if last_hidden_state is None:
            raise RuntimeError("DINOv3 model did not return hidden states")

        return last_hidden_state[:, 0]


def build_dinov3_backbone(name: str, pretrained: bool = True) -> Tuple[nn.Module, int]:
    """
    Build a DINOv3 backbone using Hugging Face Transformers.

    Args:
        name: Either a short alias (e.g. ``dinov3_large``) or a Hugging Face model id.
        pretrained: Load pretrained weights when True, otherwise initialize from config.
    """
    print(f"🔧 Building DINOv3 backbone via Hugging Face: {name}")

    try:
        from transformers import AutoConfig, AutoModel
    except ImportError as exc:
        raise RuntimeError("DINOv3 backbones require the 'transformers' package") from exc

    alias_map = {
        'dinov3_small': 'facebook/dinov3-vits16-pretrain-lvd1689m',
        'dinov3_base': 'facebook/dinov3-vitb16-pretrain-lvd1689m',
        'dinov3_large': 'facebook/dinov3-vitl16-pretrain-lvd1689m',
        'dinov3_giant': 'facebook/dinov3-vit7b16-pretrain-lvd1689m',
    }

    hf_name = alias_map.get(name, name)
    if '/' not in hf_name:
        raise ValueError(
            f"Unknown DINOv3 variant '{name}'. Provide a known alias {list(alias_map.keys())} "
            "or a Hugging Face repo id like 'facebook/dinov3-vits16-pretrain-lvd1689m'."
        )

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
