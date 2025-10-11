"""
Main model builder for Dog ReID.
Combines backbone + BN-neck + classifier in a clean, pluggable way.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .backbones import build_backbone
from .layers import BNNeck, ClassificationHead, FeatureEmbedding, MultiLevelFusion, l2_normalize, weights_init_kaiming

class DogReIDModel(nn.Module):
    """
    Dog Re-Identification Model.
    
    Architecture:
        backbone -> [embedding] -> BN-neck -> [classifier]
        
    Supports both training (with classifier) and inference (feature extraction).
    Inspired by your Tiger notebook + CLIP-ReID structure.
    """
    
    def __init__(
        self, 
        backbone_name: str = 'dinov3_vitl16',
        num_classes: int = 0,
        embed_dim: int = None,
        pretrained: bool = True,
        bn_neck: bool = True
    ):
        super().__init__()
        
        self.backbone_name = backbone_name
        self.num_classes = num_classes
        self.bn_neck = bn_neck
        
        # Build backbone
        self.backbone, backbone_feat_dim = build_backbone(backbone_name, pretrained)
        
        # Optional feature embedding (like in your Tiger notebook)
        if embed_dim is not None and embed_dim != backbone_feat_dim:
            # 🧪 RESEARCH: Use multi-level fusion for multi-level backbones
            if 'multilevel' in backbone_name:
                self.embedding = MultiLevelFusion(backbone_feat_dim, embed_dim)
                print(f"🧪 Multi-level fusion: {backbone_feat_dim} -> {embed_dim}")
            else:
                self.embedding = FeatureEmbedding(backbone_feat_dim, embed_dim)
                print(f"📊 Added embedding layer: {backbone_feat_dim} -> {embed_dim}")
            feat_dim = embed_dim
        else:
            self.embedding = None
            feat_dim = backbone_feat_dim
            
        self.feat_dim = feat_dim
        
        # BN-neck (critical for ReID performance)
        if bn_neck:
            self.bottleneck = BNNeck(feat_dim)
            self.bottleneck.apply(weights_init_kaiming)
            print(f"🔧 Added BN-neck (feat_dim: {feat_dim})")
        else:
            self.bottleneck = nn.Identity()
            
        # Classifier head (for training with ID loss)
        if num_classes > 0:
            self.classifier = ClassificationHead(feat_dim, num_classes)
            print(f"🎯 Added classifier head ({feat_dim} -> {num_classes} classes)")
        else:
            self.classifier = None
            
        print(f"✅ DogReIDModel built: {backbone_name} | feat_dim: {feat_dim} | classes: {num_classes}")
        
    def forward(self, x, return_mode='auto'):
        """
        Forward pass with flexible return modes.
        
        Args:
            x: Input images [B, 3, H, W]
            return_mode: 
                - 'auto': Return logits if training, features if eval
                - 'features': Always return features (for zero-shot/inference)
                - 'logits': Return logits and features (for training)
                - 'both': Return both logits and features
                
        Returns:
            Based on return_mode and training state
        """
        # Backbone feature extraction
        features = self.backbone(x)
        
        # Handle different backbone output formats
        if features.ndim > 2:  # Some backbones return spatial features
            features = torch.flatten(features, 1)
            
        # Optional embedding layer
        if self.embedding is not None:
            features = self.embedding(features)
            
        # BN-neck
        bn_features = self.bottleneck(features)
        
        # Determine what to return based on mode and training state
        if return_mode == 'features':
            # For inference/zero-shot - return L2 normalized features
            return l2_normalize(bn_features)
            
        elif return_mode == 'logits' and self.classifier is not None:
            # For training - return logits
            logits = self.classifier(bn_features)
            return logits
            
        elif return_mode == 'both' and self.classifier is not None:
            # Return both logits and features
            logits = self.classifier(bn_features) 
            return logits, features  # Return pre-BN features for triplet loss
            
        elif return_mode == 'auto':
            # Automatic mode based on training state
            if self.training and self.classifier is not None:
                # Training: return logits and pre-BN features (CLIP-ReID style)
                logits = self.classifier(bn_features)
                return logits, features  # logits for ID loss, features for triplet loss
            else:
                # Evaluation: return normalized features
                return l2_normalize(bn_features)
                
        else:
            # Default: just return features
            return l2_normalize(bn_features)
    
    def extract_features(self, x):
        """Convenience method for feature extraction."""
        return self.forward(x, return_mode='features')
    
    def get_feature_dim(self):
        """Get the output feature dimension."""
        return self.feat_dim
    
    def get_backbone_name(self):
        """Get the backbone name."""
        return self.backbone_name
    
    def freeze_backbone(self):
        """Freeze backbone parameters (useful for fine-tuning)."""
        for param in self.backbone.parameters():
            param.requires_grad = False
        print("🔒 Backbone frozen")
        
    def unfreeze_backbone(self):
        """Unfreeze backbone parameters."""
        for param in self.backbone.parameters():
            param.requires_grad = True
        print("🔓 Backbone unfrozen")

def make_model(backbone_name='dinov2_vitb14', num_classes=0, embed_dim=512, **kwargs):
    """
    Convenient factory function to create Dog ReID model.
    
    Args:
        backbone_name: Name of backbone architecture
        num_classes: Number of classes for classification (0 for feature-only)
        embed_dim: Optional embedding dimension (None to use backbone dim)
        **kwargs: Additional arguments passed to DogReIDModel
        
    Returns:
        DogReIDModel instance
    """
    return DogReIDModel(
        backbone_name=backbone_name,
        num_classes=num_classes, 
        embed_dim=embed_dim,
        **kwargs
    )
