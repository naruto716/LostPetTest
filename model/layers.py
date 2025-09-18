"""
Model layers and components for Dog ReID.
Following CLIP-ReID patterns with BN-neck, etc.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class BNNeck(nn.Module):
    """
    Batch Normalization Neck - key component in ReID models.
    
    The BN-neck normalizes features before classification, which helps with:
    - Feature distribution alignment
    - Better gradient flow
    - Improved generalization
    
    Based on CLIP-ReID implementation.
    """
    
    def __init__(self, feat_dim: int):
        super().__init__()
        self.bn = nn.BatchNorm1d(feat_dim)
        # Disable bias learning (common in ReID)
        self.bn.bias.requires_grad_(False)
        
    def forward(self, x):
        return self.bn(x)

class ClassificationHead(nn.Module):
    """
    Simple classification head for ID loss.
    """
    
    def __init__(self, feat_dim: int, num_classes: int):
        super().__init__()
        self.classifier = nn.Linear(feat_dim, num_classes, bias=False)
        self._init_weights()
        
    def _init_weights(self):
        """Initialize classifier weights (following CLIP-ReID)."""
        nn.init.normal_(self.classifier.weight, std=0.001)
        
    def forward(self, x):
        return self.classifier(x)

class FeatureEmbedding(nn.Module):
    """
    Optional feature embedding layer to project backbone features
    to a different dimension (like in your Tiger notebook).
    """
    
    def __init__(self, in_dim: int, embed_dim: int):
        super().__init__()
        self.embedding = nn.Linear(in_dim, embed_dim)
        self._init_weights()
        
    def _init_weights(self):
        """Initialize embedding weights."""
        nn.init.kaiming_normal_(self.embedding.weight, mode='fan_out')
        nn.init.constant_(self.embedding.bias, 0)
        
    def forward(self, x):
        return self.embedding(x)

def l2_normalize(x, dim=1):
    """L2 normalize features along specified dimension."""
    return F.normalize(x, p=2, dim=dim)

def weights_init_kaiming(m):
    """Kaiming initialization for conv/linear layers."""
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_out')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    elif classname.find('Conv') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    elif classname.find('BatchNorm') != -1:
        if m.affine:
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0.0)

def weights_init_classifier(m):
    """Classifier-specific initialization."""
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.normal_(m.weight, std=0.001)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
