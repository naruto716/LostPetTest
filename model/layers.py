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
    Classification head for ID loss.
    Uses small weight initialization following CLIP-ReID.
    """
    
    def __init__(self, feat_dim: int, num_classes: int):
        super().__init__()
        self.classifier = nn.Linear(feat_dim, num_classes, bias=False)
        # Initialize with small weights (CLIP-ReID style)
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


class MultiLevelFusion(nn.Module):
    """
    🧪 RESEARCH: Multi-layer fusion for multi-level features
    Inspired by Amur Tiger ReID - uses 2+ FC layers after concatenation!
    """
    
    def __init__(self, in_dim: int, embed_dim: int, hidden_dim: int = None, dropout: float = 0.1):
        super().__init__()
        
        # Default hidden dimension (between input and output)
        if hidden_dim is None:
            hidden_dim = max(embed_dim * 2, (in_dim + embed_dim) // 2)
        
        self.fusion = nn.Sequential(
            # Layer 1: High-dim → Hidden (learn which features matter)
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            
            # Layer 2: Hidden → Final (learn optimal combinations)
            nn.Linear(hidden_dim, embed_dim),
            nn.BatchNorm1d(embed_dim),
        )
        
        self._init_weights()
        print(f"🧪 Multi-level fusion: {in_dim} → {hidden_dim} → {embed_dim}")
        
    def _init_weights(self):
        """Initialize fusion weights."""
        for m in self.fusion.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
                    
    def forward(self, x):
        return self.fusion(x)


class AdaptiveMultiLevelFusion(nn.Module):
    """
    🔬 ADVANCED RESEARCH: Customizable multi-level fusion
    Allows different architectures for experimentation
    """
    
    def __init__(self, in_dim: int, embed_dim: int, architecture: str = 'tiger', dropout: float = 0.1):
        super().__init__()
        
        self.architecture = architecture
        
        if architecture == 'tiger':
            # Tiger paper style: gradual reduction
            hidden_dim = (in_dim + embed_dim) // 2
            self.fusion = nn.Sequential(
                nn.Linear(in_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, embed_dim),
                nn.BatchNorm1d(embed_dim),
            )
            
        elif architecture == 'bottleneck':
            # Bottleneck: compress then expand then compress
            bottleneck_dim = min(embed_dim, in_dim // 4)
            self.fusion = nn.Sequential(
                nn.Linear(in_dim, bottleneck_dim),
                nn.BatchNorm1d(bottleneck_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
                nn.Linear(bottleneck_dim, embed_dim * 2),
                nn.BatchNorm1d(embed_dim * 2),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
                nn.Linear(embed_dim * 2, embed_dim),
                nn.BatchNorm1d(embed_dim),
            )
            
        elif architecture == 'deep':
            # Deep fusion: 3 layers for complex learning
            h1 = in_dim * 3 // 4
            h2 = (h1 + embed_dim) // 2
            self.fusion = nn.Sequential(
                nn.Linear(in_dim, h1),
                nn.BatchNorm1d(h1),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
                nn.Linear(h1, h2),
                nn.BatchNorm1d(h2),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
                nn.Linear(h2, embed_dim),
                nn.BatchNorm1d(embed_dim),
            )
            
        else:
            raise ValueError(f"Unknown architecture: {architecture}")
        
        self._init_weights()
        self._print_architecture(in_dim, embed_dim)
        
    def _print_architecture(self, in_dim, embed_dim):
        """Print the fusion architecture."""
        print(f"🧪 {self.architecture.upper()} fusion: {in_dim} → ... → {embed_dim}")
        for i, layer in enumerate(self.fusion):
            if isinstance(layer, nn.Linear):
                print(f"   Linear {i//4 + 1}: {layer.in_features} → {layer.out_features}")
    
    def _init_weights(self):
        """Initialize fusion weights."""
        for m in self.fusion.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
                    
    def forward(self, x):
        return self.fusion(x)


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