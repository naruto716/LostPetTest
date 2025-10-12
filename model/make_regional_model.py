"""
Regional model for Dog ReID with facial regions.
Follows the same structure as DogReIDModel but processes multiple regions.
"""

import torch
import torch.nn as nn

from .backbones import build_backbone
from .layers import BNNeck, ClassificationHead, l2_normalize, weights_init_kaiming
from .attention_fusion import AttentionFusion


class RegionalDogReIDModel(nn.Module):
    """
    Regional Dog Re-Identification Model.
    
    Architecture:
        shared_backbone (frozen) → process global + 7 regions
        → concatenate (8 × feat_dim)
        → fusion layer (project to embed_dim)
        → BN-neck
        → [classifier]
    
    Similar to DogReIDModel but takes multiple inputs.
    """
    
    def __init__(
        self,
        backbone_name: str = 'dinov3_vitl16',
        num_classes: int = 0,
        embed_dim: int = 768,
        pretrained: bool = True,
        bn_neck: bool = True,
        use_attention: bool = False  # NEW: Use attention fusion instead of concat
    ):
        super().__init__()
        
        self.backbone_name = backbone_name
        self.num_classes = num_classes
        self.embed_dim = embed_dim
        self.bn_neck = bn_neck
        self.use_attention = use_attention
        
        # Build shared backbone (will be frozen during training)
        self.backbone, backbone_feat_dim = build_backbone(backbone_name, pretrained)
        print(f"📊 Shared backbone for all regions: {backbone_name} (feat_dim: {backbone_feat_dim})")
        
        # Define regions
        self.regions = ['left_eye', 'right_eye', 'nose', 'mouth', 
                       'left_ear', 'right_ear', 'forehead']
        
        # Choose fusion strategy
        if use_attention:
            # Attention-based fusion: learns to weight regions by importance
            num_regions = 1 + len(self.regions)  # global + 7 regions = 8
            self.fusion = AttentionFusion(
                num_regions=num_regions,
                feat_dim=backbone_feat_dim,
                hidden_dim=256
            )
            self.feat_dim = backbone_feat_dim  # Attention fusion preserves dimension
        else:
            # Simple concatenation fusion (original approach)
            concat_dim = backbone_feat_dim * 8  # 1 global + 7 regions
            hidden_dim = max(embed_dim * 2, (concat_dim + embed_dim) // 2)
            
            self.fusion = nn.Sequential(
                nn.Linear(concat_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(0.1),
                nn.Linear(hidden_dim, embed_dim),
                nn.BatchNorm1d(embed_dim),
            )
            self._init_fusion_weights()
            print(f"🔗 Fusion layer: {concat_dim} → {hidden_dim} → {embed_dim}")
            self.feat_dim = embed_dim
        
        # BN-neck
        if bn_neck:
            self.bottleneck = BNNeck(self.feat_dim)
            self.bottleneck.apply(weights_init_kaiming)
            print(f"🔧 Added BN-neck (feat_dim: {self.feat_dim})")
        else:
            self.bottleneck = nn.Identity()
        
        # Classifier
        if num_classes > 0:
            self.classifier = ClassificationHead(self.feat_dim, num_classes)
            print(f"🎯 Added classifier head ({self.feat_dim} → {num_classes} classes)")
        else:
            self.classifier = None
        
        print(f"✅ RegionalDogReIDModel built: {backbone_name} | feat_dim: {embed_dim} | classes: {num_classes}")
    
    def _init_fusion_weights(self):
        """Initialize fusion layer weights."""
        for m in self.fusion.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, global_img, regions_dict, return_mode='auto'):
        """
        Forward pass with global image and regional images.
        
        Args:
            global_img: Global face images [B, 3, H, W]
            regions_dict: Dict of region_name -> regional images [B, 3, H, W]
            return_mode: Same as DogReIDModel
        
        Returns:
            Based on return_mode and training state
        """
        # Extract global features
        global_features = self.backbone(global_img)
        if global_features.ndim > 2:
            global_features = torch.flatten(global_features, 1)
        
        # Extract regional features (using same backbone)
        regional_features = []
        for region_name in self.regions:
            region_img = regions_dict[region_name]
            region_feat = self.backbone(region_img)
            if region_feat.ndim > 2:
                region_feat = torch.flatten(region_feat, 1)
            regional_features.append(region_feat)
        
        # Fuse features (different strategies)
        if self.use_attention:
            # Attention fusion: pass list of features [global, region1, region2, ...]
            all_features_list = [global_features] + regional_features
            fused_features, attention_weights = self.fusion(all_features_list)
            # Store attention weights for debugging/visualization (optional)
            self.last_attention_weights = attention_weights
        else:
            # Simple concatenation fusion
            all_features = torch.cat([global_features] + regional_features, dim=1)
            fused_features = self.fusion(all_features)
        
        # BN-neck
        bn_features = self.bottleneck(fused_features)
        
        # Return based on mode (same logic as DogReIDModel)
        if return_mode == 'features':
            return l2_normalize(bn_features)
        
        elif return_mode == 'logits' and self.classifier is not None:
            logits = self.classifier(bn_features)
            return logits
        
        elif return_mode == 'both' and self.classifier is not None:
            logits = self.classifier(bn_features)
            return logits, fused_features  # Return pre-BN features for triplet loss
        
        elif return_mode == 'auto':
            if self.training and self.classifier is not None:
                # Training: return logits and features
                logits = self.classifier(bn_features)
                return logits, fused_features
            else:
                # Evaluation: return normalized features
                return l2_normalize(bn_features)
        else:
            return l2_normalize(bn_features)
    
    def extract_features(self, global_img, regions_dict):
        """Convenience method for feature extraction."""
        return self.forward(global_img, regions_dict, return_mode='features')
    
    def get_feature_dim(self):
        """Get the output feature dimension."""
        return self.feat_dim
    
    def get_backbone_name(self):
        """Get the backbone name."""
        return self.backbone_name
    
    def freeze_backbone(self):
        """Freeze backbone parameters."""
        for param in self.backbone.parameters():
            param.requires_grad = False
        print("🔒 Backbone frozen")
    
    def unfreeze_backbone(self):
        """Unfreeze backbone parameters."""
        for param in self.backbone.parameters():
            param.requires_grad = True
        print("🔓 Backbone unfrozen")


def make_regional_model(backbone_name='dinov3_vitl16', num_classes=0, embed_dim=768, **kwargs):
    """
    Factory function to create regional Dog ReID model.
    
    Args:
        backbone_name: Backbone architecture (default: dinov3_vitl16 with 1024-dim)
        num_classes: Number of classes
        embed_dim: Final embedding dimension after fusion (default: 768)
        **kwargs: Additional arguments
    
    Returns:
        RegionalDogReIDModel instance
    """
    return RegionalDogReIDModel(
        backbone_name=backbone_name,
        num_classes=num_classes,
        embed_dim=embed_dim,
        **kwargs
    )

