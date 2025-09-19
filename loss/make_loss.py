"""
Main loss builder for Dog ReID.
Combines ID loss, triplet loss, and optional center loss.
Based on CLIP-ReID's proven architecture.
"""

import torch
import torch.nn as nn
from .triplet import BatchHardTripletLoss
from .center import CenterLoss
from .id_loss import LabelSmoothingCrossEntropy

class ReIDLossBuilder:
    """
    Main loss builder that combines multiple loss functions for ReID.
    
    Following CLIP-ReID's proven formula:
    Total Loss = λ_id × ID_Loss + λ_tri × Triplet_Loss + λ_center × Center_Loss
    """
    
    def __init__(self, 
                 num_classes,
                 feat_dim,
                 id_loss_weight=1.0,
                 triplet_loss_weight=1.0,
                 triplet_margin=0.3,
                 label_smoothing=0.1,
                 use_center_loss=False,
                 center_loss_weight=0.0005):
        """
        Initialize ReID loss builder.
        
        Args:
            num_classes: Number of identity classes
            feat_dim: Feature dimension for center loss
            id_loss_weight: Weight for ID classification loss
            triplet_loss_weight: Weight for triplet loss  
            triplet_margin: Margin for triplet loss
            label_smoothing: Label smoothing factor for ID loss
            use_center_loss: Whether to use center loss
            center_loss_weight: Weight for center loss
        """
        self.num_classes = num_classes
        self.feat_dim = feat_dim
        
        # Loss weights
        self.id_loss_weight = id_loss_weight
        self.triplet_loss_weight = triplet_loss_weight
        self.center_loss_weight = center_loss_weight
        self.use_center_loss = use_center_loss
        
        # Initialize loss functions
        self.id_criterion = LabelSmoothingCrossEntropy(smoothing=label_smoothing)
        self.triplet_criterion = BatchHardTripletLoss(
            margin=triplet_margin, 
            normalize_feature=False  # We handle normalization in model
        )
        
        if use_center_loss:
            self.center_criterion = CenterLoss(num_classes, feat_dim)
        else:
            self.center_criterion = None
            
        print(f"🎯 ReID Loss Configuration:")
        print(f"   ID Loss Weight: {id_loss_weight}")
        print(f"   Triplet Loss Weight: {triplet_loss_weight} (margin: {triplet_margin})")
        print(f"   Label Smoothing: {label_smoothing}")
        if use_center_loss:
            print(f"   Center Loss Weight: {center_loss_weight}")
        else:
            print(f"   Center Loss: Disabled")
    
    def __call__(self, logits, features, labels):
        """
        Compute combined ReID loss.
        
        Args:
            logits: Classification logits with shape (batch_size, num_classes)
            features: Feature representations with shape (batch_size, feat_dim)  
            labels: Ground truth labels with shape (batch_size,)
            
        Returns:
            total_loss: Combined loss value
            loss_dict: Dictionary with individual loss components
        """
        # ID Loss (Classification)
        id_loss = self.id_criterion(logits, labels)
        
        # Triplet Loss (Metric Learning)
        triplet_loss = self.triplet_criterion(features, labels)
        
        # Combine losses
        total_loss = (self.id_loss_weight * id_loss + 
                     self.triplet_loss_weight * triplet_loss)
        
        # Loss components for logging
        loss_dict = {
            'id_loss': id_loss.item(),
            'triplet_loss': triplet_loss.item(),
            'total_loss': total_loss.item()
        }
        
        # Optional center loss
        if self.use_center_loss and self.center_criterion is not None:
            center_loss = self.center_criterion(features, labels)
            total_loss += self.center_loss_weight * center_loss
            loss_dict['center_loss'] = center_loss.item()
            loss_dict['total_loss'] = total_loss.item()  # Update total
        
        return total_loss, loss_dict
    
    def get_center_params(self):
        """Get center loss parameters for separate optimization."""
        if self.center_criterion is not None:
            return list(self.center_criterion.parameters())
        return []

# Convenience factory function
def make_loss(num_classes, 
              feat_dim=1024,
              id_loss_weight=1.0,
              triplet_loss_weight=1.0,
              triplet_margin=0.3,
              label_smoothing=0.1,
              use_center_loss=False,
              center_loss_weight=0.0005):
    """
    Factory function to create ReID loss builder.
    
    Args:
        num_classes: Number of identity classes
        feat_dim: Feature dimension  
        id_loss_weight: Weight for ID loss
        triplet_loss_weight: Weight for triplet loss
        triplet_margin: Margin for triplet loss
        label_smoothing: Label smoothing factor
        use_center_loss: Whether to use center loss
        center_loss_weight: Weight for center loss
        
    Returns:
        ReIDLossBuilder instance
    """
    return ReIDLossBuilder(
        num_classes=num_classes,
        feat_dim=feat_dim,
        id_loss_weight=id_loss_weight,
        triplet_loss_weight=triplet_loss_weight,
        triplet_margin=triplet_margin,
        label_smoothing=label_smoothing,
        use_center_loss=use_center_loss,
        center_loss_weight=center_loss_weight
    )
