"""
Center Loss for Dog ReID.
Based on CLIP-ReID's implementation with modern PyTorch optimizations.
"""

import torch
import torch.nn as nn

class CenterLoss(nn.Module):
    """
    Center Loss for deep feature learning.
    
    Reference:
    Wen et al. A Discriminative Feature Learning Approach for Deep Face Recognition. ECCV 2016.
    
    This pulls features of the same class closer to their respective class centers.
    """
    
    def __init__(self, num_classes, feat_dim):
        """
        Initialize center loss.
        
        Args:
            num_classes: Number of classes in the dataset
            feat_dim: Feature dimension
        """
        super().__init__()
        self.num_classes = num_classes
        self.feat_dim = feat_dim
        
        # Initialize class centers as learnable parameters
        self.centers = nn.Parameter(torch.randn(num_classes, feat_dim))
        nn.init.xavier_uniform_(self.centers)
        
    def forward(self, features, labels):
        """
        Forward pass.
        
        Args:
            features: Feature matrix with shape (batch_size, feat_dim)
            labels: Ground truth labels with shape (batch_size,)
        Returns:
            loss: Center loss value
        """
        batch_size = features.size(0)
        
        # Ensure centers are on the same device as features/labels
        if self.centers.device != features.device:
            self.centers.data = self.centers.data.to(features.device)
        
        # Get centers for current batch labels
        centers_batch = self.centers[labels]  # (batch_size, feat_dim)
        
        # Compute center loss: sum of squared distances to class centers
        loss = ((features - centers_batch) ** 2).sum(dim=1).mean()
        
        return loss
    
    def get_centers(self):
        """Get current class centers."""
        return self.centers.data.clone()
    
    def update_centers(self, features, labels, alpha=0.5):
        """
        Update centers using exponential moving average.
        This is an alternative to gradient-based updates.
        
        Args:
            features: Feature matrix with shape (batch_size, feat_dim)
            labels: Ground truth labels with shape (batch_size,)
            alpha: Update rate (0.0 = no update, 1.0 = replace completely)
        """
        with torch.no_grad():
            # Ensure centers are on the same device
            if self.centers.device != features.device:
                self.centers.data = self.centers.data.to(features.device)
            
            for label in labels.unique():
                # Get features for this class
                mask = (labels == label)
                class_features = features[mask]
                
                if len(class_features) > 0:
                    # Compute new center as mean of features
                    new_center = class_features.mean(dim=0)
                    
                    # Update with exponential moving average
                    self.centers.data[label] = (1 - alpha) * self.centers.data[label] + alpha * new_center
