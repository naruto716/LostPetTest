"""
ID Loss (Cross-Entropy with Label Smoothing) for Dog ReID.
Based on CLIP-ReID's proven implementation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class LabelSmoothingCrossEntropy(nn.Module):
    """
    Cross-entropy loss with label smoothing regularization.
    
    Reference:
    Szegedy et al. Rethinking the Inception Architecture for Computer Vision. CVPR 2016.
    
    This implementation follows CLIP-ReID's approach with modern PyTorch optimizations.
    """
    
    def __init__(self, smoothing=0.1):
        """
        Initialize label smoothing cross entropy loss.
        
        Args:
            smoothing: Label smoothing factor (0.0 = no smoothing, 0.1 = 10% smoothing)
        """
        super().__init__()
        assert 0.0 <= smoothing < 1.0, "Smoothing must be in [0.0, 1.0)"
        self.smoothing = smoothing
        self.confidence = 1.0 - smoothing
        
    def forward(self, logits, targets):
        """
        Forward pass.
        
        Args:
            logits: Prediction logits with shape (batch_size, num_classes)
            targets: Ground truth labels with shape (batch_size,)
        Returns:
            loss: Smoothed cross-entropy loss
        """
        log_probs = F.log_softmax(logits, dim=1)
        
        # Standard cross-entropy for correct class
        nll_loss = -log_probs.gather(dim=1, index=targets.unsqueeze(1)).squeeze(1)
        
        # Uniform distribution loss (smoothing term)
        smooth_loss = -log_probs.mean(dim=1)
        
        # Combine both losses
        loss = self.confidence * nll_loss + self.smoothing * smooth_loss
        
        return loss.mean()

class CrossEntropyLabelSmooth(nn.Module):
    """
    Alternative implementation following CLIP-ReID's exact approach.
    Kept for compatibility and comparison.
    """
    
    def __init__(self, num_classes, epsilon=0.1):
        super().__init__()
        self.num_classes = num_classes
        self.epsilon = epsilon
        self.logsoftmax = nn.LogSoftmax(dim=1)
        
    def forward(self, inputs, targets):
        """
        Args:
            inputs: Prediction matrix (before softmax) with shape (batch_size, num_classes)
            targets: Ground truth labels with shape (batch_size,)
        """
        log_probs = self.logsoftmax(inputs)
        
        # Create one-hot targets
        targets_one_hot = torch.zeros_like(log_probs)
        targets_one_hot.scatter_(1, targets.unsqueeze(1), 1)
        
        # Apply label smoothing
        targets_smooth = (1 - self.epsilon) * targets_one_hot + self.epsilon / self.num_classes
        
        # Compute loss
        loss = (-targets_smooth * log_probs).mean(0).sum()
        
        return loss
