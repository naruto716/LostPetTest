"""
Batch Hard Triplet Loss for Dog ReID.
Based on CLIP-ReID's proven implementation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

def euclidean_distance(x, y):
    """
    Compute euclidean distance between two tensors.
    
    Args:
        x: Tensor with shape [m, d]
        y: Tensor with shape [n, d]
    Returns:
        dist: Tensor with shape [m, n]
    """
    m, n = x.size(0), y.size(0)
    xx = torch.pow(x, 2).sum(1, keepdim=True).expand(m, n)
    yy = torch.pow(y, 2).sum(1, keepdim=True).expand(n, m).t()
    dist = xx + yy - 2 * torch.matmul(x, y.t())
    dist = dist.clamp(min=1e-12).sqrt()  # For numerical stability
    return dist

def hard_example_mining(dist_mat, labels):
    """
    For each anchor, find the hardest positive and negative sample.
    
    Args:
        dist_mat: Distance matrix with shape [N, N]
        labels: Labels with shape [N]
    Returns:
        dist_ap: Distance(anchor, positive) with shape [N]
        dist_an: Distance(anchor, negative) with shape [N]
    """
    assert len(dist_mat.size()) == 2
    assert dist_mat.size(0) == dist_mat.size(1)
    N = dist_mat.size(0)

    # Create masks for positive and negative pairs
    is_pos = labels.expand(N, N).eq(labels.expand(N, N).t())
    is_neg = labels.expand(N, N).ne(labels.expand(N, N).t())

    # Hardest positive: maximum distance among positive pairs
    dist_ap, _ = torch.max(
        dist_mat[is_pos].contiguous().view(N, -1), 1, keepdim=True)
    
    # Hardest negative: minimum distance among negative pairs  
    dist_an, _ = torch.min(
        dist_mat[is_neg].contiguous().view(N, -1), 1, keepdim=True)
    
    dist_ap = dist_ap.squeeze(1)
    dist_an = dist_an.squeeze(1)
    
    return dist_ap, dist_an

class BatchHardTripletLoss(nn.Module):
    """
    Batch Hard Triplet Loss.
    
    Reference:
    Hermans et al. In Defense of the Triplet Loss for Person Re-Identification. arXiv:1703.07737.
    
    Imported from CLIP-ReID with improvements.
    """
    
    def __init__(self, margin=0.3, normalize_feature=False):
        super().__init__()
        self.margin = margin
        self.normalize_feature = normalize_feature
        self.ranking_loss = nn.MarginRankingLoss(margin=margin)
        
    def forward(self, features, labels):
        """
        Args:
            features: Feature matrix with shape (batch_size, feat_dim)
            labels: Ground truth labels with shape (batch_size,)
        Returns:
            loss: Triplet loss value
        """
        if self.normalize_feature:
            features = F.normalize(features, p=2, dim=1)
            
        # Compute pairwise distance matrix
        dist_mat = euclidean_distance(features, features)
        
        # Hard example mining
        dist_ap, dist_an = hard_example_mining(dist_mat, labels)
        
        # Triplet ranking loss
        # We want: dist_an > dist_ap + margin
        # So we minimize: max(0, dist_ap - dist_an + margin)
        y = torch.ones_like(dist_an)
        loss = self.ranking_loss(dist_an, dist_ap, y)
        
        return loss
