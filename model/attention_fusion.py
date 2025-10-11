"""
Attention-based fusion for regional features.
Learns to weight regions by importance - downweights noisy regions automatically.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class AttentionFusion(nn.Module):
    """
    Attention-based fusion for global + regional features.
    
    Key idea: Learn attention weights for each region.
    - Good regions (clear landmarks) get high weight
    - Bad regions (poor landmarks) get low weight
    - Automatically handles noisy landmarks without manual filtering!
    
    Args:
        num_regions: Number of regional features (default: 8 for 1 global + 7 face regions)
        feat_dim: Feature dimension per region
        hidden_dim: Hidden dimension for attention network
    """
    
    def __init__(self, num_regions=8, feat_dim=1024, hidden_dim=256):
        super().__init__()
        self.num_regions = num_regions
        self.feat_dim = feat_dim
        
        # Attention network: learns importance of each region
        self.attention = nn.Sequential(
            nn.Linear(feat_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, 1)  # Output: importance score per region
        )
        
        # Feature transformation before fusion
        self.transform = nn.Sequential(
            nn.Linear(feat_dim, feat_dim),
            nn.BatchNorm1d(feat_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3)
        )
        
        print(f"✨ Attention Fusion: {num_regions} regions × {feat_dim}D → learned weighting")
    
    def forward(self, regional_features):
        """
        Args:
            regional_features: List of [B, feat_dim] tensors (one per region)
                              [global_feat, left_eye, right_eye, nose, ...]
        
        Returns:
            fused_features: [B, feat_dim] weighted combination
            attention_weights: [B, num_regions] for visualization/debugging
        """
        # Stack regions: [B, num_regions, feat_dim]
        stacked = torch.stack(regional_features, dim=1)  
        B, N, D = stacked.shape
        
        # Compute attention scores for each region
        # Reshape: [B, N, D] → [B*N, D]
        flat_features = stacked.view(B * N, D)
        
        # Get attention logits: [B*N, 1]
        attention_logits = self.attention(flat_features)
        
        # Reshape back and apply softmax: [B, N]
        attention_logits = attention_logits.view(B, N)
        attention_weights = F.softmax(attention_logits, dim=1)
        
        # Apply attention weights: [B, N, 1] × [B, N, D] → [B, N, D]
        weighted_features = attention_weights.unsqueeze(-1) * stacked
        
        # Sum weighted features: [B, N, D] → [B, D]
        fused = weighted_features.sum(dim=1)
        
        # Transform fused features
        output = self.transform(fused)
        
        return output, attention_weights


class ResidualRegionalFusion(nn.Module):
    """
    Alternative: Add regional features as residual to global.
    
    Key idea: Start with strong global features, add regional refinement.
    - If landmarks are good: regional refinement helps
    - If landmarks are bad: model learns to ignore them (small residual)
    
    This is safer than concatenation approach!
    """
    
    def __init__(self, feat_dim=1024, num_regional=7):
        super().__init__()
        
        # Process regional features
        regional_concat_dim = feat_dim * num_regional
        self.regional_proj = nn.Sequential(
            nn.Linear(regional_concat_dim, feat_dim),
            nn.BatchNorm1d(feat_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(feat_dim, feat_dim),
            nn.BatchNorm1d(feat_dim),
        )
        
        # Gating: learns how much to trust regional features
        self.gate = nn.Sequential(
            nn.Linear(feat_dim * 2, 128),  # global + regional_proj
            nn.ReLU(inplace=True),
            nn.Linear(128, 1),
            nn.Sigmoid()  # 0 = trust global only, 1 = trust regional fully
        )
        
        print(f"🔄 Residual Fusion: global + gated({num_regional} regions)")
    
    def forward(self, global_feat, regional_feats):
        """
        Args:
            global_feat: [B, feat_dim] strong baseline features
            regional_feats: List of [B, feat_dim] regional features (7 regions)
        
        Returns:
            output: [B, feat_dim] global + gated regional refinement
            gate_values: [B, 1] how much regional was trusted
        """
        # Concatenate regional features
        regional_concat = torch.cat(regional_feats, dim=1)  # [B, 7*feat_dim]
        
        # Project to same dimension as global
        regional_proj = self.regional_proj(regional_concat)  # [B, feat_dim]
        
        # Compute gate: how much to trust regional?
        gate_input = torch.cat([global_feat, regional_proj], dim=1)
        gate_value = self.gate(gate_input)  # [B, 1]
        
        # Combine: global + gated_regional
        output = global_feat + gate_value * regional_proj
        
        return output, gate_value


if __name__ == "__main__":
    # Test
    B, D = 32, 1024
    
    # Simulate global + 7 regional features
    features = [torch.randn(B, D) for _ in range(8)]
    
    # Test attention fusion
    print("\nTesting Attention Fusion:")
    attn_fusion = AttentionFusion(num_regions=8, feat_dim=D)
    fused, weights = attn_fusion(features)
    print(f"  Input: 8 regions × {D}D")
    print(f"  Output: {fused.shape}")
    print(f"  Attention weights: {weights.shape}, example: {weights[0]}")
    
    # Test residual fusion
    print("\nTesting Residual Fusion:")
    res_fusion = ResidualRegionalFusion(feat_dim=D, num_regional=7)
    global_feat = features[0]
    regional_feats = features[1:]
    output, gate = res_fusion(global_feat, regional_feats)
    print(f"  Global: {global_feat.shape}")
    print(f"  Regions: {len(regional_feats)} × {D}D")
    print(f"  Output: {output.shape}")
    print(f"  Gate: {gate.shape}, mean trust: {gate.mean().item():.3f}")

