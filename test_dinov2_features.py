#!/usr/bin/env python3
"""
Quick test to verify DINOv2 features are meaningful.
Tests feature similarity between actual dog images from your dataset.
"""

import torch
import os
from PIL import Image
from config import cfg
from model import make_model
from datasets.make_dataloader_dogreid import build_transforms

def test_feature_quality():
    """Test that DINOv2 extracts meaningful features from real dog images"""
    
    print("🔍 Testing DINOv2 Feature Quality on Real Dog Images")
    print("=" * 60)
    
    # Create model
    model = make_model(
        backbone_name=cfg.BACKBONE,
        num_classes=0,  # Zero-shot mode
        embed_dim=cfg.EMBED_DIM
    )
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()
    
    # Load transforms
    test_transforms = build_transforms(cfg, is_train=False)
    
    # Get some real images from different dogs
    image_dir = os.path.join(cfg.ROOT_DIR, cfg.IMAGES_DIR, "query")
    image_files = [f for f in os.listdir(image_dir) if f.endswith('.jpg')][:6]  # First 6 images
    
    print(f"📸 Testing with {len(image_files)} real dog images:")
    for img_file in image_files:
        print(f"   - {img_file}")
    
    # Load and process images
    images = []
    for img_file in image_files:
        img_path = os.path.join(image_dir, img_file)
        img = Image.open(img_path).convert('RGB')
        img_tensor = test_transforms(img).unsqueeze(0)  # Add batch dim
        images.append(img_tensor)
    
    # Stack into batch
    batch = torch.cat(images, dim=0).to(device)  # [6, 3, 336, 336]
    
    print(f"\n🚀 Extracting features from batch: {batch.shape}")
    
    # Extract features
    with torch.no_grad():
        features = model(batch, return_mode='features')  # [6, 768], L2 normalized
    
    print(f"✅ Features extracted: {features.shape}")
    print(f"   Feature norms: {features.norm(dim=1)}")  # Should all be 1.0
    
    # Compute similarity matrix
    similarity_matrix = features @ features.T  # [6, 6] cosine similarity
    
    print(f"\n📊 Similarity Matrix (Cosine Similarity):")
    print("     ", " ".join([f"{i:6d}" for i in range(len(image_files))]))
    for i, img_file in enumerate(image_files):
        similarities = similarity_matrix[i].cpu().numpy()
        print(f"{i}: {img_file[:12]:12s}", " ".join([f"{s:6.3f}" for s in similarities]))
    
    # Analyze results
    print(f"\n🎯 Analysis:")
    
    # Check diagonal (self-similarity should be 1.0)
    diag_similarities = torch.diag(similarity_matrix).cpu().numpy()
    print(f"   Self-similarities (should be 1.0): {diag_similarities}")
    
    # Check off-diagonal range
    mask = ~torch.eye(len(image_files), dtype=bool)
    off_diag = similarity_matrix[mask].cpu().numpy()
    print(f"   Cross-similarities range: [{off_diag.min():.3f}, {off_diag.max():.3f}]")
    print(f"   Cross-similarities mean: {off_diag.mean():.3f}")
    
    # Quality indicators
    print(f"\n✅ Quality Indicators:")
    if all(abs(s - 1.0) < 0.001 for s in diag_similarities):
        print("   ✅ Perfect self-similarity (DINOv2 features normalized correctly)")
    else:
        print("   ⚠️  Self-similarity not perfect (normalization issue)")
        
    if 0.5 < off_diag.mean() < 0.95:
        print("   ✅ Reasonable cross-similarities (features are meaningful)")
    elif off_diag.mean() > 0.95:
        print("   ⚠️  Very high similarities (features might be too generic)")
    else:
        print("   ⚠️  Very low similarities (features might be noisy)")
    
    # Find most similar pair
    similarity_no_diag = similarity_matrix.clone()
    similarity_no_diag.fill_diagonal_(-1)  # Ignore self-similarity
    max_sim_idx = similarity_no_diag.argmax()
    i, j = max_sim_idx // len(image_files), max_sim_idx % len(image_files)
    max_sim = similarity_matrix[i, j].item()
    
    print(f"   Most similar pair: {image_files[i]} ↔ {image_files[j]} (sim: {max_sim:.3f})")
    
    # Check if they're the same dog (parse filename for dog ID)
    def get_dog_id(filename):
        # Assuming filename format: dogXXX_xxx.jpg
        try:
            return filename.split('_')[0]
        except:
            return filename[:6]  # Fallback
    
    id_i, id_j = get_dog_id(image_files[i]), get_dog_id(image_files[j])
    if id_i == id_j:
        print("   ✅ Highest similarity is between same dog ID - features working great!")
    else:
        print(f"   📋 Highest similarity between different dogs ({id_i} vs {id_j}) - may be similar poses")
    
    print(f"\n🎉 DINOv2 Feature Test Complete!")
    return features, similarity_matrix

if __name__ == '__main__':
    test_feature_quality()
