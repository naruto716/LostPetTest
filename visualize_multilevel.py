#!/usr/bin/env python3
"""
Visualize Multi-Level Feature Extraction
🔍 Show exactly what we're extracting from each layer
"""

import torch
import numpy as np
from transformers import AutoModel

def visualize_multilevel_extraction():
    print("🔍 Multi-Level DINOv3 Feature Extraction Visualization")
    print("=" * 70)
    
    # Load model and create dummy input
    model = AutoModel.from_pretrained('facebook/dinov3-vits16-pretrain-lvd1689m', trust_remote_code=True)
    dummy_input = torch.randn(2, 3, 336, 336)  # 2 images
    
    print(f"📥 Input shape: {dummy_input.shape}")
    
    # Forward pass with hidden states
    with torch.no_grad():
        outputs = model(pixel_values=dummy_input, output_hidden_states=True)
        hidden_states = outputs.hidden_states
    
    print(f"🧠 Total layers: {len(hidden_states)}")
    print(f"📊 Each layer shape: {hidden_states[0].shape}")
    
    print("\n🎯 CLS Token Evolution Across Layers:")
    extract_layers = [2, 5, 8, 11, 12]  # Layers 3, 6, 9, 12, final
    
    cls_tokens = []
    for i, layer_idx in enumerate(extract_layers):
        cls_token = hidden_states[layer_idx][:, 0]  # Extract CLS token
        cls_tokens.append(cls_token)
        
        # Show some statistics
        mean_val = cls_token.mean().item()
        std_val = cls_token.std().item()
        print(f"   Layer {layer_idx+1:2d}: CLS shape {cls_token.shape} | Mean: {mean_val:6.3f} | Std: {std_val:6.3f}")
    
    # Concatenate
    multi_level = torch.cat(cls_tokens, dim=1)
    print(f"\n🔗 Concatenated multi-level features: {multi_level.shape}")
    
    print(f"\n📊 Feature Composition:")
    print(f"   Layer 3 (early):  features[0:384]     - Low-level patterns")
    print(f"   Layer 6 (mid):    features[384:768]   - Mid-level shapes") 
    print(f"   Layer 9 (high):   features[768:1152]  - High-level parts")
    print(f"   Layer 12 (late):  features[1152:1536] - Complex patterns")
    print(f"   Final layer:      features[1536:1920] - Global representation")
    
    print(f"\n🎯 This captures the hierarchical learning progression!")
    
    # Show feature similarity between layers
    print(f"\n🔍 CLS Token Similarity Between Layers:")
    for i in range(len(cls_tokens)):
        for j in range(i+1, len(cls_tokens)):
            # Cosine similarity
            sim = torch.nn.functional.cosine_similarity(
                cls_tokens[i], cls_tokens[j], dim=1
            ).mean().item()
            layer_names = [f"L{extract_layers[i]+1}", f"L{extract_layers[j]+1}"]
            print(f"   {layer_names[0]} ↔ {layer_names[1]}: {sim:.3f}")
    
    print(f"\n💡 Interpretation:")
    print(f"   Higher similarity = layers learned similar representations")
    print(f"   Lower similarity = layers learned complementary features")
    print(f"   Our goal: Combine complementary multi-scale features!")

if __name__ == "__main__":
    visualize_multilevel_extraction()
