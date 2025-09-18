#!/usr/bin/env python3
"""
DINOv2 vs DINOv3 Head-to-Head Comparison for Dog ReID

Compares the latest DINOv3 (Aug 2025) against our proven DINOv2 on the same dog images.
Tests feature quality, similarity patterns, and performance on your server.

DINOv3 promises:
- 100x larger training dataset (1.7B images)
- Up to 7B parameters  
- Enhanced training strategies
- State-of-the-art performance
"""

import torch
import time
import os
from PIL import Image
from config import cfg
from model import make_model
from datasets.make_dataloader_dogreid import build_transforms
import numpy as np

def load_test_images(num_images=8):
    """Load test images from query directory"""
    image_dir = os.path.join(cfg.ROOT_DIR, cfg.IMAGES_DIR, "query")
    image_files = [f for f in os.listdir(image_dir) if f.endswith('.jpg')][:num_images]
    
    test_transforms = build_transforms(cfg, is_train=False)
    images = []
    
    for img_file in image_files:
        img_path = os.path.join(image_dir, img_file)
        img = Image.open(img_path).convert('RGB')
        img_tensor = test_transforms(img).unsqueeze(0)
        images.append(img_tensor)
    
    batch = torch.cat(images, dim=0)
    return batch, image_files

def test_model_performance(model_name, backbone_name, batch, device):
    """Test a single model's performance and feature quality"""
    print(f"\n🔧 Testing {model_name}:")
    print(f"   Backbone: {backbone_name}")
    
    start_time = time.time()
    
    # Create model
    try:
        model = make_model(
            backbone_name=backbone_name,
            num_classes=0,
            embed_dim=cfg.EMBED_DIM
        )
        model.to(device)
        model.eval()
        
        load_time = time.time() - start_time
        print(f"   ⏱️  Load time: {load_time:.2f}s")
        
    except Exception as e:
        print(f"   ❌ Failed to load: {e}")
        return None
    
    # Test inference
    batch = batch.to(device)
    
    with torch.no_grad():
        # Warmup
        for _ in range(3):
            _ = model(batch, return_mode='features')
        
        # Benchmark
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        start_time = time.time()
        
        features = model(batch, return_mode='features')
        
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        inference_time = time.time() - start_time
    
    # Compute similarity matrix
    similarity_matrix = features @ features.T
    
    # Memory usage
    memory_mb = torch.cuda.max_memory_allocated() / (1024**2) if torch.cuda.is_available() else 0
    
    return {
        'features': features,
        'similarity_matrix': similarity_matrix,
        'inference_time': inference_time,
        'memory_mb': memory_mb,
        'load_time': load_time
    }

def analyze_similarity_quality(similarity_matrix, image_files, model_name):
    """Analyze the quality of similarity patterns"""
    
    # Extract dog IDs from filenames (assuming format: dogID_xxx.jpg)
    def get_dog_id(filename):
        return filename.split('_')[0]
    
    dog_ids = [get_dog_id(f) for f in image_files]
    n = len(image_files)
    
    # Separate same-dog vs different-dog similarities
    same_dog_sims = []
    diff_dog_sims = []
    
    for i in range(n):
        for j in range(i+1, n):  # Upper triangle only
            sim = similarity_matrix[i, j].item()
            if dog_ids[i] == dog_ids[j]:
                same_dog_sims.append(sim)
            else:
                diff_dog_sims.append(sim)
    
    # Calculate metrics
    same_dog_mean = np.mean(same_dog_sims) if same_dog_sims else 0
    diff_dog_mean = np.mean(diff_dog_sims) if diff_dog_sims else 0
    discrimination = same_dog_mean - diff_dog_mean
    
    print(f"\n   📊 {model_name} Similarity Analysis:")
    print(f"      Same dog pairs: {len(same_dog_sims)} pairs, avg sim: {same_dog_mean:.3f}")
    print(f"      Diff dog pairs: {len(diff_dog_sims)} pairs, avg sim: {diff_dog_mean:.3f}")
    print(f"      Discrimination: {discrimination:.3f} (higher is better)")
    
    # Overall similarity range
    mask = ~torch.eye(n, dtype=bool)
    all_sims = similarity_matrix[mask].cpu().numpy()
    print(f"      Similarity range: [{all_sims.min():.3f}, {all_sims.max():.3f}]")
    
    return {
        'same_dog_mean': same_dog_mean,
        'diff_dog_mean': diff_dog_mean, 
        'discrimination': discrimination,
        'similarity_range': (all_sims.min(), all_sims.max())
    }

def compare_models():
    """Main comparison function"""
    print("🥊 DINOv2 vs DINOv3 Head-to-Head Comparison")
    print("=" * 60)
    print("🎯 Testing on real dog images from your dataset")
    print("🚀 Optimized for your 4x A10G server")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🖥️  Device: {device}")
    
    # Load test images
    print(f"\n📸 Loading test images...")
    batch, image_files = load_test_images(num_images=8)
    print(f"   Images: {len(image_files)}")
    for i, img_file in enumerate(image_files):
        print(f"   {i}: {img_file}")
    
    # Test models
    models_to_test = [
        ("DINOv2-L", "dinov2_vitl14"),
        ("DINOv3-L", "dinov3_large"),  # Start with large for fair comparison
    ]
    
    results = {}
    
    for model_name, backbone_name in models_to_test:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            
        result = test_model_performance(model_name, backbone_name, batch, device)
        
        if result is not None:
            results[model_name] = result
            
            # Performance metrics
            throughput = len(batch) / result['inference_time']
            print(f"   ⚡ Performance: {throughput:.1f} imgs/sec, {result['memory_mb']:.0f} MB")
            
            # Feature quality analysis
            quality = analyze_similarity_quality(
                result['similarity_matrix'], 
                image_files, 
                model_name
            )
            results[model_name]['quality'] = quality
        
        print()  # Spacing
    
    # Head-to-head comparison
    if len(results) == 2:
        print("\n🏆 HEAD-TO-HEAD COMPARISON")
        print("=" * 40)
        
        dinov2_result = results.get("DINOv2-L")
        dinov3_result = results.get("DINOv3-L") 
        
        if dinov2_result and dinov3_result:
            
            # Performance comparison
            dinov2_fps = len(batch) / dinov2_result['inference_time']
            dinov3_fps = len(batch) / dinov3_result['inference_time']
            fps_improvement = (dinov3_fps - dinov2_fps) / dinov2_fps * 100
            
            print(f"📈 PERFORMANCE:")
            print(f"   DINOv2-L: {dinov2_fps:.1f} imgs/sec | {dinov2_result['memory_mb']:.0f} MB")
            print(f"   DINOv3-L: {dinov3_fps:.1f} imgs/sec | {dinov3_result['memory_mb']:.0f} MB")
            print(f"   Speed change: {fps_improvement:+.1f}%")
            
            # Memory comparison
            memory_change = (dinov3_result['memory_mb'] - dinov2_result['memory_mb']) / dinov2_result['memory_mb'] * 100
            print(f"   Memory change: {memory_change:+.1f}%")
            
            # Quality comparison
            dinov2_disc = dinov2_result['quality']['discrimination']
            dinov3_disc = dinov3_result['quality']['discrimination']
            disc_improvement = (dinov3_disc - dinov2_disc) / dinov2_disc * 100
            
            print(f"\n🎯 FEATURE QUALITY:")
            print(f"   DINOv2-L discrimination: {dinov2_disc:.3f}")
            print(f"   DINOv3-L discrimination: {dinov3_disc:.3f}")
            print(f"   Quality improvement: {disc_improvement:+.1f}%")
            
            # Winner determination
            print(f"\n🏆 VERDICT:")
            
            # Score based on discrimination (most important for ReID)
            if dinov3_disc > dinov2_disc * 1.05:  # 5% improvement threshold
                print(f"   🥇 DINOv3 WINS on feature quality! ({disc_improvement:+.1f}% better discrimination)")
            elif dinov2_disc > dinov3_disc * 1.05:
                print(f"   🥇 DINOv2 WINS on feature quality! ({-disc_improvement:+.1f}% better discrimination)")
            else:
                print(f"   🤝 TIE on feature quality (within 5%)")
            
            # Performance verdict
            if fps_improvement > 10:
                print(f"   ⚡ DINOv3 WINS on speed! ({fps_improvement:+.1f}% faster)")
            elif fps_improvement < -10:
                print(f"   ⚡ DINOv2 WINS on speed! ({-fps_improvement:+.1f}% faster)")
            else:
                print(f"   ⚡ Similar speed performance")
                
            # Overall recommendation
            print(f"\n💡 RECOMMENDATION:")
            if dinov3_disc > dinov2_disc and abs(fps_improvement) < 20:
                print(f"   🚀 Upgrade to DINOv3! Better features, similar performance")
            elif dinov2_disc > dinov3_disc * 1.1:
                print(f"   🛡️  Stick with DINOv2 for now - better proven performance")
            else:
                print(f"   🔬 Both are excellent! DINOv3 has potential but needs more testing")
    
    print(f"\n✅ Comparison complete! Both models tested on your server hardware.")
    return results

if __name__ == '__main__':
    compare_models()
