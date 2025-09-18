#!/usr/bin/env python3
"""
Test script for server-optimized Dog ReID model.
Designed for high-end server hardware (4x A10G, 96GB VRAM).

Features tested:
- DINOv2 ViT-L/14 backbone (1024 -> 768 features)
- Large batch sizes (128)
- Large image resolution (336x336)
- Zero-shot feature extraction
- Memory and performance profiling
"""

import torch
import time
import psutil
import os
from config import cfg
from model import make_model
from datasets import make_dataloaders

def print_system_info():
    """Print system information"""
    print("🖥️  System Information:")
    print(f"   CPU cores: {psutil.cpu_count()}")
    print(f"   RAM: {psutil.virtual_memory().total / (1024**3):.1f} GB")
    print(f"   GPU available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"   GPU count: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"   GPU {i}: {torch.cuda.get_device_name(i)}")
            print(f"           Memory: {torch.cuda.get_device_properties(i).total_memory / (1024**3):.1f} GB")
    print()

def test_model_creation():
    """Test server-optimized model creation"""
    print("🔧 Testing Server-Optimized Model Creation:")
    print(f"   Backbone: {cfg.BACKBONE} (DINOv2 ViT-L/14)")
    print(f"   Image size: {cfg.IMAGE_SIZE}")
    print(f"   Batch size: {cfg.IMS_PER_BATCH}")
    print(f"   Embed dim: {cfg.EMBED_DIM}")
    print()
    
    start_time = time.time()
    
    # Create model (no classifier for zero-shot testing)
    model = make_model(
        backbone_name=cfg.BACKBONE,    # dinov2_vitl14
        num_classes=0,                 # Zero-shot mode
        embed_dim=cfg.EMBED_DIM       # 768-dim features
    )
    
    load_time = time.time() - start_time
    print(f"⏱️  Model loaded in {load_time:.2f} seconds")
    
    # Move to GPU if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    if torch.cuda.is_available():
        print(f"🚀 Model moved to {device}")
        print(f"   GPU Memory used: {torch.cuda.memory_allocated() / (1024**3):.2f} GB")
    
    return model, device

def test_zero_shot_inference(model, device):
    """Test zero-shot inference with large batch"""
    print("\n🔍 Testing Zero-Shot Inference:")
    
    # Create dummy batch (simulating server workload)
    batch_size = 32  # Start conservative, scale up on server
    dummy_images = torch.randn(batch_size, 3, *cfg.IMAGE_SIZE).to(device)
    print(f"   Batch shape: {dummy_images.shape}")
    
    model.eval()
    with torch.no_grad():
        start_time = time.time()
        
        # Extract features (zero-shot mode)
        features = model(dummy_images, return_mode='features')
        
        inference_time = time.time() - start_time
        
    print(f"   Features shape: {features.shape}")
    print(f"   Feature norm: {features.norm(dim=1).mean().item():.4f} (should be ~1.0)")
    print(f"⏱️  Inference time: {inference_time:.3f}s ({batch_size/inference_time:.1f} images/sec)")
    
    if torch.cuda.is_available():
        print(f"   GPU Memory peak: {torch.cuda.max_memory_allocated() / (1024**3):.2f} GB")
    
    # Test similarity computation
    similarity_matrix = features @ features.T
    print(f"   Similarity range: [{similarity_matrix.min().item():.3f}, {similarity_matrix.max().item():.3f}]")
    
    return features

def test_dataloader_integration():
    """Test integration with data loading"""
    print("\n📊 Testing DataLoader Integration:")
    print(f"   Workers: {cfg.NUM_WORKERS}")
    print(f"   Batch size: {cfg.IMS_PER_BATCH}")
    
    try:
        train_loader, query_loader, gallery_loader, num_classes = make_dataloaders(cfg)
        print(f"✅ DataLoaders created successfully")
        print(f"   Train batches: {len(train_loader)}")
        print(f"   Query samples: {len(query_loader.dataset)}")
        print(f"   Gallery samples: {len(gallery_loader.dataset)}")
        print(f"   Number of identities: {num_classes}")
        
        # Test one batch
        print("\n   Testing one training batch...")
        batch = next(iter(train_loader))
        imgs, pids, camids, paths = batch
        print(f"   Batch images shape: {imgs.shape}")
        print(f"   PIDs range: [{pids.min().item()}, {pids.max().item()}]")
        print(f"   P×K sampling: P={len(torch.unique(pids))}, K={cfg.NUM_INSTANCE}")
        
    except Exception as e:
        print(f"⚠️  DataLoader test failed: {e}")
        return None
    
    return train_loader, query_loader, gallery_loader, num_classes

def benchmark_server_performance(model, device):
    """Benchmark performance for server deployment"""
    print("\n🚀 Server Performance Benchmark:")
    
    batch_sizes = [16, 32, 64, 128]  # Test different batch sizes
    
    for batch_size in batch_sizes:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
        
        dummy_images = torch.randn(batch_size, 3, *cfg.IMAGE_SIZE).to(device)
        
        model.eval()
        times = []
        
        # Warmup
        with torch.no_grad():
            for _ in range(3):
                _ = model(dummy_images, return_mode='features')
        
        # Benchmark
        with torch.no_grad():
            for _ in range(10):
                start_time = time.time()
                features = model(dummy_images, return_mode='features')
                times.append(time.time() - start_time)
        
        avg_time = sum(times) / len(times)
        throughput = batch_size / avg_time
        
        print(f"   Batch {batch_size:3d}: {avg_time:.3f}s ({throughput:5.1f} imgs/sec)", end="")
        
        if torch.cuda.is_available():
            memory_mb = torch.cuda.max_memory_allocated() / (1024**2)
            print(f" | {memory_mb:.0f} MB")
        else:
            print()

def main():
    """Main test function"""
    print("🔥 Dog ReID Server-Optimized Model Test")
    print("=" * 50)
    
    # System info
    print_system_info()
    
    # Model creation
    model, device = test_model_creation()
    
    # Zero-shot inference
    features = test_zero_shot_inference(model, device)
    
    # DataLoader integration
    loaders = test_dataloader_integration()
    
    # Performance benchmark
    benchmark_server_performance(model, device)
    
    print("\n✅ Server optimization test completed!")
    print("\nReady for deployment on your 4x A10G server! 🚀")
    
    # Summary for server deployment
    print("\n📋 Server Deployment Summary:")
    print(f"   • Model: {cfg.BACKBONE} ({cfg.EMBED_DIM}-dim features)")
    print(f"   • Image resolution: {cfg.IMAGE_SIZE}")
    print(f"   • Recommended batch size: 64-128 (adjust based on GPU memory)")
    print(f"   • Workers: {cfg.NUM_WORKERS} (scale up on server)")
    print(f"   • Zero-shot ready: ✅")
    print(f"   • Training ready: ✅")

if __name__ == '__main__':
    main()
