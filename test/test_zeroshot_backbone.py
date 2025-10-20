#!/usr/bin/env python3
"""
Zero-shot inference script for pretrained backbones.
Evaluates pretrained models (SWIN, ResNet, etc.) without any fine-tuning.
Pure feature matching using ImageNet-pretrained weights.
"""

import sys
import os
import argparse
import random
import numpy as np
import torch
import torch.nn as nn
import logging

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config_petface import cfg as base_cfg
from model.backbones import build_backbone
from utils import setup_logger


def set_seed(seed=42):
    """Set random seed for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def extract_features(model, data_loader, device):
    """
    Extract features from backbone model.
    
    Args:
        model: Backbone model
        data_loader: DataLoader for images
        device: Device to run on
    
    Returns:
        features: [N, feat_dim] feature matrix
        pids: [N] person IDs
        camids: [N] camera IDs
    """
    model.eval()
    features_list = []
    pids_list = []
    camids_list = []
    
    logger = logging.getLogger()
    
    with torch.no_grad():
        for n_iter, batch in enumerate(data_loader):
            # Unpack batch (standard format: img, pid, camid, path)
            img = batch[0]
            pid = batch[1]
            camid = batch[2]
            
            if (n_iter + 1) % 20 == 0:
                logger.info(f"   Processed {n_iter + 1}/{len(data_loader)} batches")
            
            img = img.to(device)
            
            # Extract features directly from backbone
            feat = model(img)
            
            # Flatten if needed (some backbones output [B, D, 1, 1])
            if feat.ndim > 2:
                feat = torch.flatten(feat, 1)
            
            # L2 normalize features (standard for ReID)
            feat = nn.functional.normalize(feat, p=2, dim=1)
            
            features_list.append(feat.cpu())
            pids_list.extend(pid.cpu().numpy())
            camids_list.extend(camid.cpu().numpy())
    
    features = torch.cat(features_list, dim=0)
    pids = np.asarray(pids_list)
    camids = np.asarray(camids_list)
    
    return features, pids, camids


def main():
    """Main zero-shot inference function."""
    parser = argparse.ArgumentParser(description="Zero-Shot Backbone Inference (No Training)")
    parser.add_argument(
        "--backbone",
        required=True,
        help="Backbone name (e.g., 'swin_base_patch4_window7_224', 'resnet50')"
    )
    parser.add_argument(
        "--test_query_csv",
        required=True,
        help="Path to test query CSV file"
    )
    parser.add_argument(
        "--test_gallery_csv",
        required=True,
        help="Path to test gallery CSV file"
    )
    parser.add_argument(
        "--output_dir",
        default="./zeroshot_results",
        help="Output directory for results (default: ./zeroshot_results)"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=128,
        help="Batch size for inference (default: 128)"
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=4,
        help="Number of data loading workers (default: 4)"
    )
    parser.add_argument(
        "--image_size",
        type=int,
        nargs=2,
        default=None,
        help="Image size as two integers (e.g., 224 224). Auto-detected if not provided."
    )
    
    args = parser.parse_args()
    
    # Verify test splits exist
    if not os.path.exists(args.test_query_csv):
        raise FileNotFoundError(f"Test query CSV not found: {args.test_query_csv}")
    if not os.path.exists(args.test_gallery_csv):
        raise FileNotFoundError(f"Test gallery CSV not found: {args.test_gallery_csv}")
    
    # Create a config copy with test splits
    class InferenceConfig:
        def __init__(self):
            # Copy all attributes from base config
            for attr in dir(base_cfg):
                if not attr.startswith('_'):
                    setattr(self, attr, getattr(base_cfg, attr))
            
            # Override test splits with absolute paths
            self.TEST_QUERY_SPLIT = os.path.abspath(args.test_query_csv)
            self.TEST_GALLERY_SPLIT = os.path.abspath(args.test_gallery_csv)
            self.TEST_BATCH_SIZE = args.batch_size
            self.NUM_WORKERS = args.num_workers
            
            # Override image size if provided
            if args.image_size:
                self.IMAGE_SIZE = tuple(args.image_size)
            # SWIN requires 224x224
            elif 'swin' in args.backbone.lower():
                self.IMAGE_SIZE = (224, 224)
    
    cfg = InferenceConfig()
    
    # Set seed
    set_seed(42)
    
    # Create output directory
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    
    # Setup logger
    logger = setup_logger("zeroshot_inference", args.output_dir, if_train=False)
    logger.info("Zero-Shot Backbone Inference (No Training)")
    logger.info(f"Backbone: {args.backbone}")
    logger.info(f"Test query CSV: {args.test_query_csv}")
    logger.info(f"Test gallery CSV: {args.test_gallery_csv}")
    logger.info(f"Output directory: {args.output_dir}")
    
    # Print configuration
    print("\n" + "="*80)
    print("ZERO-SHOT BACKBONE INFERENCE")
    print("="*80)
    print(f"Backbone:        {args.backbone}")
    print(f"Pretrained:      ImageNet weights (no dog fine-tuning)")
    print(f"Image Size:      {cfg.IMAGE_SIZE}")
    print(f"Batch Size:      {args.batch_size}")
    print(f"Test Query CSV:  {args.test_query_csv}")
    print(f"Test Gallery CSV: {args.test_gallery_csv}")
    print("="*80 + "\n")
    
    # Build backbone
    logger.info(f"Building pretrained backbone: {args.backbone}")
    backbone, feat_dim = build_backbone(args.backbone, pretrained=True)
    logger.info(f"   Feature dimension: {feat_dim}")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    backbone.to(device)
    
    if torch.cuda.device_count() > 1:
        logger.info(f"Using {torch.cuda.device_count()} GPUs with DataParallel")
        backbone = nn.DataParallel(backbone)
    
    # Create test data loaders
    logger.info("Creating test data loaders...")
    import time
    start_time = time.time()
    
    from datasets.make_dataloader_petface import make_petface_dataloaders
    
    # Note: make_petface_dataloaders returns all loaders, we only need test loaders
    (_, _, _, test_query_loader, test_gallery_loader, _) = make_petface_dataloaders(cfg)
    
    logger.info(f"Data loaders created in {time.time() - start_time:.1f}s")
    logger.info(f"   Test query samples: {len(test_query_loader.dataset)}")
    logger.info(f"   Test gallery samples: {len(test_gallery_loader.dataset)}")
    
    # Extract query features
    logger.info("\n" + "="*80)
    logger.info("Extracting query features...")
    logger.info("="*80)
    query_start = time.time()
    query_features, query_pids, query_camids = extract_features(backbone, test_query_loader, device)
    query_time = time.time() - query_start
    logger.info(f"Query features extracted: {query_features.shape} in {query_time:.1f}s")
    
    # Extract gallery features
    logger.info("\n" + "="*80)
    logger.info("Extracting gallery features...")
    logger.info("="*80)
    gallery_start = time.time()
    gallery_features, gallery_pids, gallery_camids = extract_features(backbone, test_gallery_loader, device)
    gallery_time = time.time() - gallery_start
    logger.info(f"Gallery features extracted: {gallery_features.shape} in {gallery_time:.1f}s")
    
    # Compute distance matrix and metrics
    logger.info("\n" + "="*80)
    logger.info("Computing distance matrix and metrics...")
    logger.info("="*80)
    
    from utils.metrics import euclidean_distance, eval_func
    
    metric_start = time.time()
    distmat = euclidean_distance(query_features, gallery_features)
    cmc, mAP = eval_func(distmat, query_pids, gallery_pids, query_camids, gallery_camids, max_rank=50)
    metric_time = time.time() - metric_start
    
    logger.info(f"Metrics computed in {metric_time:.1f}s")
    
    # Print results
    total_time = query_time + gallery_time + metric_time
    
    logger.info("\n" + "="*80)
    logger.info("ZERO-SHOT EVALUATION RESULTS")
    logger.info("="*80)
    logger.info(f"Backbone: {args.backbone}")
    logger.info(f"Feature Dim: {feat_dim}")
    logger.info(f"Pretrained: ImageNet (no dog-specific training)")
    logger.info("")
    logger.info(f"mAP: {mAP:.1%}")
    logger.info("\nCMC Curve:")
    for r in [1, 5, 10, 20]:
        if r <= len(cmc):
            logger.info(f"  Rank-{r:<3}: {cmc[r - 1]:.1%}")
    logger.info(f"\nTotal inference time: {total_time:.1f}s")
    logger.info(f"Avg time per image: {total_time / (len(test_query_loader.dataset) + len(test_gallery_loader.dataset)) * 1000:.2f}ms")
    logger.info("="*80)
    
    # Save results
    results = {
        'backbone': args.backbone,
        'pretrained': 'ImageNet',
        'feat_dim': feat_dim,
        'test_query_csv': args.test_query_csv,
        'test_gallery_csv': args.test_gallery_csv,
        'num_query': len(test_query_loader.dataset),
        'num_gallery': len(test_gallery_loader.dataset),
        'mAP': float(mAP),
        'cmc': [float(c) for c in cmc[:20]],
        'inference_time': total_time,
        'query_time': query_time,
        'gallery_time': gallery_time,
        'metric_time': metric_time,
    }
    
    results_path = os.path.join(args.output_dir, 'zeroshot_results.pth')
    torch.save(results, results_path)
    logger.info(f"\nResults saved to: {results_path}")
    
    # Also save as text file
    text_results_path = os.path.join(args.output_dir, 'zeroshot_results.txt')
    with open(text_results_path, 'w') as f:
        f.write("="*80 + "\n")
        f.write("Zero-Shot Backbone Inference Results\n")
        f.write("="*80 + "\n\n")
        f.write(f"Backbone: {args.backbone}\n")
        f.write(f"Feature Dim: {feat_dim}\n")
        f.write(f"Pretrained: ImageNet (no dog-specific training)\n\n")
        f.write(f"Test Query CSV: {args.test_query_csv}\n")
        f.write(f"Test Gallery CSV: {args.test_gallery_csv}\n")
        f.write(f"Num Query: {len(test_query_loader.dataset)}\n")
        f.write(f"Num Gallery: {len(test_gallery_loader.dataset)}\n\n")
        f.write("="*80 + "\n")
        f.write(f"mAP: {mAP:.1%}\n\n")
        f.write("CMC Curve:\n")
        for r in [1, 5, 10, 20]:
            if r <= len(cmc):
                f.write(f"  Rank-{r:<3}: {cmc[r - 1]:.1%}\n")
        f.write(f"\nTotal inference time: {total_time:.1f}s\n")
        f.write(f"Query time: {query_time:.1f}s\n")
        f.write(f"Gallery time: {gallery_time:.1f}s\n")
        f.write(f"Metric computation time: {metric_time:.1f}s\n")
        f.write(f"Avg time per image: {total_time / (len(test_query_loader.dataset) + len(test_gallery_loader.dataset)) * 1000:.2f}ms\n")
        f.write("="*80 + "\n")
    
    logger.info(f"Text results saved to: {text_results_path}")
    logger.info("\nZero-shot inference completed successfully!")


if __name__ == '__main__':
    main()
