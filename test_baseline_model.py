#!/usr/bin/env python3
"""
Inference script for trained Baseline Dog ReID model (DINOv3-B or similar).
Runs evaluation on specified test query/gallery splits.
"""

import sys
import os
import argparse
import random
import numpy as np
import torch

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config_petface import cfg as base_cfg
from model import make_model
from processor import do_inference
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


def main():
    """Main inference function."""
    parser = argparse.ArgumentParser(description="Baseline Dog ReID Inference")
    parser.add_argument(
        "--checkpoint", 
        required=True, 
        help="Path to trained model checkpoint (.pth file)"
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
        default="./inference_results",
        help="Output directory for results (default: ./inference_results)"
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
    
    args = parser.parse_args()
    
    # Verify checkpoint exists
    if not os.path.exists(args.checkpoint):
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")
    
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
            
            # Override test splits
            self.TEST_QUERY_SPLIT = args.test_query_csv
            self.TEST_GALLERY_SPLIT = args.test_gallery_csv
            self.TEST_BATCH_SIZE = args.batch_size
            self.NUM_WORKERS = args.num_workers
    
    cfg = InferenceConfig()
    
    # Set seed
    set_seed(42)
    
    # Create output directory
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    
    # Setup logger
    logger = setup_logger("baseline_inference", args.output_dir, if_train=False)
    logger.info("🔍 Starting Baseline Dog ReID Inference")
    logger.info(f"Checkpoint: {args.checkpoint}")
    logger.info(f"Test query CSV: {args.test_query_csv}")
    logger.info(f"Test gallery CSV: {args.test_gallery_csv}")
    logger.info(f"Output directory: {args.output_dir}")
    
    # Print configuration
    print("\n" + "="*80)
    print("🎯 BASELINE DOG REID INFERENCE CONFIGURATION")
    print("="*80)
    print(f"Checkpoint:        {args.checkpoint}")
    print(f"Backbone:          {cfg.BACKBONE}")
    print(f"Embedding Dim:     {cfg.EMBED_DIM}")
    print(f"Batch Size:        {args.batch_size}")
    print(f"Test Query CSV:    {args.test_query_csv}")
    print(f"Test Gallery CSV:  {args.test_gallery_csv}")
    print("="*80 + "\n")
    
    # Load checkpoint to get num_classes
    logger.info("📂 Loading checkpoint...")
    checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    
    # Infer num_classes from checkpoint
    # The classifier weight shape is [num_classes, feat_dim]
    if 'model' in checkpoint:
        state_dict = checkpoint['model']
    else:
        state_dict = checkpoint
    
    # Find num_classes from classifier weight
    num_classes = None
    for key in state_dict.keys():
        if 'classifier' in key and 'weight' in key and 'bn_neck' not in key:
            num_classes = state_dict[key].shape[0]
            break
    
    if num_classes is None:
        raise ValueError("Could not determine num_classes from checkpoint")
    
    logger.info(f"   Detected {num_classes} classes from checkpoint")
    if 'epoch' in checkpoint:
        logger.info(f"   Checkpoint epoch: {checkpoint['epoch']}")
    if 'best_mAP' in checkpoint:
        logger.info(f"   Best mAP: {checkpoint['best_mAP']:.1%}")
    
    # Create model
    logger.info("🏗️  Creating model...")
    model = make_model(
        backbone_name=cfg.BACKBONE,
        num_classes=num_classes,
        embed_dim=cfg.EMBED_DIM,
        pretrained=False,  # We're loading weights from checkpoint
        bn_neck=cfg.BN_NECK
    )
    
    # Load checkpoint weights
    logger.info("⚙️  Loading model weights...")
    model.load_state_dict(state_dict)
    logger.info("✅ Model loaded successfully")
    
    # Create test data loaders
    logger.info("📊 Creating test data loaders...")
    import time
    start_time = time.time()
    
    from datasets.make_dataloader_petface import make_petface_dataloaders
    
    # Note: make_petface_dataloaders returns all loaders, we only need test loaders
    # The function will create train/val loaders from the CSVs in cfg, but we only use test
    (_, _, _, test_query_loader, test_gallery_loader, _) = make_petface_dataloaders(cfg)
    
    logger.info(f"✅ Data loaders created in {time.time() - start_time:.1f}s")
    logger.info(f"   Test query samples: {len(test_query_loader.dataset)}")
    logger.info(f"   Test gallery samples: {len(test_gallery_loader.dataset)}")
    
    # Run inference
    logger.info("\n" + "="*80)
    logger.info("🚀 Starting inference...")
    logger.info("="*80)
    
    inference_start = time.time()
    cmc, mAP = do_inference(cfg, model, test_query_loader, test_gallery_loader)
    inference_time = time.time() - inference_start
    
    # Print results
    logger.info("\n" + "="*80)
    logger.info("📊 EVALUATION RESULTS")
    logger.info("="*80)
    logger.info(f"mAP: {mAP:.1%}")
    logger.info("\nCMC Curve:")
    for r in [1, 5, 10, 20]:
        if r <= len(cmc):
            logger.info(f"  Rank-{r:<3}: {cmc[r - 1]:.1%}")
    logger.info(f"\nInference time: {inference_time:.1f}s")
    logger.info(f"Avg time per image: {inference_time / (len(test_query_loader.dataset) + len(test_gallery_loader.dataset)) * 1000:.2f}ms")
    logger.info("="*80)
    
    # Save results
    results = {
        'checkpoint': args.checkpoint,
        'test_query_csv': args.test_query_csv,
        'test_gallery_csv': args.test_gallery_csv,
        'num_query': len(test_query_loader.dataset),
        'num_gallery': len(test_gallery_loader.dataset),
        'mAP': float(mAP),
        'cmc': [float(c) for c in cmc[:20]],  # Save top-20
        'inference_time': inference_time,
        'config': {
            'backbone': cfg.BACKBONE,
            'embed_dim': cfg.EMBED_DIM,
            'num_classes': num_classes
        }
    }
    
    results_path = os.path.join(args.output_dir, 'inference_results.pth')
    torch.save(results, results_path)
    logger.info(f"\n💾 Results saved to: {results_path}")
    
    # Also save as text file for easy viewing
    text_results_path = os.path.join(args.output_dir, 'inference_results.txt')
    with open(text_results_path, 'w') as f:
        f.write("="*80 + "\n")
        f.write("Baseline Dog ReID Inference Results\n")
        f.write("="*80 + "\n\n")
        f.write(f"Checkpoint: {args.checkpoint}\n")
        f.write(f"Test Query CSV: {args.test_query_csv}\n")
        f.write(f"Test Gallery CSV: {args.test_gallery_csv}\n")
        f.write(f"Num Query: {len(test_query_loader.dataset)}\n")
        f.write(f"Num Gallery: {len(test_gallery_loader.dataset)}\n\n")
        f.write(f"Backbone: {cfg.BACKBONE}\n")
        f.write(f"Embedding Dim: {cfg.EMBED_DIM}\n")
        f.write(f"Num Classes: {num_classes}\n\n")
        f.write("="*80 + "\n")
        f.write(f"mAP: {mAP:.1%}\n\n")
        f.write("CMC Curve:\n")
        for r in [1, 5, 10, 20]:
            if r <= len(cmc):
                f.write(f"  Rank-{r:<3}: {cmc[r - 1]:.1%}\n")
        f.write(f"\nInference time: {inference_time:.1f}s\n")
        f.write(f"Avg time per image: {inference_time / (len(test_query_loader.dataset) + len(test_gallery_loader.dataset)) * 1000:.2f}ms\n")
        f.write("="*80 + "\n")
    
    logger.info(f"📄 Text results saved to: {text_results_path}")
    logger.info("\n✅ Inference completed successfully!")


if __name__ == '__main__':
    main()

