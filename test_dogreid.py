"""
Evaluation script for Dog ReID.
Based on CLIP-ReID's test.py structure.
"""

import argparse
import os
import torch
from config_training import cfg
from datasets import make_dataloaders
from model import make_model
from processor import do_inference
from utils import setup_logger

def main():
    """Main evaluation function."""
    parser = argparse.ArgumentParser(description="Dog ReID Evaluation")
    parser.add_argument(
        "--checkpoint", 
        required=True,
        help="Path to model checkpoint"
    )
    parser.add_argument(
        "--output_dir", 
        default="./outputs/evaluation", 
        help="Output directory for results"
    )
    
    args = parser.parse_args()
    
    # Create output directory
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    
    # Setup logger
    logger = setup_logger("dogreid", args.output_dir, if_train=False)
    logger.info("🔍 Starting Dog ReID Evaluation")
    logger.info(f"Model checkpoint: {args.checkpoint}")
    
    # Create data loaders
    logger.info("📊 Creating data loaders...")
    train_loader, query_loader, gallery_loader, num_classes = make_dataloaders(cfg)
    
    # Combine query and gallery for evaluation
    val_dataset = torch.utils.data.ConcatDataset([query_loader.dataset, gallery_loader.dataset])
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=64,
        shuffle=False,
        num_workers=cfg.NUM_WORKERS,
        pin_memory=True,
        collate_fn=query_loader.collate_fn
    )
    num_query = len(query_loader.dataset)
    
    logger.info(f"   Query samples: {len(query_loader.dataset)}")
    logger.info(f"   Gallery samples: {len(gallery_loader.dataset)}")
    logger.info(f"   Number of classes: {num_classes}")
    
    # Create model
    logger.info("🏗️  Creating model...")
    model = make_model(
        backbone_name=cfg.BACKBONE,
        num_classes=num_classes,
        embed_dim=cfg.EMBED_DIM,
        pretrained=cfg.PRETRAINED,
        bn_neck=cfg.BN_NECK
    )
    
    # Load checkpoint
    logger.info(f"📂 Loading checkpoint...")
    checkpoint = torch.load(args.checkpoint, map_location='cpu')
    
    if 'model' in checkpoint:
        model.load_state_dict(checkpoint['model'])
        if 'epoch' in checkpoint:
            logger.info(f"   Checkpoint from epoch: {checkpoint['epoch']}")
        if 'mAP' in checkpoint:
            logger.info(f"   Checkpoint mAP: {checkpoint['mAP']:.1%}")
    else:
        # Direct state dict
        model.load_state_dict(checkpoint)
    
    # Run evaluation
    logger.info("🔍 Running evaluation...")
    cmc, mAP = do_inference(cfg, model, val_loader, num_query)
    
    # Print results
    logger.info("=" * 50)
    logger.info("🏆 EVALUATION RESULTS")
    logger.info("=" * 50)
    logger.info("mAP: {:.1%}".format(mAP))
    for r in [1, 5, 10]:
        logger.info("CMC curve, Rank-{:<3}:{:.1%}".format(r, cmc[r - 1]))
    logger.info("=" * 50)
    
    # Save results
    results = {
        'mAP': mAP,
        'cmc': cmc,
        'rank1': cmc[0],
        'rank5': cmc[4],
        'rank10': cmc[9]
    }
    
    results_path = os.path.join(args.output_dir, 'evaluation_results.pth')
    torch.save(results, results_path)
    logger.info(f"💾 Results saved to: {results_path}")

if __name__ == '__main__':
    main()
