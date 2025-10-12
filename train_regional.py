#!/usr/bin/env python3
"""
Regional Dog ReID Training Script
Trains with facial region features (eyes, nose, mouth, ears, forehead)
"""

import sys
import os
import argparse
import random
import numpy as np
import torch

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config_regional import cfg
from datasets.make_dataloader_regional import make_regional_dataloaders
from model import make_regional_model
from loss import make_loss
from solver import make_optimizer, WarmupMultiStepLR
from processor.processor_regional import do_train, do_inference
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


def print_config():
    """Print training configuration."""
    print("\n" + "="*80)
    print("🎯 REGIONAL DOG REID TRAINING CONFIGURATION")
    print("="*80)
    print(f"Backbone:          {cfg.BACKBONE}")
    print(f"Embedding Dim:     {cfg.EMBED_DIM}")
    print(f"Freeze Backbone:   {cfg.FREEZE_BACKBONE}")
    print(f"Max Epochs:        {cfg.MAX_EPOCHS}")
    print(f"Base LR:           {cfg.BASE_LR}")
    print(f"Batch Size:        {cfg.IMS_PER_BATCH} ({cfg.IMS_PER_BATCH // cfg.NUM_INSTANCE} IDs × {cfg.NUM_INSTANCE} images)")
    print(f"Image Size:        {cfg.IMAGE_SIZE}")
    print(f"Regional Size:     {cfg.REGIONAL_SIZE}")
    print(f"Output Dir:        {cfg.OUTPUT_DIR}")
    print("="*80 + "\n")


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description="Regional Dog ReID Training")
    parser.add_argument("--output_dir", default=None, help="Output directory (overrides config)")
    parser.add_argument("--epochs", type=int, default=None, help="Number of epochs (overrides config)")
    parser.add_argument("--eval_only", action='store_true', help="Only run evaluation")
    parser.add_argument("--resume", default=None, help="Path to checkpoint to resume from")
    
    args = parser.parse_args()
    
    # Update config
    if args.output_dir:
        cfg.OUTPUT_DIR = args.output_dir
    if args.epochs:
        cfg.MAX_EPOCHS = args.epochs
    
    # Set seed
    set_seed(42)
    
    # Create output directory
    if not os.path.exists(cfg.OUTPUT_DIR):
        os.makedirs(cfg.OUTPUT_DIR)
    
    # Setup logger
    logger = setup_logger("regional_dogreid", cfg.OUTPUT_DIR, if_train=True)
    logger.info("🚀 Starting Regional Dog ReID Training")
    logger.info(f"Saving model in: {cfg.OUTPUT_DIR}")
    
    # Print configuration
    print_config()
    logger.info(f"Arguments: {args}")
    
    # Create data loaders
    logger.info("📊 Creating regional data loaders...")
    (train_loader, val_query_loader, val_gallery_loader,
     test_query_loader, test_gallery_loader, num_classes) = make_regional_dataloaders(
        cfg=cfg,
        landmarks_dir=cfg.LANDMARKS_DIR
    )
    
    logger.info(f"   Training samples: {len(train_loader.dataset)}")
    logger.info(f"   Val query samples: {len(val_query_loader.dataset)}")
    logger.info(f"   Val gallery samples: {len(val_gallery_loader.dataset)}")
    logger.info(f"   Number of classes: {num_classes}")
    
    # Create model
    logger.info("🏗️  Creating regional model...")
    use_attention = getattr(cfg, 'USE_ATTENTION', False)  # Default to False if not specified
    model = make_regional_model(
        backbone_name=cfg.BACKBONE,
        num_classes=num_classes,
        embed_dim=cfg.EMBED_DIM,
        pretrained=cfg.PRETRAINED,
        bn_neck=cfg.BN_NECK,
        use_attention=use_attention
    )
    if use_attention:
        logger.info("✨ Using attention-based fusion (learns to weight regions by importance)")
    
    # Freeze backbone (standard approach for DINOv3)
    if cfg.FREEZE_BACKBONE:
        model.freeze_backbone()
        logger.info("🔒 Backbone frozen - only training fusion/BN-neck/classifier")
    
    # Create loss function
    logger.info("🎯 Creating loss function...")
    loss_fn = make_loss(
        num_classes=num_classes,
        feat_dim=model.get_feature_dim(),
        id_loss_weight=cfg.ID_LOSS_WEIGHT,
        triplet_loss_weight=cfg.TRIPLET_LOSS_WEIGHT,
        triplet_margin=cfg.TRIPLET_MARGIN,
        label_smoothing=cfg.LABEL_SMOOTHING,
        use_center_loss=cfg.USE_CENTER_LOSS,
        center_loss_weight=cfg.CENTER_LOSS_WEIGHT
    )
    
    # Get center loss criterion
    center_criterion = None
    if cfg.USE_CENTER_LOSS:
        center_criterion = loss_fn.center_criterion
    
    # Create optimizer
    logger.info("⚙️  Creating optimizer...")
    optimizer, optimizer_center = make_optimizer(
        cfg,
        model,
        center_criterion,
        freeze_backbone=cfg.FREEZE_BACKBONE
    )
    
    # Create scheduler
    logger.info("📈 Creating scheduler...")
    scheduler = WarmupMultiStepLR(
        optimizer,
        milestones=cfg.STEPS,
        gamma=cfg.GAMMA,
        warmup_factor=cfg.WARMUP_FACTOR,
        warmup_iters=cfg.WARMUP_ITERS,
        warmup_method=cfg.WARMUP_METHOD
    )
    
    start_epoch = 1
    best_mAP = 0.0
    
    # Resume from checkpoint
    if args.resume:
        logger.info(f"📂 Resuming from checkpoint: {args.resume}")
        checkpoint = torch.load(args.resume, map_location='cpu', weights_only=False)
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler'])
        start_epoch = checkpoint['epoch'] + 1
        best_mAP = checkpoint.get('best_mAP', 0.0)
        logger.info(f"   Resumed from epoch {start_epoch-1}, best mAP: {best_mAP:.1%}")
    
    # Evaluation only mode
    if args.eval_only:
        logger.info("🔍 Evaluation only mode")
        cmc, mAP = do_inference(cfg, model, val_query_loader, val_gallery_loader)
        logger.info("Evaluation Results:")
        logger.info("mAP: {:.1%}".format(mAP))
        for r in [1, 5, 10]:
            logger.info("CMC curve, Rank-{:<3}:{:.1%}".format(r, cmc[r - 1]))
        return
    
    # Start training
    logger.info("🚀 Starting training loop...")
    do_train(
        cfg=cfg,
        model=model,
        center_criterion=center_criterion,
        train_loader=train_loader,
        query_loader=val_query_loader,
        gallery_loader=val_gallery_loader,
        optimizer=optimizer,
        optimizer_center=optimizer_center,
        scheduler=scheduler,
        loss_fn=loss_fn,
        start_epoch=start_epoch
    )
    
    logger.info("✅ Training completed!")


if __name__ == '__main__':
    print("🎯 Starting Regional Dog ReID Training with DINOv3-L")
    print(f"📁 Output: {cfg.OUTPUT_DIR}")
    print(f"🧠 Backbone: {cfg.BACKBONE} (FROZEN)")
    print(f"🔗 Regional features: 1 global + 7 face regions")
    print(f"📊 Final embedding: {cfg.EMBED_DIM}-dim")
    print("=" * 60)
    
    main()

