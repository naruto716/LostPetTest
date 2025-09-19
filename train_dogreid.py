"""
Main training script for Dog ReID.
Based on CLIP-ReID's train_clipreid.py structure.
"""

import argparse
import os
import random
import numpy as np
import torch
from config_training import cfg, print_config
from datasets import make_dataloaders
from model import make_model
from loss import make_loss
from solver import make_optimizer, WarmupMultiStepLR
from processor import do_train
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
    """Main training function."""
    parser = argparse.ArgumentParser(description="Dog ReID Training")
    parser.add_argument(
        "--freeze_backbone", 
        action='store_true', 
        help="Freeze backbone during training"
    )
    parser.add_argument(
        "--output_dir", 
        default=None, 
        help="Output directory (overrides config)"
    )
    parser.add_argument(
        "--epochs", 
        type=int, 
        default=None, 
        help="Number of epochs (overrides config)"
    )
    parser.add_argument(
        "--eval_only", 
        action='store_true', 
        help="Only run evaluation with pretrained model"
    )
    parser.add_argument(
        "--resume", 
        default=None, 
        help="Path to checkpoint to resume from"
    )
    
    args = parser.parse_args()
    
    # Update config based on arguments
    if args.output_dir:
        cfg.OUTPUT_DIR = args.output_dir
    if args.epochs:
        cfg.MAX_EPOCHS = args.epochs
    if args.freeze_backbone:
        cfg.FREEZE_BACKBONE = True
    
    # Set random seed
    set_seed(42)
    
    # Create output directory
    if not os.path.exists(cfg.OUTPUT_DIR):
        os.makedirs(cfg.OUTPUT_DIR)
    
    # Setup logger
    logger = setup_logger("dogreid", cfg.OUTPUT_DIR, if_train=True)
    logger.info("🚀 Starting Dog ReID Training")
    logger.info(f"Saving model in: {cfg.OUTPUT_DIR}")
    
    # Print configuration
    print_config()
    logger.info(f"Arguments: {args}")
    
    # Create data loaders
    logger.info("📊 Creating data loaders...")
    train_loader, query_loader, gallery_loader, num_classes = make_dataloaders(cfg)
    
    # Combine query and gallery for evaluation
    val_dataset = torch.utils.data.ConcatDataset([query_loader.dataset, gallery_loader.dataset])
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=64,  # Larger batch for inference
        shuffle=False,
        num_workers=cfg.NUM_WORKERS,
        pin_memory=True,
        collate_fn=query_loader.collate_fn
    )
    num_query = len(query_loader.dataset)
    
    logger.info(f"   Training samples: {len(train_loader.dataset)}")
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
    
    # Resume from checkpoint if provided
    if args.resume:
        logger.info(f"📂 Resuming from checkpoint: {args.resume}")
        checkpoint = torch.load(args.resume, map_location='cpu')
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler'])
        start_epoch = checkpoint['epoch'] + 1
        best_mAP = checkpoint.get('best_mAP', 0.0)
        logger.info(f"   Resumed from epoch {start_epoch-1}, best mAP: {best_mAP:.1%}")
    
    # Evaluation only mode
    if args.eval_only:
        logger.info("🔍 Evaluation only mode")
        from processor import do_inference
        cmc, mAP = do_inference(cfg, model, val_loader, num_query)
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
        val_loader=val_loader,
        optimizer=optimizer,
        optimizer_center=optimizer_center,
        scheduler=scheduler,
        loss_fn=loss_fn,
        num_query=num_query,
        start_epoch=start_epoch
    )
    
    logger.info("✅ Training completed!")

if __name__ == '__main__':
    main()
