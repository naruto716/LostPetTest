"""
Training script for PetFace dataset with proper train/val/test separation.
"""

import argparse
import os
import random
import numpy as np
import torch
from config_petface import cfg
from datasets.make_dataloader_petface import make_petface_dataloaders
from model import make_model
from loss import make_loss
from solver import make_optimizer, WarmupMultiStepLR
from processor import do_train, do_inference
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
    """Print configuration."""
    print("\n" + "="*80)
    print("Configuration:")
    print("="*80)
    for key in dir(cfg):
        if not key.startswith('_'):
            print(f"  {key}: {getattr(cfg, key)}")
    print("="*80 + "\n")

def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description="PetFace ReID Training")
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
        "--eval_split",
        choices=['val', 'test'],
        default='val',
        help="Which split to evaluate (val or test)"
    )
    parser.add_argument(
        "--resume", 
        default=None, 
        help="Path to checkpoint to resume from"
    )
    parser.add_argument(
        "--final_test",
        action='store_true',
        help="Run final test evaluation after training"
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
    logger = setup_logger("petface", cfg.OUTPUT_DIR, if_train=True)
    logger.info("🐕 Starting PetFace ReID Training")
    logger.info(f"Saving model in: {cfg.OUTPUT_DIR}")
    
    # Print configuration
    print_config()
    logger.info(f"Arguments: {args}")
    
    # Create data loaders with proper train/val/test separation
    logger.info("📊 Creating data loaders...")
    (train_loader, 
     val_query_loader, 
     val_gallery_loader,
     test_query_loader, 
     test_gallery_loader, 
     num_classes) = make_petface_dataloaders(cfg)
    
    logger.info(f"   Training samples: {len(train_loader.dataset)}")
    logger.info(f"   Val query samples: {len(val_query_loader.dataset)}")
    logger.info(f"   Val gallery samples: {len(val_gallery_loader.dataset)}")
    logger.info(f"   Test query samples: {len(test_query_loader.dataset)}")
    logger.info(f"   Test gallery samples: {len(test_gallery_loader.dataset)}")
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
        checkpoint = torch.load(args.resume, map_location='cpu', weights_only=False)
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler'])
        start_epoch = checkpoint['epoch'] + 1
        best_mAP = checkpoint.get('best_mAP', 0.0)
        logger.info(f"   Resumed from epoch {start_epoch-1}, best mAP: {best_mAP:.1%}")
    
    # Evaluation only mode
    if args.eval_only:
        logger.info(f"🔍 Evaluation only mode on {args.eval_split} set")
        
        if args.eval_split == 'val':
            query_loader = val_query_loader
            gallery_loader = val_gallery_loader
        else:  # test
            query_loader = test_query_loader
            gallery_loader = test_gallery_loader
        
        cmc, mAP = do_inference(cfg, model, query_loader, gallery_loader)
        logger.info(f"Evaluation Results ({args.eval_split} set):")
        logger.info("mAP: {:.1%}".format(mAP))
        for r in [1, 5, 10]:
            logger.info("CMC curve, Rank-{:<3}:{:.1%}".format(r, cmc[r - 1]))
        return
    
    # Start training (use VALIDATION set during training)
    logger.info("🚀 Starting training loop...")
    logger.info("   Using VALIDATION set for model selection during training")
    logger.info("   Test set will be used only for final evaluation")
    
    do_train(
        cfg=cfg,
        model=model,
        center_criterion=center_criterion,
        train_loader=train_loader,
        query_loader=val_query_loader,      # Use VAL for checkpointing
        gallery_loader=val_gallery_loader,  # Use VAL for checkpointing
        optimizer=optimizer,
        optimizer_center=optimizer_center,
        scheduler=scheduler,
        loss_fn=loss_fn,
        start_epoch=start_epoch
    )
    
    logger.info("✅ Training completed!")
    
    # Final test evaluation (optional)
    if args.final_test:
        logger.info("\n" + "="*80)
        logger.info("🏁 Running FINAL evaluation on TEST set")
        logger.info("="*80)
        
        # Load best model
        best_model_path = os.path.join(cfg.OUTPUT_DIR, 'best_model.pth')
        if os.path.exists(best_model_path):
            logger.info(f"Loading best model from {best_model_path}")
            checkpoint = torch.load(best_model_path, map_location='cpu', weights_only=False)
            model.load_state_dict(checkpoint['model'])
        
        cmc, mAP = do_inference(cfg, model, test_query_loader, test_gallery_loader)
        logger.info("Final Test Results:")
        logger.info("mAP: {:.1%}".format(mAP))
        for r in [1, 5, 10]:
            logger.info("CMC curve, Rank-{:<3}:{:.1%}".format(r, cmc[r - 1]))
        
        # Save final test results
        final_results = {
            'test_mAP': mAP,
            'test_cmc': cmc,
        }
        torch.save(final_results, os.path.join(cfg.OUTPUT_DIR, 'final_test_results.pth'))
        logger.info(f"Final test results saved to {cfg.OUTPUT_DIR}/final_test_results.pth")

if __name__ == '__main__':
    main()

