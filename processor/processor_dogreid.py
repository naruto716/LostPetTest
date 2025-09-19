"""
Training and evaluation processor for Dog ReID.
Based on CLIP-ReID's processor_clipreid_stage2.py with adaptations.
"""

import logging
import os
import time
from datetime import timedelta
import torch
import torch.nn as nn
from torch.cuda import amp
from utils.meter import AverageMeter
from utils.metrics import R1_mAP_eval

def do_train(
    cfg,
    model,
    center_criterion,
    train_loader,
    val_loader,
    optimizer,
    optimizer_center,
    scheduler,
    loss_fn,
    num_query,
    start_epoch=1
):
    """
    Main training loop for Dog ReID.
    
    Args:
        cfg: Configuration object
        model: Model to train
        center_criterion: Center loss criterion (can be None)
        train_loader: Training data loader
        val_loader: Validation data loader (query + gallery)
        optimizer: Main optimizer
        optimizer_center: Center loss optimizer (can be None)
        scheduler: Learning rate scheduler
        loss_fn: Loss function
        num_query: Number of query samples in val_loader
        start_epoch: Starting epoch number
    """
    log_period = cfg.LOG_PERIOD
    checkpoint_period = cfg.CHECKPOINT_PERIOD
    eval_period = cfg.EVAL_PERIOD
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    epochs = cfg.MAX_EPOCHS
    
    logger = logging.getLogger("dogreid.train")
    logger.info('🚀 Starting Dog ReID training')
    
    # Setup model for training
    model.to(device)
    if torch.cuda.device_count() > 1:
        print(f'🔧 Using {torch.cuda.device_count()} GPUs for training with DataParallel')
        model = nn.DataParallel(model)
    else:
        print(f'🔧 Using single GPU: {device}')
        
    # Training meters
    loss_meter = AverageMeter()
    acc_meter = AverageMeter()
    
    # Evaluator
    evaluator = R1_mAP_eval(num_query, max_rank=50, feat_norm=cfg.FEAT_NORM)
    
    # Mixed precision scaler
    scaler = amp.GradScaler('cuda')
    
    # Training loop
    all_start_time = time.monotonic()
    best_mAP = 0.0
    
    for epoch in range(start_epoch, epochs + 1):
        start_time = time.time()
        loss_meter.reset()
        acc_meter.reset()
        evaluator.reset()
        
        # Training phase
        model.train()
        for n_iter, (img, pid, camid, _) in enumerate(train_loader):
            optimizer.zero_grad()
            if optimizer_center is not None:
                optimizer_center.zero_grad()
                
            img = img.to(device)
            target = pid.to(device)
            
            with amp.autocast(enabled=True):
                # Forward pass
                logits, features = model(img, return_mode='auto')
                
                # Compute loss
                loss, loss_dict = loss_fn(logits, features, target)
            
            # Backward pass
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            # Center loss step
            if center_criterion is not None and optimizer_center is not None:
                # Center loss updates
                for param in center_criterion.parameters():
                    if param.grad is not None:
                        param.grad.data *= (1. / cfg.CENTER_LOSS_WEIGHT)
                scaler.step(optimizer_center)
                scaler.update()
            
            # Compute accuracy
            acc = (logits.max(1)[1] == target).float().mean()
            
            # Update meters
            loss_meter.update(loss.item(), img.shape[0])
            acc_meter.update(acc.item(), 1)
            
            torch.cuda.synchronize()
            if (n_iter + 1) % log_period == 0:
                logger.info(
                    "Epoch[{}] Iteration[{}/{}] Loss: {:.3f}, Acc: {:.3f}, Base Lr: {:.2e}".format(
                        epoch, (n_iter + 1), len(train_loader),
                        loss_meter.avg, acc_meter.avg, scheduler.get_lr()[0]
                    )
                )
                
                # Log individual loss components
                if isinstance(loss_dict, dict):
                    loss_components = ", ".join([f"{k}: {v:.3f}" for k, v in loss_dict.items()])
                    logger.info(f"   Loss components: {loss_components}")
        
        end_time = time.time()
        time_per_batch = (end_time - start_time) / (n_iter + 1)
        logger.info(
            "Epoch {} done. Time per batch: {:.3f}[s] Speed: {:.1f}[samples/s]".format(
                epoch, time_per_batch, train_loader.batch_size / time_per_batch
            )
        )
        
        # Save checkpoint
        if epoch % checkpoint_period == 0:
            checkpoint = {
                'model': model.module.state_dict() if hasattr(model, 'module') else model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
                'epoch': epoch,
                'best_mAP': best_mAP
            }
            
            checkpoint_path = os.path.join(cfg.OUTPUT_DIR, f'checkpoint_epoch_{epoch}.pth')
            torch.save(checkpoint, checkpoint_path)
            logger.info(f"Checkpoint saved: {checkpoint_path}")
        
        # Evaluation
        if epoch % eval_period == 0:
            logger.info(f"🔍 Evaluating at epoch {epoch}")
            cmc, mAP = do_inference(cfg, model, val_loader, num_query)
            
            logger.info("Validation Results - Epoch: {}".format(epoch))
            logger.info("mAP: {:.1%}".format(mAP))
            for r in [1, 5, 10]:
                logger.info("CMC curve, Rank-{:<3}:{:.1%}".format(r, cmc[r - 1]))
            
            # Save best model
            if mAP > best_mAP:
                best_mAP = mAP
                best_checkpoint = {
                    'model': model.module.state_dict() if hasattr(model, 'module') else model.state_dict(),
                    'epoch': epoch,
                    'mAP': mAP,
                    'cmc': cmc
                }
                
                best_path = os.path.join(cfg.OUTPUT_DIR, 'best_model.pth')
                torch.save(best_checkpoint, best_path)
                logger.info(f"🏆 New best model saved: mAP={mAP:.1%} at epoch {epoch}")
            
            torch.cuda.empty_cache()
        
        # Step scheduler at end of epoch
        scheduler.step()
    
    all_end_time = time.monotonic()
    total_time = timedelta(seconds=all_end_time - all_start_time)
    logger.info("Total running time: {}".format(total_time))
    logger.info(f"🏆 Best mAP achieved: {best_mAP:.1%}")

def do_inference(cfg, model, val_loader, num_query):
    """
    Perform inference and evaluation.
    
    Args:
        cfg: Configuration object  
        model: Model to evaluate
        val_loader: Validation data loader (query + gallery)
        num_query: Number of query samples
    Returns:
        cmc: Cumulative matching characteristics
        mAP: Mean average precision  
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger = logging.getLogger("dogreid.test")
    
    evaluator = R1_mAP_eval(num_query, max_rank=50, feat_norm=cfg.FEAT_NORM)
    evaluator.reset()
    
    # Setup model for evaluation
    if torch.cuda.device_count() > 1 and not isinstance(model, nn.DataParallel):
        print(f'Wrapping model with DataParallel for evaluation')
        model = nn.DataParallel(model)
    model.to(device)
    model.eval()
    
    with torch.no_grad():
        for n_iter, (img, pid, camid, _) in enumerate(val_loader):
            img = img.to(device)
            
            # Extract features
            feat = model(img, return_mode='features')
            evaluator.update((feat, pid, camid))
    
    # Compute metrics
    cmc, mAP, _, _, _, _, _ = evaluator.compute()
    
    return cmc, mAP
