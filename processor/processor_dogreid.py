"""
Training and evaluation processor for Dog ReID.
Based on CLIP-ReID's processor_clipreid_stage2.py with adaptations.
"""

import logging
import os
import time
from datetime import timedelta

import numpy as np
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
    query_loader,
    gallery_loader,
    optimizer,
    optimizer_center,
    scheduler,
    loss_fn,
    start_epoch=1
):
    """
    Main training loop for Dog ReID.
    
    Args:
        cfg: Configuration object
        model: Model to train
        center_criterion: Center loss criterion (can be None)
        train_loader: Training data loader
        query_loader: Query data loader for evaluation
        gallery_loader: Gallery data loader for evaluation
        optimizer: Main optimizer
        optimizer_center: Center loss optimizer (can be None)
        scheduler: Learning rate scheduler
        loss_fn: Loss function
        start_epoch: Starting epoch number
    """
    log_period = cfg.LOG_PERIOD
    checkpoint_period = cfg.CHECKPOINT_PERIOD
    eval_period = cfg.EVAL_PERIOD
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    epochs = cfg.MAX_EPOCHS
    
    # Use root logger to inherit handlers from train script setup
    logger = logging.getLogger()
    logger.info('🚀 Starting Dog ReID training')
    
    # Setup model for training
    logger.info(f'📦 Moving model to {device}...')
    if device == "cuda":
        logger.info(f'   GPU: {torch.cuda.get_device_name(0)}')
        logger.info(f'   GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB')
    
    import time
    start_time = time.time()
    model.to(device)
    elapsed = time.time() - start_time
    logger.info(f'✅ Model moved to {device} in {elapsed:.1f}s')
    
    # Check if we should skip DataParallel (for models with hooks like SWIN)
    backbone_name = getattr(cfg, 'BACKBONE', '')
    skip_dataparallel = 'swin' in backbone_name.lower() and 'multilevel' in backbone_name.lower()
    
    if torch.cuda.device_count() > 1 and not skip_dataparallel:
        print(f'🔧 Using {torch.cuda.device_count()} GPUs for training with DataParallel')
        model = nn.DataParallel(model)
    else:
        if skip_dataparallel:
            print(f'🔧 Using single GPU: {device} (DataParallel disabled for multi-level SWIN)')
        else:
            print(f'🔧 Using single GPU: {device}')
        
    # Training meters
    loss_meter = AverageMeter()
    acc_meter = AverageMeter()
    
    # Note: Evaluator is now handled in do_inference with separate loaders
    
    # Mixed precision scaler
    scaler = amp.GradScaler()
    
    # Training loop
    all_start_time = time.monotonic()
    best_mAP = 0.0
    
    # 🔍 Zero-shot evaluation before training starts
    if start_epoch == 1:  # Only on fresh training
        logger.info("🚀 Zero-shot evaluation (before training)...")
        cmc, mAP = do_inference(cfg, model, query_loader, gallery_loader)
        
        logger.info("Zero-shot Results (Epoch 0)")
        logger.info("mAP: {:.1%}".format(mAP))
        for r in [1, 5, 10]:
            logger.info("CMC curve, Rank-{:<3}:{:.1%}".format(r, cmc[r - 1]))
        logger.info("=" * 50)
        torch.cuda.empty_cache()
    
    print("\n" + "="*80)
    print("🚀 STARTING TRAINING")
    print("="*80)
    logger.info(f"Starting training for {epochs - start_epoch + 1} epochs")
    logger.info(f"Training set: {len(train_loader.dataset)} images, {len(train_loader)} batches")
    logger.info(f"Logging every {log_period} iterations")
    print("About to enter training loop...")
    logger.info("About to enter training loop...")
    
    for epoch in range(start_epoch, epochs + 1):
        logger.info(f"\n{'='*60}")
        logger.info(f"Epoch {epoch}/{epochs}")
        logger.info(f"{'='*60}")
        start_time = time.time()
        loss_meter.reset()
        acc_meter.reset()
        
        # Training phase
        model.train()
        print(f"[Epoch {epoch}] Starting to iterate through training data...")
        logger.info("Starting to iterate through training data...")
        
        for n_iter, (img, pid, camid, _) in enumerate(train_loader):
            # Log first iteration to confirm training started
            if n_iter == 0:
                print(f"✅ First batch loaded! (batch size: {img.shape[0]}, unique PIDs: {len(set(pid.tolist()))})")
                logger.info(f"✅ First batch loaded! (batch size: {img.shape[0]})")
            
            optimizer.zero_grad()
            if optimizer_center is not None:
                optimizer_center.zero_grad()
                
            img = img.to(device)
            target = pid.to(device)
            
            if n_iter == 0:
                print("   Running forward pass...")
            
            with amp.autocast():
                # Forward pass
                if skip_dataparallel:
                    logits, features = model(img, return_mode='auto')  # Keyword args for single GPU
                else:
                    logits, features = model(img, 'auto')  # Positional args for DataParallel
                
                if n_iter == 0:
                    print(f"   Forward pass done! logits: {logits.shape}, features: {features.shape}")
                    print("   Computing loss...")
                
                # Compute loss
                loss, loss_dict = loss_fn(logits, features, target)
                
                if n_iter == 0:
                    print(f"   Loss computed! loss: {loss.item():.4f}")
            
            # Backward pass
            if n_iter == 0:
                print("   Running backward pass...")
            
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
            
            if n_iter == 0:
                print(f"✅ First iteration complete! Loss: {loss.item():.4f}, Acc: {acc.item():.3%}")
            
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
        epoch_time = end_time - start_time
        
        logger.info("\n" + "="*60)
        logger.info(f"✅ Epoch {epoch} Training Complete")
        logger.info(f"   Time: {epoch_time:.1f}s ({epoch_time/60:.1f} min)")
        logger.info(f"   Avg Loss: {loss_meter.avg:.4f}")
        logger.info(f"   Avg Acc: {acc_meter.avg:.3%}")
        logger.info(f"   Speed: {train_loader.batch_size / time_per_batch:.1f} samples/s")
        logger.info("="*60)
        
        print(f"\n>>> EPOCH {epoch} DONE. eval_period={eval_period}, will validate: {epoch % eval_period == 0}")
        
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
            print("\n" + "="*80)
            print(f"🔍 STARTING VALIDATION AT EPOCH {epoch}")
            print("="*80)
            logger.info("\n" + "🔍"*30)
            logger.info(f"🔍 VALIDATION - Epoch {epoch}")
            logger.info("🔍"*30)
            val_start = time.time()
            print("Calling do_inference...")
            cmc, mAP = do_inference(cfg, model, query_loader, gallery_loader)
            print(f"do_inference completed! mAP: {mAP:.2%}")
            val_time = time.time() - val_start
            
            # Print results (guaranteed to show)
            print("\n" + "="*80)
            print(f"📊 VALIDATION RESULTS - Epoch {epoch}")
            print("="*80)
            print(f"   mAP:      {mAP:.2%}")
            print(f"   Rank-1:   {cmc[0]:.2%}")
            print(f"   Rank-5:   {cmc[4]:.2%}")
            print(f"   Rank-10:  {cmc[9]:.2%}")
            print(f"   Eval time: {val_time:.1f}s")
            print("="*80 + "\n")
            
            logger.info("\n" + "📊"*30)
            logger.info(f"📊 Validation Results - Epoch {epoch}")
            logger.info(f"   mAP: {mAP:.2%}")
            logger.info(f"   Rank-1:  {cmc[0]:.2%}")
            logger.info(f"   Rank-5:  {cmc[4]:.2%}")
            logger.info(f"   Rank-10: {cmc[9]:.2%}")
            logger.info(f"   Eval time: {val_time:.1f}s")
            logger.info("📊"*30 + "\n")
            
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

def do_inference(cfg, model, query_loader, gallery_loader):
    """
    Perform inference and evaluation using separate query and gallery loaders.
    
    Args:
        cfg: Configuration object  
        model: Model to evaluate
        query_loader: Query data loader 
        gallery_loader: Gallery data loader
    Returns:
        cmc: Cumulative matching characteristics
        mAP: Mean average precision  
    """
    print(">>> do_inference() called")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # Use root logger to inherit handlers from train script setup
    logger = logging.getLogger()
    print(f">>> Device: {device}, Query batches: {len(query_loader)}, Gallery batches: {len(gallery_loader)}")
    
    # Setup model for evaluation
    # Check if we should skip DataParallel (for models with hooks like SWIN)
    backbone_name = getattr(cfg, 'BACKBONE', '')
    skip_dataparallel = 'swin' in backbone_name.lower() and 'multilevel' in backbone_name.lower()
    
    if torch.cuda.device_count() > 1 and not isinstance(model, nn.DataParallel) and not skip_dataparallel:
        print(f'Wrapping model with DataParallel for evaluation')
        model = nn.DataParallel(model)
    elif skip_dataparallel:
        print(f'Using single GPU for evaluation (DataParallel disabled for multi-level SWIN)')
        
    model.to(device)
    model.eval()
    
    # Extract query features
    logger.info("📊 Extracting query features...")
    logger.info(f"   Query set: {len(query_loader.dataset)} samples, {len(query_loader)} batches")
    query_features, query_pids, query_camids = [], [], []
    
    with torch.no_grad():
        for n_iter, (img, pid, camid, _) in enumerate(query_loader):
            if n_iter == 0:
                logger.info(f"   Loading first query batch...")
            if (n_iter + 1) % 10 == 0:
                logger.info(f"   Processed {n_iter + 1}/{len(query_loader)} query batches")
            
            img = img.to(device)
            if skip_dataparallel:
                feat = model(img, return_mode='features')  # Keyword args for single GPU  
            else:
                feat = model(img, 'features')  # Positional args for DataParallel
            
            query_features.append(feat.cpu())
            query_pids.extend(pid.numpy())
            query_camids.extend(camid.numpy())
    
    logger.info(f"✅ Query features extracted: {len(query_features)} batches")
    
    # Extract gallery features  
    logger.info("📊 Extracting gallery features...")
    logger.info(f"   Gallery set: {len(gallery_loader.dataset)} samples, {len(gallery_loader)} batches")
    gallery_features, gallery_pids, gallery_camids = [], [], []
    
    with torch.no_grad():
        for n_iter, (img, pid, camid, _) in enumerate(gallery_loader):
            if n_iter == 0:
                logger.info(f"   Loading first gallery batch...")
            if (n_iter + 1) % 20 == 0:
                logger.info(f"   Processed {n_iter + 1}/{len(gallery_loader)} gallery batches")
            
            img = img.to(device)
            if skip_dataparallel:
                feat = model(img, return_mode='features')  # Keyword args for single GPU
            else:
                feat = model(img, 'features')  # Positional args for DataParallel
            
            gallery_features.append(feat.cpu())
            gallery_pids.extend(pid.numpy())
            gallery_camids.extend(camid.numpy())
    
    logger.info(f"✅ Gallery features extracted: {len(gallery_features)} batches")
    
    # Concatenate all features
    qf = torch.cat(query_features, dim=0)
    gf = torch.cat(gallery_features, dim=0) 
    
    q_pids = np.array(query_pids)
    q_camids = np.array(query_camids)
    g_pids = np.array(gallery_pids)
    g_camids = np.array(gallery_camids)
    
    logger.info(f"Query: {len(qf)} samples, {len(set(q_pids))} identities")
    logger.info(f"Gallery: {len(gf)} samples, {len(set(g_pids))} identities")
    
    # Debug: Check PID overlap  
    query_set = set(q_pids)
    gallery_set = set(g_pids)
    overlap = query_set.intersection(gallery_set)
    logger.info(f"PID overlap: {len(overlap)} identities out of {len(query_set)} query IDs")
    
    # Compute distance matrix and evaluate
    logger.info("🧮 Computing distance matrix...")
    from utils.metrics import euclidean_distance, eval_func
    distmat = euclidean_distance(qf, gf)
    cmc, mAP = eval_func(distmat, q_pids, g_pids, q_camids, g_camids, max_rank=50)
    
    return cmc, mAP
