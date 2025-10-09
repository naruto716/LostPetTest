"""
Training and evaluation processor for Regional Dog ReID.
Based on processor_dogreid.py with modifications for regional features.
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
    Main training loop for Regional Dog ReID.
    
    Key difference from processor_dogreid: handles (img, regions, pid, camid, path) batches
    """
    log_period = cfg.LOG_PERIOD
    checkpoint_period = cfg.CHECKPOINT_PERIOD
    eval_period = cfg.EVAL_PERIOD
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    epochs = cfg.MAX_EPOCHS
    
    logger = logging.getLogger()
    logger.info('🚀 Starting Regional Dog ReID training')
    
    # Setup model
    model.to(device)
    
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
    
    # Mixed precision scaler
    scaler = amp.GradScaler()
    
    # Training loop
    all_start_time = time.monotonic()
    best_mAP = 0.0
    
    # Zero-shot evaluation before training
    if start_epoch == 1:
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
        
        # KEY CHANGE: Unpack regions from batch
        for n_iter, (img, regions, pid, camid, _) in enumerate(train_loader):
            if n_iter == 0:
                print(f"✅ First batch loaded! (batch size: {img.shape[0]}, unique PIDs: {len(set(pid.tolist()))})")
                logger.info(f"✅ First batch loaded! (batch size: {img.shape[0]})")
            
            optimizer.zero_grad()
            if optimizer_center is not None:
                optimizer_center.zero_grad()
            
            # Move to device
            img = img.to(device)
            target = pid.to(device)
            
            # Move regions to device
            regions = {name: region_tensor.to(device) for name, region_tensor in regions.items()}
            
            if n_iter == 0:
                print("   Running forward pass with regions...")
            
            with amp.autocast():
                # KEY CHANGE: Pass both img and regions to model
                if skip_dataparallel:
                    logits, features = model(img, regions, return_mode='auto')
                else:
                    logits, features = model(img, regions, 'auto')
                
                if n_iter == 0:
                    print(f"   Forward pass done! logits: {logits.shape}, features: {features.shape}")
                    print("   Computing loss...")
                
                # Compute loss (same as baseline)
                loss, loss_dict = loss_fn(logits, features, target)
                
                if n_iter == 0:
                    print(f"   Loss computed! loss: {loss.item():.4f}")
            
            # Backward pass (same as baseline)
            if n_iter == 0:
                print("   Running backward pass...")
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            # Center loss step
            if center_criterion is not None and optimizer_center is not None:
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
            
            # Logging
            if (n_iter + 1) % log_period == 0:
                elapsed_time = time.time() - start_time
                time_per_batch = elapsed_time / (n_iter + 1)
                eta = time_per_batch * (len(train_loader) - n_iter - 1)
                
                logger.info(f"Epoch[{epoch}] Iteration[{n_iter + 1}/{len(train_loader)}] "
                          f"Loss: {loss_meter.avg:.4f}, Acc: {acc_meter.avg:.3f}, "
                          f"Base Lr: {scheduler.get_lr()[0]:.2e}, "
                          f"ETA: {timedelta(seconds=int(eta))}")
        
        # End of epoch
        epoch_time = time.time() - start_time
        time_per_batch = epoch_time / len(train_loader)
        
        logger.info("\n" + "="*60)
        logger.info(f"✅ Epoch {epoch} Training Complete")
        logger.info(f"   Time: {epoch_time:.1f}s ({epoch_time/60:.1f} min)")
        logger.info(f"   Avg Loss: {loss_meter.avg:.4f}")
        logger.info(f"   Avg Acc: {acc_meter.avg:.3%}")
        logger.info(f"   Speed: {train_loader.batch_size / time_per_batch:.1f} samples/s")
        logger.info("="*60)
        
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
            
            cmc, mAP = do_inference(cfg, model, query_loader, gallery_loader)
            val_time = time.time() - val_start
            
            # Print results
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
            logger.info(f"   Rank-1: {cmc[0]:.2%}")
            logger.info(f"   Rank-5: {cmc[4]:.2%}")
            logger.info(f"   Rank-10: {cmc[9]:.2%}")
            logger.info(f"   Time: {val_time:.1f}s")
            logger.info("📊"*30)
            
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
        
        # Step scheduler
        scheduler.step()
    
    all_end_time = time.monotonic()
    total_time = timedelta(seconds=all_end_time - all_start_time)
    logger.info("Total running time: {}".format(total_time))
    logger.info(f"🏆 Best mAP achieved: {best_mAP:.1%}")


def do_inference(cfg, model, query_loader, gallery_loader):
    """
    Perform inference and evaluation for regional model.
    
    Key difference: handles (img, regions, pid, camid, path) batches
    """
    print(">>> do_inference() called for regional model")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger = logging.getLogger()
    print(f">>> Device: {device}, Query batches: {len(query_loader)}, Gallery batches: {len(gallery_loader)}")
    
    # Setup model
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
        # KEY CHANGE: Unpack regions from batch
        for n_iter, (img, regions, pid, camid, _) in enumerate(query_loader):
            if n_iter == 0:
                logger.info(f"   Loading first query batch...")
            if (n_iter + 1) % 10 == 0:
                logger.info(f"   Processed {n_iter + 1}/{len(query_loader)} query batches")
            
            img = img.to(device)
            regions = {name: region_tensor.to(device) for name, region_tensor in regions.items()}
            
            # KEY CHANGE: Pass both img and regions
            if skip_dataparallel:
                feat = model(img, regions, return_mode='features')
            else:
                feat = model(img, regions, 'features')
            
            query_features.append(feat.cpu())
            query_pids.extend(pid.cpu().numpy())
            query_camids.extend(camid.cpu().numpy())
    
    query_features = torch.cat(query_features, dim=0)
    query_pids = np.asarray(query_pids)
    query_camids = np.asarray(query_camids)
    
    logger.info(f"   ✅ Query features extracted: {query_features.shape}")
    
    # Extract gallery features
    logger.info("📊 Extracting gallery features...")
    logger.info(f"   Gallery set: {len(gallery_loader.dataset)} samples, {len(gallery_loader)} batches")
    gallery_features, gallery_pids, gallery_camids = [], [], []
    
    with torch.no_grad():
        # KEY CHANGE: Unpack regions from batch
        for n_iter, (img, regions, pid, camid, _) in enumerate(gallery_loader):
            if n_iter == 0:
                logger.info(f"   Loading first gallery batch...")
            if (n_iter + 1) % 20 == 0:
                logger.info(f"   Processed {n_iter + 1}/{len(gallery_loader)} gallery batches")
            
            img = img.to(device)
            regions = {name: region_tensor.to(device) for name, region_tensor in regions.items()}
            
            # KEY CHANGE: Pass both img and regions
            if skip_dataparallel:
                feat = model(img, regions, return_mode='features')
            else:
                feat = model(img, regions, 'features')
            
            gallery_features.append(feat.cpu())
            gallery_pids.extend(pid.cpu().numpy())
            gallery_camids.extend(camid.cpu().numpy())
    
    gallery_features = torch.cat(gallery_features, dim=0)
    gallery_pids = np.asarray(gallery_pids)
    gallery_camids = np.asarray(gallery_camids)
    
    logger.info(f"   ✅ Gallery features extracted: {gallery_features.shape}")
    
    # Compute metrics (same as baseline)
    print("Computing distance matrix and ranking...")
    logger.info("📊 Computing metrics...")
    
    evaluator = R1_mAP_eval(len(query_pids), max_rank=50, feat_norm='yes')
    evaluator.reset()
    evaluator.update((query_features, query_pids, query_camids, gallery_features, gallery_pids, gallery_camids))
    cmc, mAP = evaluator.compute()
    
    logger.info("✅ Evaluation complete")
    
    return cmc, mAP

