"""
Test script for the complete Dog ReID training pipeline.
Verifies all components work together without full training.
"""

import torch
import os
from config_training import cfg, print_config
from datasets import make_dataloaders
from model import make_model
from loss import make_loss
from solver import make_optimizer, WarmupMultiStepLR
from utils import setup_logger, AverageMeter, R1_mAP_eval

def test_training_pipeline():
    """Test the complete training pipeline."""
    
    print("🧪 Testing Dog ReID Training Pipeline")
    print("=" * 60)
    
    # Print configuration
    print_config()
    
    # Test 1: Data loading
    print("\n📊 Test 1: Data Loading")
    try:
        train_loader, query_loader, gallery_loader, num_classes = make_dataloaders(cfg)
        print(f"   ✅ Data loaders created successfully")
        print(f"   Training batches: {len(train_loader)}")
        print(f"   Query samples: {len(query_loader.dataset)}")
        print(f"   Gallery samples: {len(gallery_loader.dataset)}")
        print(f"   Number of classes: {num_classes}")
        
        # Test batch loading
        batch = next(iter(train_loader))
        img, pid, camid, paths = batch
        print(f"   Sample batch shape: {img.shape}")
        print(f"   Sample PIDs: {pid[:8].tolist()}")
        
    except Exception as e:
        print(f"   ❌ Data loading failed: {e}")
        return False
    
    # Test 2: Model creation
    print("\n🏗️  Test 2: Model Creation")
    try:
        model = make_model(
            backbone_name=cfg.BACKBONE,
            num_classes=num_classes,
            embed_dim=cfg.EMBED_DIM,
            pretrained=cfg.PRETRAINED,
            bn_neck=cfg.BN_NECK
        )
        print(f"   ✅ Model created successfully")
        print(f"   Feature dimension: {model.get_feature_dim()}")
        
    except Exception as e:
        print(f"   ❌ Model creation failed: {e}")
        return False
    
    # Test 3: Loss function  
    print("\n🎯 Test 3: Loss Function")
    try:
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
        print(f"   ✅ Loss function created successfully")
        
    except Exception as e:
        print(f"   ❌ Loss function creation failed: {e}")
        return False
    
    # Test 4: Optimizer and scheduler
    print("\n⚙️  Test 4: Optimizer & Scheduler")
    try:
        center_criterion = loss_fn.center_criterion if cfg.USE_CENTER_LOSS else None
        optimizer, optimizer_center = make_optimizer(
            cfg, model, center_criterion, freeze_backbone=cfg.FREEZE_BACKBONE
        )
        
        scheduler = WarmupMultiStepLR(
            optimizer,
            milestones=cfg.STEPS,
            gamma=cfg.GAMMA,
            warmup_factor=cfg.WARMUP_FACTOR,
            warmup_iters=cfg.WARMUP_ITERS,
            warmup_method=cfg.WARMUP_METHOD
        )
        print(f"   ✅ Optimizer and scheduler created successfully")
        print(f"   Optimizer: {cfg.OPTIMIZER_NAME}")
        print(f"   Initial LR: {optimizer.param_groups[0]['lr']}")
        
    except Exception as e:
        print(f"   ❌ Optimizer/scheduler creation failed: {e}")
        return False
    
    # Test 5: Forward pass and loss computation
    print("\n🔄 Test 5: Forward Pass & Loss Computation")
    try:
        model.train()
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model.to(device)
        
        # Get a small batch
        img, pid, camid, _ = next(iter(train_loader))
        img = img[:8].to(device)  # Use smaller batch for testing
        pid = pid[:8].to(device)
        
        # Forward pass
        logits, features = model(img, return_mode='auto')
        print(f"   ✅ Forward pass successful")
        print(f"   Logits shape: {logits.shape}")
        print(f"   Features shape: {features.shape}")
        
        # Loss computation
        loss, loss_dict = loss_fn(logits, features, pid)
        print(f"   ✅ Loss computation successful")
        print(f"   Total loss: {loss.item():.4f}")
        for name, value in loss_dict.items():
            print(f"   {name}: {value:.4f}")
        
    except Exception as e:
        print(f"   ❌ Forward pass/loss failed: {e}")
        return False
    
    # Test 6: Backward pass and optimizer step
    print("\n🔄 Test 6: Backward Pass & Optimizer Step")
    try:
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()
        
        print(f"   ✅ Backward pass and optimizer step successful")
        print(f"   LR after step: {scheduler.get_lr()[0]:.6f}")
        
    except Exception as e:
        print(f"   ❌ Backward pass/optimizer step failed: {e}")
        return False
    
    # Test 7: Evaluation setup
    print("\n🔍 Test 7: Evaluation Setup")
    try:
        # Create combined validation loader
        val_dataset = torch.utils.data.ConcatDataset([query_loader.dataset, gallery_loader.dataset])
        val_loader = torch.utils.data.DataLoader(
            val_dataset, batch_size=32, shuffle=False, num_workers=0,
            collate_fn=query_loader.collate_fn
        )
        num_query = len(query_loader.dataset)
        
        # Test evaluation
        evaluator = R1_mAP_eval(num_query, max_rank=50, feat_norm=cfg.FEAT_NORM)
        evaluator.reset()
        
        model.eval()
        with torch.no_grad():
            for i, (img, pid, camid, _) in enumerate(val_loader):
                if i >= 2:  # Only test first 2 batches
                    break
                img = img.to(device)
                feat = model(img, return_mode='features')
                evaluator.update((feat, pid, camid))
        
        print(f"   ✅ Evaluation setup successful")
        print(f"   Validation batches tested: 2")
        
    except Exception as e:
        print(f"   ❌ Evaluation setup failed: {e}")
        return False
    
    # Test 8: Logger and utilities
    print("\n📝 Test 8: Logger & Utilities")
    try:
        # Test output directory creation
        output_dir = "./outputs/pipeline_test"
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        # Test logger
        logger = setup_logger("test", output_dir, if_train=True)
        logger.info("Test log message")
        
        # Test meter
        meter = AverageMeter()
        meter.update(1.5, 2)
        meter.update(2.5, 3)
        
        print(f"   ✅ Logger and utilities working")
        print(f"   Average meter test: {meter.avg:.2f}")
        
    except Exception as e:
        print(f"   ❌ Logger/utilities failed: {e}")
        return False
    
    print("\n🎉 ALL TESTS PASSED!")
    print("=" * 60)
    print("✅ Training pipeline is ready!")
    print("🚀 You can now run: python train_dogreid.py")
    print("=" * 60)
    
    return True

if __name__ == '__main__':
    success = test_training_pipeline()
    if not success:
        print("\n❌ Pipeline test failed. Please fix the issues above.")
        exit(1)
    print("\n🎯 Ready for training on your server!")
