"""
Simple test script for PetFace - evaluate best model on test set.
"""

import torch
import argparse
from config_petface import cfg
from datasets.make_dataloader_petface import make_petface_dataloaders
from model import make_model
from processor import do_inference
from utils import setup_logger
import os

def main():
    parser = argparse.ArgumentParser(description="Test PetFace Model")
    parser.add_argument(
        '--test_query',
        default=None,
        help='Path to test query CSV (overrides config)'
    )
    parser.add_argument(
        '--test_gallery',
        default=None,
        help='Path to test gallery CSV (overrides config)'
    )
    parser.add_argument(
        '--model',
        default=None,
        help='Path to model checkpoint (default: output_petface/best_model.pth)'
    )
    args = parser.parse_args()
    
    # Override config if custom splits provided
    if args.test_query and args.test_gallery:
        print(f"📂 Using custom test splits:")
        print(f"   Query:   {args.test_query}")
        print(f"   Gallery: {args.test_gallery}")
        cfg.TEST_QUERY_SPLIT = args.test_query
        cfg.TEST_GALLERY_SPLIT = args.test_gallery
    # Setup
    logger = setup_logger("petface_test", cfg.OUTPUT_DIR, if_train=False)
    logger.info("🔍 PetFace Test Evaluation")
    
    # Load data
    logger.info("📊 Loading test data...")
    (_, _, _, 
     test_query_loader, 
     test_gallery_loader, 
     num_classes) = make_petface_dataloaders(cfg)
    
    logger.info(f"   Test query: {len(test_query_loader.dataset)} images")
    logger.info(f"   Test gallery: {len(test_gallery_loader.dataset)} images")
    
    # Load model
    logger.info("🏗️  Loading best model...")
    model = make_model(
        backbone_name=cfg.BACKBONE,
        num_classes=num_classes,
        embed_dim=cfg.EMBED_DIM,
        pretrained=cfg.PRETRAINED,
        bn_neck=cfg.BN_NECK
    )
    
    # Load checkpoint
    if args.model:
        best_model_path = args.model
    else:
        best_model_path = os.path.join(cfg.OUTPUT_DIR, 'best_model.pth')
    
    checkpoint = torch.load(best_model_path, map_location='cpu', weights_only=False)
    model.load_state_dict(checkpoint['model'])
    logger.info(f"✅ Loaded from {best_model_path}")
    logger.info(f"   Trained for {checkpoint.get('epoch', '?')} epochs")
    logger.info(f"   Best val mAP: {checkpoint.get('mAP', 0):.2%}")
    
    # Evaluate
    logger.info("\n" + "="*80)
    logger.info("🏁 Running test evaluation...")
    logger.info("="*80)
    
    cmc, mAP = do_inference(cfg, model, test_query_loader, test_gallery_loader)
    
    # Print results
    print("\n" + "="*80)
    print("📊 FINAL TEST RESULTS")
    print("="*80)
    print(f"   mAP:      {mAP:.2%}")
    print(f"   Rank-1:   {cmc[0]:.2%}")
    print(f"   Rank-5:   {cmc[4]:.2%}")
    print(f"   Rank-10:  {cmc[9]:.2%}")
    print("="*80 + "\n")
    
    logger.info("\n📊 Final Test Results:")
    logger.info(f"   mAP: {mAP:.1%}")
    for r in [1, 5, 10]:
        logger.info(f"   CMC Rank-{r}: {cmc[r-1]:.1%}")
    
    # Save results
    results = {
        'test_mAP': mAP,
        'test_cmc': cmc.tolist(),
        'epoch': checkpoint.get('epoch'),
        'val_mAP': checkpoint.get('mAP')
    }
    results_path = os.path.join(cfg.OUTPUT_DIR, 'test_results.pth')
    torch.save(results, results_path)
    logger.info(f"\n💾 Results saved to {results_path}")

if __name__ == '__main__':
    main()

