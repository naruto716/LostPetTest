"""
Example usage of PetFace dataloader.
Shows how to load and iterate through the PetFace dataset.
"""

from config_petface import cfg
from datasets.make_dataloader_petface import make_petface_dataloaders

def main():
    print("=" * 80)
    print("PetFace DataLoader Example")
    print("=" * 80)
    
    # Create dataloaders
    print("\n📦 Creating dataloaders...")
    (
        train_loader,
        val_query_loader,
        val_gallery_loader,
        test_query_loader,
        test_gallery_loader,
        num_classes
    ) = make_petface_dataloaders(cfg)
    
    # Print dataset statistics
    print(f"\n📊 Dataset Statistics:")
    print(f"  Training samples:      {len(train_loader.dataset):,}")
    print(f"  Val query samples:     {len(val_query_loader.dataset):,}")
    print(f"  Val gallery samples:   {len(val_gallery_loader.dataset):,}")
    print(f"  Test query samples:    {len(test_query_loader.dataset):,}")
    print(f"  Test gallery samples:  {len(test_gallery_loader.dataset):,}")
    print(f"  Number of classes:     {num_classes}")
    
    print(f"\n📦 Batch Configuration:")
    print(f"  Training batch size:   {cfg.IMS_PER_BATCH}")
    print(f"  Instances per ID (K):  {cfg.NUM_INSTANCE}")
    print(f"  IDs per batch (P):     {cfg.IMS_PER_BATCH // cfg.NUM_INSTANCE}")
    print(f"  Test batch size:       {cfg.TEST_BATCH_SIZE}")
    
    # Show example batch from training
    print("\n🔍 Sample Training Batch:")
    for imgs, pids, camids, paths in train_loader:
        print(f"  Image tensor shape:    {imgs.shape}")
        print(f"  PIDs shape:            {pids.shape}")
        print(f"  PIDs (first 10):       {pids[:10].tolist()}")
        print(f"  Unique PIDs in batch:  {len(pids.unique())}")
        print(f"  CamIDs (all zeros):    {camids[:10].tolist()}")
        print(f"  Example paths:")
        for i, path in enumerate(paths[:3]):
            print(f"    [{i}] {path}")
        break
    
    # Show example batch from validation query
    print("\n🔍 Sample Validation Query Batch:")
    for imgs, pids, camids, paths in val_query_loader:
        print(f"  Image tensor shape:    {imgs.shape}")
        print(f"  PIDs shape:            {pids.shape}")
        print(f"  PIDs (first 10):       {pids[:10].tolist()}")
        print(f"  Example paths:")
        for i, path in enumerate(paths[:3]):
            print(f"    [{i}] {path}")
        break
    
    print("\n✅ DataLoader working correctly!")
    print("\nNext steps:")
    print("  1. Training: Use train_loader with PK sampling for triplet loss")
    print("  2. Validation: Use val_query_loader + val_gallery_loader during training")
    print("  3. Final Test: Use test_query_loader + test_gallery_loader after training")
    print("=" * 80)

if __name__ == "__main__":
    main()

