"""
Test script for regional dataloader.
Verifies that data loading works correctly with regions.
"""

import sys
import torch


def test_regional_dataloader():
    """Test the regional dataloader."""
    
    print("Testing Regional DataLoader")
    print("="*60)
    
    # Import after adding to path
    from datasets.dog_face_regional_dataset import DogFaceRegional
    from datasets.make_dataloader_regional import RegionalCollator
    from torchvision import transforms
    
    # Simple config for testing
    class SimpleConfig:
        ROOT_DIR = "/home/sagemaker-user/LostPet/LostPetTest"
        IMAGES_DIR = "/home/sagemaker-user/LostPet/PetFace/dog"
        TRAIN_SPLIT = "splits_petface_valid/train.csv"
        VAL_QUERY_SPLIT = "splits_petface_valid/val_query.csv"
        VAL_GALLERY_SPLIT = "splits_petface_valid/val_gallery.csv"
        PIXEL_MEAN = [0.485, 0.456, 0.406]
        PIXEL_STD = [0.229, 0.224, 0.225]
        USE_CAMID = False
    
    cfg = SimpleConfig()
    landmarks_dir = "/home/sagemaker-user/LostPet/dogface_landmark_estimation_hrcnn/petface_landmarks_json_all"
    
    # Create simple transforms
    transform_global = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=cfg.PIXEL_MEAN, std=cfg.PIXEL_STD)
    ])
    
    transform_regional = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize(mean=cfg.PIXEL_MEAN, std=cfg.PIXEL_STD)
    ])
    
    # Test 1: Create dataset
    print("\n1. Creating dataset...")
    try:
        dataset = DogFaceRegional(
            root=cfg.ROOT_DIR,
            split_csv=cfg.TRAIN_SPLIT,
            landmarks_dir=landmarks_dir,
            images_dir=cfg.IMAGES_DIR,
            transform_global=transform_global,
            transform_regional=transform_regional,
            use_camid=cfg.USE_CAMID,
            relabel=True
        )
        print(f"✅ Dataset created: {len(dataset)} images, {dataset.num_classes} classes")
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Test 2: Load a single sample
    print("\n2. Loading single sample...")
    try:
        img, regions, pid, camid, path = dataset[0]
        print(f"✅ Sample loaded:")
        print(f"   Global image: {img.shape}")
        print(f"   PID: {pid} (type: {type(pid).__name__})")
        print(f"   Regions: {list(regions.keys())}")
        for region_name, region_tensor in regions.items():
            print(f"     {region_name:12}: {region_tensor.shape}")
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Test 3: Create dataloader
    print("\n3. Creating dataloader...")
    try:
        collator = RegionalCollator()
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=4,
            shuffle=False,
            num_workers=0,
            collate_fn=collator
        )
        print(f"✅ DataLoader created")
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Test 4: Load a batch
    print("\n4. Loading one batch...")
    try:
        for batch in dataloader:
            imgs, regions_batch, pids, camids, paths = batch
            print(f"✅ Batch loaded:")
            print(f"   Batch size: {len(pids)}")
            print(f"   Global images: {imgs.shape}")
            print(f"   PIDs: {pids.tolist()}")
            print(f"   Regional batches:")
            for region_name, region_tensor in regions_batch.items():
                print(f"     {region_name:12}: {region_tensor.shape}")
            break
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return
    
    print("\n" + "="*60)
    print("✅ All tests passed!")
    print("="*60)
    print("\nDataloader returns:")
    print("  - imgs: [B, 3, 224, 224]")
    print("  - regions: Dict of region_name -> [B, 3, 64, 64]")
    print("  - pids: [B]")
    print("  - camids: [B]")
    print("  - paths: List of B strings")


if __name__ == "__main__":
    test_regional_dataloader()

