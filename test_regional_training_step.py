"""
Test one training step with regional processor.
Verifies the entire pipeline: dataloader → model → loss → backward.
"""

import torch
import sys


def test_training_step():
    """Test one complete training step."""
    
    print("Testing Regional Training Step")
    print("="*60)
    
    # Import components
    from datasets.dog_face_regional_dataset import DogFaceRegional
    from datasets.make_dataloader_regional import RegionalCollator
    from model import make_regional_model
    from loss import make_loss
    from torchvision import transforms
    
    # Simple config
    class SimpleConfig:
        ROOT_DIR = "/home/sagemaker-user/LostPet/LostPetTest"
        IMAGES_DIR = "/home/sagemaker-user/LostPet/PetFace/dog"
        TRAIN_SPLIT = "splits_petface_valid/train.csv"
        PIXEL_MEAN = [0.485, 0.456, 0.406]
        PIXEL_STD = [0.229, 0.224, 0.225]
        USE_CAMID = False
    
    cfg = SimpleConfig()
    landmarks_dir = "/home/sagemaker-user/LostPet/dogface_landmark_estimation_hrcnn/petface_landmarks_json_all"
    
    # 1. Create dataloader
    print("\n1. Creating dataloader...")
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=cfg.PIXEL_MEAN, std=cfg.PIXEL_STD)
    ])
    
    transform_regional = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize(mean=cfg.PIXEL_MEAN, std=cfg.PIXEL_STD)
    ])
    
    dataset = DogFaceRegional(
        root=cfg.ROOT_DIR,
        split_csv=cfg.TRAIN_SPLIT,
        landmarks_dir=landmarks_dir,
        images_dir=cfg.IMAGES_DIR,
        transform_global=transform,
        transform_regional=transform_regional,
        relabel=True
    )
    
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=4,
        shuffle=False,
        num_workers=0,
        collate_fn=RegionalCollator()
    )
    
    print(f"✅ Dataloader created: {len(dataset)} images, {dataset.num_classes} classes")
    
    # 2. Create model
    print("\n2. Creating model...")
    model = make_regional_model(
        backbone_name='dinov3_vitl16',
        num_classes=dataset.num_classes,
        embed_dim=768,
        pretrained=False  # Fast for testing
    )
    model.freeze_backbone()
    print(f"✅ Model created and backbone frozen")
    
    # 3. Create loss
    print("\n3. Creating loss...")
    loss_fn = make_loss(
        num_classes=dataset.num_classes,
        feat_dim=768,
        id_loss_weight=1.0,
        triplet_loss_weight=1.0
    )
    print(f"✅ Loss created")
    
    # 4. Create optimizer
    print("\n4. Creating optimizer...")
    optimizer = torch.optim.Adam(
        [p for p in model.parameters() if p.requires_grad],
        lr=3e-4
    )
    print(f"✅ Optimizer created")
    
    # 5. Load one batch
    print("\n5. Loading one batch...")
    imgs, regions, pids, camids, paths = next(iter(dataloader))
    print(f"✅ Batch loaded:")
    print(f"   Images: {imgs.shape}")
    print(f"   Regions: {list(regions.keys())}")
    print(f"   PIDs: {pids.tolist()}")
    
    # 6. Forward pass
    print("\n6. Running forward pass...")
    model.train()
    
    logits, features = model(imgs, regions)
    print(f"✅ Forward pass:")
    print(f"   Logits: {logits.shape}")
    print(f"   Features: {features.shape}")
    
    # 7. Compute loss
    print("\n7. Computing loss...")
    loss, loss_dict = loss_fn(logits, features, pids)
    print(f"✅ Loss computed:")
    print(f"   Total loss: {loss.item():.4f}")
    print(f"   ID loss: {loss_dict['id_loss']:.4f}")
    print(f"   Triplet loss: {loss_dict['triplet_loss']:.4f}")
    
    # 8. Backward pass
    print("\n8. Running backward pass...")
    optimizer.zero_grad()
    loss.backward()
    print(f"✅ Backward pass completed")
    
    # Check gradients
    has_grad = sum(1 for p in model.parameters() if p.grad is not None and p.requires_grad)
    total_trainable = sum(1 for p in model.parameters() if p.requires_grad)
    print(f"   Parameters with gradients: {has_grad}/{total_trainable}")
    
    # 9. Optimizer step
    print("\n9. Running optimizer step...")
    optimizer.step()
    print(f"✅ Optimizer step completed")
    
    print("\n" + "="*60)
    print("✅ Complete training step successful!")
    print("="*60)
    print("\nPipeline verified:")
    print("  ✓ Dataloader loads regional data")
    print("  ✓ Model processes global + regions")
    print("  ✓ Loss computes on outputs")
    print("  ✓ Backward pass computes gradients")
    print("  ✓ Optimizer updates trainable params")
    print("\nReady for full training!")


if __name__ == "__main__":
    test_training_step()

