import os
import csv
import random
from pathlib import Path

def create_petface_splits(
    data_root="/home/sagemaker-user/LostPet/petface/dog",
    output_dir="./splits_petface",
    train_ratio=0.7,
    val_ratio=0.1,
    test_ratio=0.2,
    seed=42
):
    """
    Create train/val/test splits for petface dataset.
    
    Args:
        data_root: Root directory containing dog ID folders
        output_dir: Directory to save CSV split files
        train_ratio: Proportion of dogs for training (default: 0.7)
        val_ratio: Proportion of dogs for validation (default: 0.1)
        test_ratio: Proportion of dogs for testing (default: 0.2)
        seed: Random seed for reproducibility
    """
    random.seed(seed)
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Get all dog ID folders
    data_path = Path(data_root)
    if not data_path.exists():
        print(f"ERROR: Directory {data_root} does not exist!")
        return
    
    dog_ids = sorted([d.name for d in data_path.iterdir() if d.is_dir()])
    print(f"Found {len(dog_ids)} dog IDs")
    
    if len(dog_ids) == 0:
        print("ERROR: No dog ID folders found!")
        return
    
    # Shuffle and split dog IDs
    random.shuffle(dog_ids)
    
    n_total = len(dog_ids)
    n_train = int(n_total * train_ratio)
    n_val = int(n_total * val_ratio)
    
    train_ids = dog_ids[:n_train]
    val_ids = dog_ids[n_train:n_train + n_val]
    test_ids = dog_ids[n_train + n_val:]
    
    print(f"\nSplit summary:")
    print(f"  Train: {len(train_ids)} dogs ({len(train_ids)/n_total*100:.1f}%)")
    print(f"  Val:   {len(val_ids)} dogs ({len(val_ids)/n_total*100:.1f}%)")
    print(f"  Test:  {len(test_ids)} dogs ({len(test_ids)/n_total*100:.1f}%)")
    
    # Helper function to collect images for a set of dog IDs
    def collect_images(dog_id_list):
        data = []
        for dog_id in dog_id_list:
            dog_folder = data_path / dog_id
            
            # Find all image files in the folder
            image_files = []
            for ext in ['*.png', '*.jpg', '*.jpeg', '*.PNG', '*.JPG', '*.JPEG']:
                image_files.extend(dog_folder.glob(ext))
            
            # Sort by filename
            image_files = sorted(image_files, key=lambda x: x.name)
            
            if len(image_files) == 0:
                print(f"WARNING: No images found for dog ID {dog_id}")
                continue
            
            # Add each image to the dataset
            for img_file in image_files:
                # Relative path from the dog root: e.g., "067642/00.png"
                img_rel_path = f"{dog_id}/{img_file.name}"
                
                data.append({
                    'img_rel_path': img_rel_path,
                    'pid': dog_id,
                    'camid': 0  # Not applicable for this dataset
                })
        
        return data
    
    # Collect images for each split
    print("\nCollecting images...")
    train_data = collect_images(train_ids)
    val_data = collect_images(val_ids)
    test_data = collect_images(test_ids)
    
    print(f"\nImage counts:")
    print(f"  Train: {len(train_data)} images")
    print(f"  Val:   {len(val_data)} images")
    print(f"  Test:  {len(test_data)} images")
    
    # Write CSV files
    def write_csv(data, filename):
        csv_path = os.path.join(output_dir, filename)
        with open(csv_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=['img_rel_path', 'pid', 'camid'])
            writer.writeheader()
            writer.writerows(data)
        print(f"Saved: {csv_path}")
    
    print("\nWriting CSV files...")
    write_csv(train_data, 'train.csv')
    write_csv(val_data, 'val.csv')
    write_csv(test_data, 'test.csv')
    
    # Also create query/gallery splits from test set (for ReID evaluation)
    # Typically: 1 image per dog in query, rest in gallery
    print("\nCreating query/gallery splits from test set...")
    
    query_data = []
    gallery_data = []
    
    for dog_id in test_ids:
        # Get all images for this dog in test set
        dog_images = [d for d in test_data if d['pid'] == dog_id]
        
        if len(dog_images) > 0:
            # First image goes to query
            query_data.append(dog_images[0])
            # Rest go to gallery
            gallery_data.extend(dog_images[1:])
            
            # If only one image, also add to gallery for matching
            if len(dog_images) == 1:
                gallery_data.append(dog_images[0])
    
    print(f"  Query:   {len(query_data)} images")
    print(f"  Gallery: {len(gallery_data)} images")
    
    write_csv(query_data, 'query.csv')
    write_csv(gallery_data, 'gallery.csv')
    
    print("\n✓ Done! Split files created in:", output_dir)

if __name__ == "__main__":
    create_petface_splits()

