"""
Create train/val/test splits from filtered valid images JSON.
Splits by dog ID with 7:1:2 ratio.
"""

import json
import csv
import random
import os
from pathlib import Path


def create_valid_splits(
    valid_json_path="petface_valid_images_with_ears.json",
    output_dir="./splits_petface_valid",
    train_ratio=0.7,
    val_ratio=0.1,
    test_ratio=0.2,
    num_dogs=30000,
    seed=42
):
    """
    Create train/val/test splits from filtered valid images.
    
    Args:
        valid_json_path: Path to filtered valid images JSON
        output_dir: Directory to save CSV split files
        train_ratio: Proportion of dogs for training (default: 0.7)
        val_ratio: Proportion of dogs for validation (default: 0.1)
        test_ratio: Proportion of dogs for testing (default: 0.2)
        num_dogs: Number of dogs to use (None = use all, default: 3000 for dev)
        seed: Random seed for reproducibility
    """
    random.seed(seed)
    
    # Load filtered valid images
    print("Loading filtered valid images...")
    with open(valid_json_path, 'r') as f:
        data = json.load(f)
    
    dog_images = data['dog_images']
    all_dog_ids = sorted(dog_images.keys())
    
    print(f"Total valid dog IDs: {len(all_dog_ids)}")
    print(f"Total valid images: {data['metadata']['final_valid_images']}")
    
    # Limit number of dogs if specified
    if num_dogs is not None and num_dogs < len(all_dog_ids):
        print(f"Using subset: {num_dogs} dogs (for dev/testing)")
        random.seed(seed)
        dog_ids = random.sample(all_dog_ids, num_dogs)
    else:
        dog_ids = all_dog_ids
        print(f"Using all {len(dog_ids)} dogs")
    
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
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Helper function to collect images for a set of dog IDs
    def collect_images(dog_id_list):
        """Collect all images for given dog IDs."""
        data = []
        for dog_id in dog_id_list:
            photo_ids = dog_images[dog_id]
            
            for photo_id in photo_ids:
                # Format: dog_id/photo_id.png (e.g., "011103/00.png")
                img_rel_path = f"{dog_id}/{photo_id}.png"
                
                data.append({
                    'img_rel_path': img_rel_path,
                    'pid': dog_id,  # Use dog_id as PID
                    'camid': 0      # No camera ID for PetFace
                })
        
        return data
    
    # Helper function to create query/gallery splits
    def create_query_gallery(dog_id_list):
        """
        Create query and gallery splits.
        Query: first image of each dog
        Gallery: all images (including query for single-image dogs)
        """
        query = []
        gallery = []
        
        for dog_id in dog_id_list:
            photo_ids = sorted(dog_images[dog_id])
            
            # First image as query
            query_photo = photo_ids[0]
            query.append({
                'img_rel_path': f"{dog_id}/{query_photo}.png",
                'pid': dog_id,
                'camid': 0
            })
            
            # All images in gallery (includes query for consistency)
            for photo_id in photo_ids:
                gallery.append({
                    'img_rel_path': f"{dog_id}/{photo_id}.png",
                    'pid': dog_id,
                    'camid': 0
                })
        
        return query, gallery
    
    # Collect training data (all images from train dog IDs)
    print("\nCreating training split...")
    train_data = collect_images(train_ids)
    print(f"  Train images: {len(train_data)}")
    
    # Create val query/gallery
    print("Creating validation splits...")
    val_query, val_gallery = create_query_gallery(val_ids)
    print(f"  Val query: {len(val_query)} images ({len(val_ids)} dogs)")
    print(f"  Val gallery: {len(val_gallery)} images")
    
    # Create test query/gallery
    print("Creating test splits...")
    test_query, test_gallery = create_query_gallery(test_ids)
    print(f"  Test query: {len(test_query)} images ({len(test_ids)} dogs)")
    print(f"  Test gallery: {len(test_gallery)} images")
    
    # Write CSV files
    def write_csv(filename, data):
        filepath = os.path.join(output_dir, filename)
        with open(filepath, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=['img_rel_path', 'pid', 'camid'])
            writer.writeheader()
            writer.writerows(data)
        print(f"  Saved: {filepath}")
    
    print("\nWriting CSV files...")
    write_csv('train.csv', train_data)
    write_csv('val_query.csv', val_query)
    write_csv('val_gallery.csv', val_gallery)
    write_csv('test_query.csv', test_query)
    write_csv('test_gallery.csv', test_gallery)
    
    print("\n" + "="*60)
    print("✅ Splits created successfully!")
    print("="*60)
    print(f"Output directory: {output_dir}/")
    print("\nFiles created:")
    print(f"  - train.csv        ({len(train_data):,} images)")
    print(f"  - val_query.csv    ({len(val_query):,} images)")
    print(f"  - val_gallery.csv  ({len(val_gallery):,} images)")
    print(f"  - test_query.csv   ({len(test_query):,} images)")
    print(f"  - test_gallery.csv ({len(test_gallery):,} images)")
    
    # Summary statistics
    print("\nDataset statistics:")
    print(f"  Total dog IDs: {len(dog_ids)}")
    print(f"  Train dogs: {len(train_ids)} ({len(train_ids)/len(dog_ids)*100:.1f}%)")
    print(f"  Val dogs:   {len(val_ids)} ({len(val_ids)/len(dog_ids)*100:.1f}%)")
    print(f"  Test dogs:  {len(test_ids)} ({len(test_ids)/len(dog_ids)*100:.1f}%)")
    print(f"\n  Total images: {len(train_data) + len(val_gallery) + len(test_gallery)}")
    print(f"  Train images: {len(train_data)} ({len(train_data)/(len(train_data)+len(val_gallery)+len(test_gallery))*100:.1f}%)")
    print(f"  Val images:   {len(val_gallery)} ({len(val_gallery)/(len(train_data)+len(val_gallery)+len(test_gallery))*100:.1f}%)")
    print(f"  Test images:  {len(test_gallery)} ({len(test_gallery)/(len(train_data)+len(val_gallery)+len(test_gallery))*100:.1f}%)")


if __name__ == "__main__":
    create_valid_splits()
