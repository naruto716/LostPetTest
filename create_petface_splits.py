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
    
    # Helper function to create query/gallery splits
    def create_query_gallery(dog_id_list, all_data):
        """
        Create query and gallery splits from a set of dog IDs.
        Query: first image of each dog
        Gallery: remaining images (or same image if only 1 image per dog)
        """
        query = []
        gallery = []
        
        for dog_id in dog_id_list:
            # Get all images for this dog
            dog_images = [d for d in all_data if d['pid'] == dog_id]
            
            if len(dog_images) > 0:
                # First image goes to query
                query.append(dog_images[0])
                # Rest go to gallery
                gallery.extend(dog_images[1:])
                
                # If only one image, also add to gallery for matching
                if len(dog_images) == 1:
                    gallery.append(dog_images[0])
        
        return query, gallery
    
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
    
    # Create query/gallery splits for VALIDATION
    print("\nCreating VALIDATION query/gallery splits...")
    val_query, val_gallery = create_query_gallery(val_ids, val_data)
    print(f"  Val Query:   {len(val_query)} images ({len(val_ids)} dogs)")
    print(f"  Val Gallery: {len(val_gallery)} images")
    
    write_csv(val_query, 'val_query.csv')
    write_csv(val_gallery, 'val_gallery.csv')
    
    # Create query/gallery splits for TEST
    print("\nCreating TEST query/gallery splits...")
    test_query, test_gallery = create_query_gallery(test_ids, test_data)
    print(f"  Test Query:   {len(test_query)} images ({len(test_ids)} dogs)")
    print(f"  Test Gallery: {len(test_gallery)} images")
    
    write_csv(test_query, 'test_query.csv')
    write_csv(test_gallery, 'test_gallery.csv')
    
    print("\n✓ Done! Split files created in:", output_dir)

def create_petface_splits_subset(
    data_root="/home/sagemaker-user/LostPet/petface/dog",
    output_dir="./splits_petface_subset",
    target_train_images=2000,  # Target number of training images
    train_ratio=0.7,
    val_ratio=0.1,
    test_ratio=0.2,
    seed=42
):
    """
    Create a small subset of petface for quick experimentation.
    Samples dogs to reach approximately target_train_images, then splits 7:1:2.
    
    Args:
        data_root: Root directory containing dog ID folders
        output_dir: Directory to save CSV split files
        target_train_images: Approximate number of training images desired
        train_ratio: Proportion of sampled dogs for training (default: 0.7)
        val_ratio: Proportion of sampled dogs for validation (default: 0.1)
        test_ratio: Proportion of sampled dogs for testing (default: 0.2)
        seed: Random seed for reproducibility
    """
    random.seed(seed)
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Get all dog ID folders
    data_path = Path(data_root)
    if not data_path.exists():
        print(f"ERROR: Directory {data_root} does not exist!")
        return
    
    all_dog_ids = sorted([d.name for d in data_path.iterdir() if d.is_dir()])
    print(f"Found {len(all_dog_ids)} total dog IDs")
    
    if len(all_dog_ids) == 0:
        print("ERROR: No dog ID folders found!")
        return
    
    # Sample dogs to reach target training images
    print(f"\nSampling dogs to reach ~{target_train_images} training images...")
    random.shuffle(all_dog_ids)
    
    sampled_dogs = []
    total_images = 0
    
    for dog_id in all_dog_ids:
        dog_folder = data_path / dog_id
        
        # Count images for this dog
        image_files = []
        for ext in ['*.png', '*.jpg', '*.jpeg', '*.PNG', '*.JPG', '*.JPEG']:
            image_files.extend(dog_folder.glob(ext))
        
        num_images = len(image_files)
        if num_images > 0:
            sampled_dogs.append(dog_id)
            total_images += num_images
            
            # Stop when we have enough images (accounting for train_ratio)
            if total_images >= target_train_images / train_ratio:
                break
    
    n_sampled = len(sampled_dogs)
    estimated_train_images = int(total_images * train_ratio)
    
    print(f"Sampled {n_sampled} dogs with {total_images} total images")
    print(f"Estimated training images: {estimated_train_images}")
    
    # Now split the sampled dogs 7:1:2
    n_train = int(n_sampled * train_ratio)
    n_val = int(n_sampled * val_ratio)
    
    train_ids = sampled_dogs[:n_train]
    val_ids = sampled_dogs[n_train:n_train + n_val]
    test_ids = sampled_dogs[n_train + n_val:]
    
    print(f"\nSubset split summary:")
    print(f"  Train: {len(train_ids)} dogs ({len(train_ids)/n_sampled*100:.1f}%)")
    print(f"  Val:   {len(val_ids)} dogs ({len(val_ids)/n_sampled*100:.1f}%)")
    print(f"  Test:  {len(test_ids)} dogs ({len(test_ids)/n_sampled*100:.1f}%)")
    
    # Helper function to collect images for a set of dog IDs
    def collect_images(dog_id_list):
        data = []
        for dog_id in dog_id_list:
            dog_folder = data_path / dog_id
            
            # Find all image files
            image_files = []
            for ext in ['*.png', '*.jpg', '*.jpeg', '*.PNG', '*.JPG', '*.JPEG']:
                image_files.extend(dog_folder.glob(ext))
            
            image_files = sorted(image_files, key=lambda x: x.name)
            
            if len(image_files) == 0:
                continue
            
            for img_file in image_files:
                img_rel_path = f"{dog_id}/{img_file.name}"
                data.append({
                    'img_rel_path': img_rel_path,
                    'pid': dog_id,
                    'camid': 0
                })
        
        return data
    
    # Helper function to create query/gallery splits
    def create_query_gallery(dog_id_list, all_data):
        query = []
        gallery = []
        
        for dog_id in dog_id_list:
            dog_images = [d for d in all_data if d['pid'] == dog_id]
            
            if len(dog_images) > 0:
                query.append(dog_images[0])
                gallery.extend(dog_images[1:])
                
                if len(dog_images) == 1:
                    gallery.append(dog_images[0])
        
        return query, gallery
    
    # Collect images
    print("\nCollecting images...")
    train_data = collect_images(train_ids)
    val_data = collect_images(val_ids)
    test_data = collect_images(test_ids)
    
    print(f"\nActual image counts:")
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
    
    # Create query/gallery for validation
    print("\nCreating VALIDATION query/gallery splits...")
    val_query, val_gallery = create_query_gallery(val_ids, val_data)
    print(f"  Val Query:   {len(val_query)} images ({len(val_ids)} dogs)")
    print(f"  Val Gallery: {len(val_gallery)} images")
    
    write_csv(val_query, 'val_query.csv')
    write_csv(val_gallery, 'val_gallery.csv')
    
    # Create query/gallery for test
    print("\nCreating TEST query/gallery splits...")
    test_query, test_gallery = create_query_gallery(test_ids, test_data)
    print(f"  Test Query:   {len(test_query)} images ({len(test_ids)} dogs)")
    print(f"  Test Gallery: {len(test_gallery)} images")
    
    write_csv(test_query, 'test_query.csv')
    write_csv(test_gallery, 'test_gallery.csv')
    
    print("\n✓ Done! Subset split files created in:", output_dir)
    print(f"\n📊 Quick Stats:")
    print(f"   Total dogs: {n_sampled}")
    print(f"   Training images: {len(train_data)} (~{len(train_data)} images, {len(train_ids)} dogs)")
    print(f"   Estimated training time: ~{len(train_data) / 64 * 0.5 / 60:.1f} min/epoch")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "--subset":
        # Create subset for quick experimentation
        target = int(sys.argv[2]) if len(sys.argv) > 2 else 2000
        print(f"Creating subset with ~{target} training images...")
        create_petface_splits_subset(target_train_images=target)
    else:
        # Create full splits
        print("Creating full dataset splits...")
        print("(Use --subset [num_images] for a small subset)")
        create_petface_splits()

