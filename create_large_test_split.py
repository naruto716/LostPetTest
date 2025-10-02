"""
Create a large test split from PetFace for generalization testing.

Purpose: Test a model trained on ~784 dogs on a much larger set of UNSEEN dogs
to evaluate how well it generalizes to many more identities.

The trained model was trained on subset dogs (train split).
This creates a NEW test set with many more different dogs.
"""

from pathlib import Path
import random
import csv
import argparse

def create_large_test_split(
    data_root="/home/sagemaker-user/LostPet/petface/dog",
    output_dir="./splits_petface_large_test",
    num_test_dogs=10000,  # Target number of test dogs
    trained_dogs_file="./splits_petface_subset/train.csv",  # Dogs already used for training
    seed=42
):
    """
    Create a large test set with many unseen dog identities.
    
    Args:
        data_root: Root directory containing dog ID folders
        output_dir: Where to save the new test split CSVs
        num_test_dogs: How many NEW dogs to include in test set
        trained_dogs_file: CSV of dogs used for training (we exclude these)
        seed: Random seed
    """
    print("\n" + "="*80)
    print("🧪 Creating Large Test Split for Generalization Testing")
    print("="*80)
    
    random.seed(seed)
    data_path = Path(data_root)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Get all available dog IDs
    all_dog_ids = sorted([d.name for d in data_path.iterdir() if d.is_dir()])
    print(f"📊 Total dogs in dataset: {len(all_dog_ids)}")
    
    # Load dogs that were used for training (to EXCLUDE them)
    print(f"📂 Loading trained dogs from: {trained_dogs_file}")
    trained_dog_ids = set()
    if Path(trained_dogs_file).exists():
        with open(trained_dogs_file, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                trained_dog_ids.add(str(row['pid']))
        print(f"   Trained on: {len(trained_dog_ids)} dogs")
    else:
        print("   ⚠️  Training file not found, using all dogs")
    
    # Get UNSEEN dogs (not used for training)
    unseen_dogs = [dog_id for dog_id in all_dog_ids if dog_id not in trained_dog_ids]
    print(f"✅ Available unseen dogs: {len(unseen_dogs)}")
    
    # Sample test dogs
    if len(unseen_dogs) < num_test_dogs:
        print(f"⚠️  Only {len(unseen_dogs)} unseen dogs available, using all")
        test_dog_ids = unseen_dogs
    else:
        test_dog_ids = random.sample(unseen_dogs, num_test_dogs)
    
    print(f"🎯 Selected {len(test_dog_ids)} dogs for large test set")
    
    # Collect images for test dogs
    def collect_images(dog_id_list):
        data = []
        dogs_with_images = 0
        total_images = 0
        
        for i, dog_id in enumerate(dog_id_list):
            if (i + 1) % 1000 == 0:
                print(f"   Processing: {i+1}/{len(dog_id_list)} dogs...")
            
            dog_folder = data_path / dog_id
            
            # Find all image files
            image_files = []
            for ext in ['*.png', '*.jpg', '*.jpeg', '*.PNG', '*.JPG', '*.JPEG']:
                image_files.extend(dog_folder.glob(ext))
            
            image_files = sorted(image_files, key=lambda x: x.name)
            
            if len(image_files) == 0:
                continue
            
            dogs_with_images += 1
            total_images += len(image_files)
            
            # Add all images
            for img_file in image_files:
                img_rel_path = f"{dog_id}/{img_file.name}"
                data.append({
                    'img_rel_path': img_rel_path,
                    'pid': dog_id,
                    'camid': 0
                })
        
        print(f"   ✅ Found {total_images} images from {dogs_with_images} dogs")
        return data
    
    # Create query/gallery split
    def create_query_gallery(dog_id_list, all_data):
        """Split into query (first image per dog) and gallery (rest)"""
        query_data = []
        gallery_data = []
        
        # Group by dog ID
        dog_to_images = {}
        for record in all_data:
            pid = record['pid']
            if pid not in dog_to_images:
                dog_to_images[pid] = []
            dog_to_images[pid].append(record)
        
        for dog_id in dog_id_list:
            if dog_id not in dog_to_images:
                continue
            
            images = dog_to_images[dog_id]
            if len(images) > 0:
                query_data.append(images[0])  # First image = query
                gallery_data.extend(images[1:])  # Rest = gallery
        
        return query_data, gallery_data
    
    # Collect test images
    print("\n📸 Collecting test images...")
    test_data = collect_images(test_dog_ids)
    
    # Create query/gallery split
    print("\n🔀 Creating query/gallery split...")
    test_query_data, test_gallery_data = create_query_gallery(test_dog_ids, test_data)
    
    # Save CSVs
    test_query_path = output_path / "test_query.csv"
    test_gallery_path = output_path / "test_gallery.csv"
    
    # Write query CSV
    with open(test_query_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['img_rel_path', 'pid', 'camid'])
        writer.writeheader()
        writer.writerows(test_query_data)
    
    # Write gallery CSV
    with open(test_gallery_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['img_rel_path', 'pid', 'camid'])
        writer.writeheader()
        writer.writerows(test_gallery_data)
    
    # Count unique dogs
    query_dogs = len(set(record['pid'] for record in test_query_data))
    gallery_dogs = len(set(record['pid'] for record in test_gallery_data))
    
    # Summary
    print("\n" + "="*80)
    print("✅ Large Test Split Created!")
    print("="*80)
    print(f"Test Query:   {len(test_query_data):>6} images, {query_dogs:>6} dogs")
    print(f"Test Gallery: {len(test_gallery_data):>6} images, {gallery_dogs:>6} dogs")
    print(f"\nSaved to: {output_dir}/")
    print(f"  - test_query.csv")
    print(f"  - test_gallery.csv")
    print("="*80)
    
    print("\n💡 Next steps:")
    print(f"   1. Update config_petface.py to point to these splits")
    print(f"   2. Run: uv run python test_petface.py")
    print(f"   3. Compare performance: ~784 training dogs → {len(test_dog_ids)} test dogs")
    print()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Create large test split")
    parser.add_argument(
        '--num_test_dogs',
        type=int,
        default=10000,
        help='Number of dogs to include in test set (default: 10000)'
    )
    parser.add_argument(
        '--data_root',
        default='/home/sagemaker-user/LostPet/petface/dog',
        help='Root directory with dog images'
    )
    parser.add_argument(
        '--output_dir',
        default='./splits_petface_large_test',
        help='Output directory for splits'
    )
    
    args = parser.parse_args()
    
    create_large_test_split(
        data_root=args.data_root,
        output_dir=args.output_dir,
        num_test_dogs=args.num_test_dogs
    )

