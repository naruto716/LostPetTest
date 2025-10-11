"""
Create a large test set (query/gallery) from filtered valid images JSON.
For testing model on a large number of dogs without training.
"""

import json
import csv
import random
import os
from pathlib import Path


def create_large_test_split(
    valid_json_path="petface_valid_images_with_ears.json",
    output_dir="./splits_petface_test_10k",
    num_dogs=10000,
    seed=42
):
    """
    Create test query/gallery splits from filtered valid images.
    
    Args:
        valid_json_path: Path to filtered valid images JSON
        output_dir: Directory to save CSV split files
        num_dogs: Number of dogs to use for test set (default: 10000)
        seed: Random seed for reproducibility
    """
    random.seed(seed)
    
    # Load filtered valid images
    print("Loading filtered valid images...")
    with open(valid_json_path, 'r') as f:
        data = json.load(f)
    
    dog_images = data['dog_images']
    all_dog_ids = sorted(dog_images.keys())
    
    print(f"Total valid dog IDs available: {len(all_dog_ids)}")
    print(f"Total valid images: {data['metadata']['final_valid_images']}")
    
    # Select dogs for test set
    if num_dogs is not None and num_dogs < len(all_dog_ids):
        print(f"\nSelecting {num_dogs} dogs for test set")
        random.seed(seed)
        test_dog_ids = random.sample(all_dog_ids, num_dogs)
    else:
        test_dog_ids = all_dog_ids
        print(f"\nUsing all {len(test_dog_ids)} dogs for test set")
    
    # Shuffle for good measure
    random.shuffle(test_dog_ids)
    
    print(f"Test set: {len(test_dog_ids)} dogs")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
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
    
    # Create test query/gallery
    print("\nCreating test splits...")
    test_query, test_gallery = create_query_gallery(test_dog_ids)
    print(f"  Test query: {len(test_query)} images ({len(test_dog_ids)} dogs)")
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
    write_csv('test_query.csv', test_query)
    write_csv('test_gallery.csv', test_gallery)
    
    print("\n" + "="*60)
    print("✅ Test splits created successfully!")
    print("="*60)
    print(f"Output directory: {output_dir}/")
    print("\nFiles created:")
    print(f"  - test_query.csv   ({len(test_query):,} images)")
    print(f"  - test_gallery.csv ({len(test_gallery):,} images)")
    
    # Summary statistics
    print("\nDataset statistics:")
    print(f"  Total dogs: {len(test_dog_ids):,}")
    print(f"  Total images: {len(test_gallery):,}")
    print(f"  Avg images per dog: {len(test_gallery)/len(test_dog_ids):.1f}")
    
    # Calculate dogs with multiple images
    multi_image_dogs = sum(1 for dog_id in test_dog_ids if len(dog_images[dog_id]) > 1)
    print(f"  Dogs with multiple images: {multi_image_dogs:,} ({multi_image_dogs/len(test_dog_ids)*100:.1f}%)")
    print(f"  Dogs with single image: {len(test_dog_ids)-multi_image_dogs:,} ({(len(test_dog_ids)-multi_image_dogs)/len(test_dog_ids)*100:.1f}%)")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Create large test split from PetFace valid images")
    parser.add_argument(
        '--num_dogs',
        type=int,
        default=10000,
        help='Number of dogs for test set (default: 10000)'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default=None,
        help='Output directory (default: ./splits_petface_test_<num_dogs>)'
    )
    parser.add_argument(
        '--valid_json',
        type=str,
        default='petface_valid_images_with_ears.json',
        help='Path to valid images JSON'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for reproducibility'
    )
    
    args = parser.parse_args()
    
    # Set default output dir based on num_dogs
    if args.output_dir is None:
        args.output_dir = f"./splits_petface_test_{args.num_dogs//1000}k"
    
    create_large_test_split(
        valid_json_path=args.valid_json,
        output_dir=args.output_dir,
        num_dogs=args.num_dogs,
        seed=args.seed
    )
