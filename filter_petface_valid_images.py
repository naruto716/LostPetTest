"""
Filter PetFace dataset to only valid images with complete landmarks.
Generates JSON with dog_id -> list of valid image_ids.
"""

import os
import json
from pathlib import Path
from tqdm import tqdm
from collections import defaultdict


def is_valid_image(landmarks):
    """
    Check if an image has valid landmarks.
    Returns (is_valid, issues_list)
    """
    issues = []
    
    if 'region_bboxes' not in landmarks:
        return False, ["No region_bboxes"]
    
    region_bboxes = landmarks['region_bboxes']
    
    # Required regions
    required_regions = ['left_eye', 'right_eye', 'nose', 'mouth', 'forehead']
    optional_regions = ['left_ear', 'right_ear']  # Ears often occluded
    
    # Check for missing required regions
    for region in required_regions:
        if region not in region_bboxes:
            issues.append(f"Missing {region}")
    
    # Check for unreasonably small detections
    min_sizes = {
        'left_eye': 20,
        'right_eye': 20,
        'nose': 100,
        'mouth': 500,
        'forehead': 500,
        'left_ear': 200,
        'right_ear': 200
    }
    
    for region, bbox in region_bboxes.items():
        area = bbox['width'] * bbox['height']
        min_size = min_sizes.get(region, 0)
        
        if region in required_regions and area < min_size:
            issues.append(f"Tiny {region}: {area}px² (min: {min_size})")
        elif region in optional_regions and area > 0 and area < min_size:
            # For optional regions, only flag if present but too small
            issues.append(f"Tiny {region}: {area}px²")
    
    # Valid if no issues with required regions
    required_issues = [iss for iss in issues if any(req in iss for req in required_regions)]
    is_valid = len(required_issues) == 0
    
    return is_valid, issues


def filter_petface_dataset(
    landmarks_dir,
    output_json="petface_valid_images.json",
    min_images_per_dog=2
):
    """
    Filter PetFace dataset to valid images only.
    
    Args:
        landmarks_dir: Directory with landmark JSONs
        output_json: Output JSON file
        min_images_per_dog: Minimum valid images required per dog_id
    """
    
    landmarks_dir = Path(landmarks_dir)
    
    # Get all landmark files
    landmark_files = sorted(landmarks_dir.glob("*.json"))
    print(f"Found {len(landmark_files)} landmark files")
    print(f"Processing...\n")
    
    # Track valid images per dog
    dog_valid_images = defaultdict(list)
    
    # Statistics
    total_images = 0
    valid_images = 0
    invalid_images = 0
    issue_counts = defaultdict(int)
    
    # Process each landmark file
    for json_file in tqdm(landmark_files, desc="Filtering images"):
        total_images += 1
        
        # Parse dog_id and photo_id from filename
        # Format: 029364_02.json -> dog_id=029364, photo_id=02
        filename = json_file.stem
        parts = filename.split('_')
        if len(parts) != 2:
            continue
        
        dog_id, photo_id = parts
        
        # Load landmarks
        try:
            with open(json_file, 'r') as f:
                landmarks = json.load(f)
        except Exception as e:
            invalid_images += 1
            issue_counts["JSON parse error"] += 1
            continue
        
        # Check if valid
        is_valid, issues = is_valid_image(landmarks)
        
        if is_valid:
            valid_images += 1
            dog_valid_images[dog_id].append(photo_id)
        else:
            invalid_images += 1
            # Track issue types
            for issue in issues:
                issue_counts[issue] += 1
    
    # Filter dogs with insufficient valid images
    dogs_before = len(dog_valid_images)
    dog_valid_images = {
        dog_id: images 
        for dog_id, images in dog_valid_images.items() 
        if len(images) >= min_images_per_dog
    }
    dogs_after = len(dog_valid_images)
    dogs_removed = dogs_before - dogs_after
    
    # Count total valid images after dog filtering
    total_valid_after_filter = sum(len(images) for images in dog_valid_images.values())
    
    # Save to JSON
    output_data = {
        'metadata': {
            'total_images_processed': total_images,
            'valid_images': valid_images,
            'invalid_images': invalid_images,
            'total_dogs_with_valid_images': dogs_before,
            'dogs_removed_insufficient_images': dogs_removed,
            'final_dog_count': dogs_after,
            'final_valid_images': total_valid_after_filter,
            'min_images_per_dog': min_images_per_dog
        },
        'dog_images': dog_valid_images
    }
    
    with open(output_json, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    # Print summary
    print("\n" + "="*60)
    print("FILTERING SUMMARY")
    print("="*60)
    print(f"Total images processed:     {total_images:,}")
    print(f"Valid images:               {valid_images:,} ({valid_images/total_images*100:.1f}%)")
    print(f"Invalid images:             {invalid_images:,} ({invalid_images/total_images*100:.1f}%)")
    print()
    print(f"Dogs with valid images:     {dogs_before:,}")
    print(f"Dogs removed (<{min_images_per_dog} images): {dogs_removed:,}")
    print(f"Final dog count:            {dogs_after:,}")
    print(f"Final valid images:         {total_valid_after_filter:,}")
    print()
    print("Top issues found:")
    for issue, count in sorted(issue_counts.items(), key=lambda x: x[1], reverse=True)[:10]:
        print(f"  {issue:40} - {count:,} images")
    
    print(f"\n✅ Saved to: {output_json}")
    
    # Show some example dog_ids
    print("\nExample valid dog_ids:")
    for dog_id, images in list(dog_valid_images.items())[:5]:
        print(f"  {dog_id}: {len(images)} images - {images[:5]}{'...' if len(images) > 5 else ''}")
    
    return output_data


if __name__ == "__main__":
    # Run on sagemaker
    filter_petface_dataset(
        landmarks_dir="/home/sagemaker-user/LostPet/dogface_landmark_estimation_hrcnn/petface_landmarks_json_all",
        output_json="petface_valid_images.json",
        min_images_per_dog=2
    )
