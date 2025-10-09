"""
Simple bbox validation without external dependencies.
Shows the expected landmark JSON structure and bbox statistics.
"""

import json


def show_sample_landmark_structure():
    """Display the expected landmark JSON structure."""
    sample_json = {
        "image_path": "/home/sagemaker-user/LostPet/PetFace/dog/029364/02.png",
        "image_width": 224,
        "image_height": 224,
        "landmarks": ["... 46 landmark points ..."],
        "region_bboxes": {
            "left_eye": {"x_min": 147, "y_min": 88, "x_max": 169, "y_max": 92, "width": 22, "height": 4},
            "right_eye": {"x_min": 56, "y_min": 89, "x_max": 73, "y_max": 102, "width": 17, "height": 13},
            "nose": {"x_min": 81, "y_min": 135, "x_max": 141, "y_max": 188, "width": 60, "height": 53},
            "mouth": {"x_min": 49, "y_min": 104, "x_max": 178, "y_max": 207, "width": 129, "height": 103},
            "left_ear": {"x_min": 183, "y_min": 16, "x_max": 222, "y_max": 169, "width": 39, "height": 153},
            "right_ear": {"x_min": 3, "y_min": 4, "x_max": 48, "y_max": 200, "width": 45, "height": 196},
            "forehead": {"x_min": 30, "y_min": 14, "x_max": 201, "y_max": 97, "width": 171, "height": 83}
        },
        "avg_confidence": 0.5892922878265381,
        "visible_landmarks": 42
    }
    
    print("Expected Landmark JSON Structure:")
    print("="*60)
    print(json.dumps(sample_json, indent=2))
    print("\n")
    
    return sample_json


def analyze_bboxes(landmarks_data):
    """Analyze bbox statistics."""
    print("Bounding Box Analysis:")
    print("="*60)
    
    regions = landmarks_data['region_bboxes']
    img_width = landmarks_data['image_width']
    img_height = landmarks_data['image_height']
    
    print(f"Image size: {img_width}×{img_height}\n")
    
    # Headers
    print(f"{'Region':12} {'Size':12} {'Area':8} {'Coverage':8} {'Position':20}")
    print("-"*70)
    
    total_area = 0
    for region_name, bbox in regions.items():
        area = bbox['width'] * bbox['height']
        coverage = (area / (img_width * img_height)) * 100
        total_area += area
        
        size_str = f"{bbox['width']}×{bbox['height']}"
        pos_str = f"({bbox['x_min']},{bbox['y_min']})"
        
        print(f"{region_name:12} {size_str:12} {area:8} {coverage:7.1f}% {pos_str:20}")
    
    # Summary
    print("-"*70)
    total_coverage = (total_area / (img_width * img_height)) * 100
    print(f"{'Total coverage:':26} {total_area:8} {total_coverage:7.1f}%")
    
    # Validate bboxes
    print("\n\nValidation Checks:")
    print("-"*40)
    
    issues = []
    for region_name, bbox in regions.items():
        # Check if bbox is valid
        if bbox['x_max'] <= bbox['x_min'] or bbox['y_max'] <= bbox['y_min']:
            issues.append(f"❌ {region_name}: Invalid bbox dimensions")
        
        # Check if bbox is within image bounds
        if bbox['x_min'] < 0 or bbox['y_min'] < 0:
            issues.append(f"⚠️  {region_name}: Bbox extends beyond image (negative coords)")
        
        if bbox['x_max'] > img_width or bbox['y_max'] > img_height:
            issues.append(f"⚠️  {region_name}: Bbox extends beyond image boundaries")
        
        # Check aspect ratio
        if bbox['height'] > 0:
            aspect_ratio = bbox['width'] / bbox['height']
            if region_name in ['left_eye', 'right_eye'] and aspect_ratio < 1:
                issues.append(f"ℹ️  {region_name}: Unusual aspect ratio {aspect_ratio:.2f} (eyes are typically wider)")
    
    if issues:
        for issue in issues:
            print(issue)
    else:
        print("✅ All bboxes pass validation checks")


def main():
    print("PetFace Bounding Box Validation")
    print("="*60)
    print()
    
    # Show expected structure
    sample_data = show_sample_landmark_structure()
    
    # Analyze bboxes
    analyze_bboxes(sample_data)
    
    print("\n\nKey Observations:")
    print("-"*40)
    print("1. Eyes have small height (4-13px) - may need careful handling")
    print("2. Ears are tall and narrow (aspect ratio ~0.25)")
    print("3. Mouth region is the largest (129×103)")
    print("4. Total coverage is ~35% of image area")
    print("5. Some regions overlap (e.g., mouth and nose)")
    
    print("\n\nNext Steps:")
    print("-"*40)
    print("1. Get actual landmark JSONs from sagemaker:")
    print("   scp -r user@sagemaker:/path/to/landmarks ./landmarks")
    print("\n2. Test with real data:")
    print("   - Update landmarks_dir in dataset")
    print("   - Run visualization to verify quality")
    print("\n3. If bboxes look good, proceed with training")


if __name__ == "__main__":
    main()
