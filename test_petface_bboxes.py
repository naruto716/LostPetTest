"""
Test bounding boxes on first 100 PetFace images.
Simple and focused - just load images and visualize bboxes.
"""

import os
import json
import csv
from pathlib import Path


def test_petface_bboxes(
    csv_path="splits_petface/train.csv",
    image_dir="images",
    landmarks_dir=None,  # Set this when you have landmarks
    num_images=100,
    output_dir="bbox_visualizations"
):
    """
    Test bboxes on PetFace images.
    
    Args:
        csv_path: Path to PetFace CSV
        image_dir: Path to images directory
        landmarks_dir: Path to landmark JSONs (when available)
        num_images: Number of images to test
        output_dir: Where to save visualizations
    """
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load CSV data
    print(f"Loading PetFace data from: {csv_path}")
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        data = list(reader)[:num_images]
    
    print(f"Testing {len(data)} images...\n")
    
    # Stats tracking
    found_landmarks = 0
    missing_landmarks = []
    bbox_stats = {region: [] for region in 
                 ['left_eye', 'right_eye', 'nose', 'mouth', 'left_ear', 'right_ear', 'forehead']}
    
    # Process each image
    for i, row in enumerate(data):
        img_path = Path(image_dir) / row['img_rel_path']
        
        # Extract dog_id and photo_id
        parts = row['img_rel_path'].split('/')
        dog_id = parts[0]
        photo_id = parts[1].split('.')[0]
        
        print(f"[{i+1:3d}/{len(data)}] {dog_id}/{photo_id}", end="")
        
        # Check if image exists
        if not img_path.exists():
            print(" - Image not found!")
            continue
            
        # Try to load landmarks
        if landmarks_dir:
            landmark_path = Path(landmarks_dir) / f"{dog_id}_{photo_id}.json"
            
            # Try without leading zeros if not found
            if not landmark_path.exists():
                dog_id_int = str(int(dog_id))
                photo_id_int = str(int(photo_id))
                landmark_path = Path(landmarks_dir) / f"{dog_id_int}_{photo_id_int}.json"
            
            if landmark_path.exists():
                with open(landmark_path, 'r') as f:
                    landmarks = json.load(f)
                found_landmarks += 1
                print(" ✓ Landmarks found")
                
                # Collect bbox statistics
                if 'region_bboxes' in landmarks:
                    for region, bbox in landmarks['region_bboxes'].items():
                        if region in bbox_stats:
                            bbox_stats[region].append({
                                'width': bbox['width'],
                                'height': bbox['height'],
                                'area': bbox['width'] * bbox['height']
                            })
                
                # Visualize first 10 images
                if i < 10:
                    visualize_path = f"bbox_vis_{dog_id}_{photo_id}.png"
                    if visualize_if_available(img_path, landmarks, visualize_path):
                        print(f" → Saved visualization: {visualize_path}")
                    else:
                        print(" → Visualization skipped (PIL not available)")
                
            else:
                print(" ✗ No landmarks")
                missing_landmarks.append(f"{dog_id}_{photo_id}")
        else:
            print(" - No landmarks directory provided")
    
    # Print summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    
    if landmarks_dir:
        print(f"Found landmarks: {found_landmarks}/{len(data)}")
        print(f"Missing landmarks: {len(missing_landmarks)}")
        
        if missing_landmarks and len(missing_landmarks) < 20:
            print("\nMissing landmark files:")
            for missing in missing_landmarks[:10]:
                print(f"  - {missing}.json")
            if len(missing_landmarks) > 10:
                print(f"  ... and {len(missing_landmarks) - 10} more")
        
        # Print bbox statistics
        if found_landmarks > 0:
            print("\nBounding Box Statistics:")
            print("-"*50)
            print(f"{'Region':12} {'Avg Width':>10} {'Avg Height':>11} {'Avg Area':>10}")
            print("-"*50)
            
            for region, stats in bbox_stats.items():
                if stats:
                    avg_width = sum(s['width'] for s in stats) / len(stats)
                    avg_height = sum(s['height'] for s in stats) / len(stats)
                    avg_area = sum(s['area'] for s in stats) / len(stats)
                    print(f"{region:12} {avg_width:10.1f} {avg_height:11.1f} {avg_area:10.0f}")
    else:
        print("⚠️  No landmarks directory provided")
        print("\nTo test with landmarks:")
        print("1. Copy landmarks from sagemaker:")
        print("   scp -r user@sagemaker:/path/to/petface_landmarks_json_all ./landmarks")
        print("\n2. Run again with landmarks_dir:")
        print("   test_petface_bboxes(landmarks_dir='./landmarks')")


def visualize_if_available(img_path, landmarks, output_path):
    """
    Create comprehensive visualization with both bounding boxes and cropped regions.
    """
    try:
        from PIL import Image, ImageDraw, ImageFont
        import numpy as np
        
        # Load image
        img = Image.open(img_path).convert('RGB')
        
        # Define colors - bright and distinct
        colors = {
            'left_eye': '#FF0000',     # Red
            'right_eye': '#0000FF',    # Blue
            'nose': '#00FF00',         # Green
            'mouth': '#FFFF00',        # Yellow
            'left_ear': '#FF00FF',     # Magenta
            'right_ear': '#FFA500',    # Orange
            'forehead': '#00FFFF'      # Cyan
        }
        
        # Region order for display
        region_order = ['left_eye', 'right_eye', 'nose', 'mouth', 'left_ear', 'right_ear', 'forehead']
        
        # Create figure with multiple panels
        # Left: original with bboxes, Right: grid of cropped regions
        fig_width = img.width + 400  # Extra space for regions
        fig_height = max(img.height, 600)
        
        # Create white background
        fig = Image.new('RGB', (fig_width, fig_height), 'white')
        
        # Paste original image with bboxes
        img_with_boxes = img.copy()
        draw = ImageDraw.Draw(img_with_boxes)
        
        # Try to load a font
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf", 14)
            font_small = ImageFont.truetype("/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf", 10)
        except:
            font = None
            font_small = None
        
        # Draw bboxes on left image
        if 'region_bboxes' in landmarks:
            for region, bbox in landmarks['region_bboxes'].items():
                color = colors.get(region, 'white')
                
                # Draw rectangle with thick border
                draw.rectangle([
                    bbox['x_min'], bbox['y_min'],
                    bbox['x_max'], bbox['y_max']
                ], outline=color, width=3)
                
                # Add label
                label = region.replace('_', ' ').title()
                label_y = max(bbox['y_min'] - 20, 5)
                
                if font:
                    text_bbox = draw.textbbox((bbox['x_min'], label_y), label, font=font)
                    draw.rectangle(text_bbox, fill='black')
                    draw.text((bbox['x_min'], label_y), label, fill=color, font=font)
                else:
                    draw.text((bbox['x_min'], label_y), label, fill=color)
        
        # Add title
        title_text = f"{os.path.basename(img_path)} - Bounding Boxes"
        draw.text((10, 10), title_text, fill='white', font=font)
        draw.text((9, 9), title_text, fill='black', font=font)  # Shadow
        
        # Paste to figure
        fig.paste(img_with_boxes, (0, 0))
        
        # Extract and display regions on the right
        x_offset = img.width + 20
        y_offset = 20
        region_size = 80  # Target size for each region display
        
        # Add title for regions
        regions_draw = ImageDraw.Draw(fig)
        regions_draw.text((x_offset, 5), "Extracted Regions:", fill='black', font=font)
        
        if 'region_bboxes' in landmarks:
            for i, region in enumerate(region_order):
                if region in landmarks['region_bboxes']:
                    bbox = landmarks['region_bboxes'][region]
                    
                    # Crop region from original image
                    region_img = img.crop((
                        bbox['x_min'], bbox['y_min'],
                        bbox['x_max'], bbox['y_max']
                    ))
                    
                    # Calculate position in grid (2 columns)
                    row = i // 2
                    col = i % 2
                    x = x_offset + col * (region_size + 100)
                    y = y_offset + row * (region_size + 40)
                    
                    # Resize region to fit display size while maintaining aspect ratio
                    region_img.thumbnail((region_size, region_size), Image.Resampling.LANCZOS)
                    
                    # Draw border around region
                    border_img = Image.new('RGB', 
                                         (region_img.width + 6, region_img.height + 6), 
                                         colors.get(region, 'black'))
                    border_img.paste(region_img, (3, 3))
                    
                    # Paste to figure
                    fig.paste(border_img, (x, y))
                    
                    # Add label
                    label = region.replace('_', ' ').title()
                    size_text = f"{bbox['width']}×{bbox['height']}"
                    
                    regions_draw.text((x, y + border_img.height + 5), label, 
                                    fill=colors.get(region, 'black'), font=font_small)
                    regions_draw.text((x, y + border_img.height + 20), size_text, 
                                    fill='gray', font=font_small)
        
        # Save combined visualization
        fig.save(output_path)
        return True
        
    except ImportError:
        return False


if __name__ == "__main__":
    # Run on sagemaker with PetFace data
    print("Testing PetFace Bounding Boxes on First 100 Images")
    print("="*60)
    
    test_petface_bboxes(
        csv_path="splits_petface/train.csv",
        image_dir="/home/sagemaker-user/LostPet/PetFace/dog",
        landmarks_dir="/home/sagemaker-user/LostPet/dogface_landmark_estimation_hrcnn/petface_landmarks_json_all",
        num_images=100,
        output_dir="bbox_visualizations"
    )
