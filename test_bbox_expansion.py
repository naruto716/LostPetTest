"""
Test different bbox expansion ratios on valid images.
Visualize 100 images with expanded bboxes for manual inspection.
"""

import json
import os
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont


# Region-specific expansion ratios (adjust these after inspection)
EXPANSION_RATIOS = {
    'left_eye': 0.4,
    'right_eye': 0.4,
    'nose': 0.15,
    'mouth': 0.18,
    'left_ear': 0.2,
    'right_ear': 0.2,
    'forehead': 0.2
}


def expand_bbox(bbox, expansion_ratio, img_width, img_height):
    """
    Expand bbox by percentage while staying within image bounds.
    
    Args:
        bbox: Original bbox dict with x_min, y_min, x_max, y_max, width, height
        expansion_ratio: Percentage to expand (e.g., 0.2 = 20%)
        img_width, img_height: Image dimensions for boundary checking
    
    Returns:
        Expanded bbox dict
    """
    width = bbox['width']
    height = bbox['height']
    
    # Calculate expansion in pixels (split evenly on both sides)
    expand_w = int(width * expansion_ratio / 2)
    expand_h = int(height * expansion_ratio / 2)
    
    # Expand bbox
    x_min = max(0, bbox['x_min'] - expand_w)
    y_min = max(0, bbox['y_min'] - expand_h)
    x_max = min(img_width, bbox['x_max'] + expand_w)
    y_max = min(img_height, bbox['y_max'] + expand_h)
    
    return {
        'x_min': x_min,
        'y_min': y_min,
        'x_max': x_max,
        'y_max': y_max,
        'width': x_max - x_min,
        'height': y_max - y_min
    }


def visualize_with_expansion(img_path, landmarks, output_path, expansion_ratios):
    """Create visualization with original and expanded bboxes."""
    try:
        # Load image
        img = Image.open(img_path).convert('RGB')
        img_width, img_height = img.size
        
        # Colors
        colors = {
            'left_eye': '#FF0000',
            'right_eye': '#0000FF',
            'nose': '#00FF00',
            'mouth': '#FFFF00',
            'left_ear': '#FF00FF',
            'right_ear': '#FFA500',
            'forehead': '#00FFFF'
        }
        
        region_order = ['left_eye', 'right_eye', 'nose', 'mouth', 'left_ear', 'right_ear', 'forehead']
        
        # Create figure
        fig_width = img.width + 400
        fig_height = max(img.height, 600)
        fig = Image.new('RGB', (fig_width, fig_height), 'white')
        
        # Draw bboxes
        img_with_boxes = img.copy()
        draw = ImageDraw.Draw(img_with_boxes)
        
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf", 12)
            font_small = ImageFont.truetype("/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf", 10)
        except:
            font = None
            font_small = None
        
        if 'region_bboxes' in landmarks:
            for region, orig_bbox in landmarks['region_bboxes'].items():
                color = colors.get(region, 'white')
                
                # Get expansion ratio for this region
                expansion = expansion_ratios.get(region, 0.2)
                
                # Expand bbox
                expanded_bbox = expand_bbox(orig_bbox, expansion, img_width, img_height)
                
                # Draw original bbox (thin, dashed)
                draw.rectangle([
                    orig_bbox['x_min'], orig_bbox['y_min'],
                    orig_bbox['x_max'], orig_bbox['y_max']
                ], outline='gray', width=1)
                
                # Draw expanded bbox (thick, colored)
                draw.rectangle([
                    expanded_bbox['x_min'], expanded_bbox['y_min'],
                    expanded_bbox['x_max'], expanded_bbox['y_max']
                ], outline=color, width=3)
                
                # Label
                label = f"{region.replace('_', ' ').title()} (+{int(expansion*100)}%)"
                label_y = max(expanded_bbox['y_min'] - 25, 5)
                
                if font:
                    text_bbox = draw.textbbox((expanded_bbox['x_min'], label_y), label, font=font)
                    draw.rectangle(text_bbox, fill='black')
                    draw.text((expanded_bbox['x_min'], label_y), label, fill=color, font=font)
                else:
                    draw.text((expanded_bbox['x_min'], label_y), label, fill=color)
        
        # Title
        title = f"{os.path.basename(img_path)} - Gray=Original, Color=Expanded"
        draw.text((10, 10), title, fill='white', font=font)
        draw.text((9, 9), title, fill='black', font=font)
        
        fig.paste(img_with_boxes, (0, 0))
        
        # Extract regions using EXPANDED bboxes
        x_offset = img.width + 20
        y_offset = 30
        region_size = 80
        
        regions_draw = ImageDraw.Draw(fig)
        regions_draw.text((x_offset, 5), "Expanded Regions:", fill='black', font=font)
        
        if 'region_bboxes' in landmarks:
            for i, region in enumerate(region_order):
                if region in landmarks['region_bboxes']:
                    orig_bbox = landmarks['region_bboxes'][region]
                    expansion = expansion_ratios.get(region, 0.2)
                    expanded_bbox = expand_bbox(orig_bbox, expansion, img_width, img_height)
                    
                    # Crop using EXPANDED bbox
                    region_img = img.crop((
                        expanded_bbox['x_min'], expanded_bbox['y_min'],
                        expanded_bbox['x_max'], expanded_bbox['y_max']
                    ))
                    
                    row = i // 2
                    col = i % 2
                    x = x_offset + col * (region_size + 100)
                    y = y_offset + row * (region_size + 40)
                    
                    region_img.thumbnail((region_size, region_size), Image.Resampling.LANCZOS)
                    
                    border_img = Image.new('RGB', 
                                         (region_img.width + 6, region_img.height + 6), 
                                         colors.get(region, 'black'))
                    border_img.paste(region_img, (3, 3))
                    
                    fig.paste(border_img, (x, y))
                    
                    # Show original and expanded sizes
                    label = region.replace('_', ' ').title()
                    orig_size = f"{orig_bbox['width']}×{orig_bbox['height']}"
                    exp_size = f"{expanded_bbox['width']}×{expanded_bbox['height']}"
                    
                    regions_draw.text((x, y + border_img.height + 5), label, 
                                    fill=colors.get(region, 'black'), font=font_small)
                    regions_draw.text((x, y + border_img.height + 18), f"Was: {orig_size}", 
                                    fill='gray', font=font_small)
                    regions_draw.text((x, y + border_img.height + 30), f"Now: {exp_size}", 
                                    fill='black', font=font_small)
        
        fig.save(output_path)
        return True
        
    except Exception as e:
        print(f"Error: {e}")
        return False


def test_expansion_on_100(
    valid_json="petface_valid_images_with_ears.json",
    image_dir="/home/sagemaker-user/LostPet/PetFace/dog",
    landmarks_dir="/home/sagemaker-user/LostPet/dogface_landmark_estimation_hrcnn/petface_landmarks_json_all",
    output_dir="bbox_expansion_test"
):
    """Test bbox expansion on first 100 valid images."""
    
    # Load valid images
    with open(valid_json, 'r') as f:
        data = json.load(f)
    
    dog_images = data['dog_images']
    
    print("Testing Bbox Expansion")
    print("="*60)
    print(f"Expansion ratios:")
    for region, ratio in EXPANSION_RATIOS.items():
        print(f"  {region:12} - {ratio*100:.0f}%")
    print("="*60)
    print()
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Get first 100 images
    all_images = []
    for dog_id, photo_ids in sorted(dog_images.items()):
        for photo_id in photo_ids:
            all_images.append((dog_id, photo_id))
    
    images_to_test = all_images[:100]
    
    print(f"Testing on {len(images_to_test)} images...")
    print(f"Output: {output_dir}/\n")
    
    success = 0
    for i, (dog_id, photo_id) in enumerate(images_to_test, 1):
        img_path = Path(image_dir) / dog_id / f"{photo_id}.png"
        landmark_path = Path(landmarks_dir) / f"{dog_id}_{photo_id}.json"
        
        if not img_path.exists() or not landmark_path.exists():
            continue
        
        with open(landmark_path, 'r') as f:
            landmarks = json.load(f)
        
        output_path = os.path.join(output_dir, f"expand_{i:03d}_{dog_id}_{photo_id}.png")
        
        if visualize_with_expansion(img_path, landmarks, output_path, EXPANSION_RATIOS):
            success += 1
            if i <= 10 or i % 10 == 0:
                print(f"[{i:3d}/100] {dog_id}/{photo_id} ✓")
    
    print()
    print("="*60)
    print(f"✅ Created {success}/100 visualizations")
    print(f"📁 Check: {output_dir}/")
    print()
    print("In each image:")
    print("  - Gray boxes = Original tight bboxes")
    print("  - Colored boxes = Expanded bboxes")
    print("  - Right panel = Extracted regions with expanded bboxes")
    print()
    print("After inspecting, adjust EXPANSION_RATIOS at top of script")
    print("="*60)


if __name__ == "__main__":
    test_expansion_on_100()
