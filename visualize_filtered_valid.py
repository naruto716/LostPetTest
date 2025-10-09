"""
Visualize first 100 images from the filtered valid dataset.
For manual quality checking.
"""

import json
import os
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont


def visualize_image(img_path, landmarks, output_path):
    """Create visualization with bboxes and cropped regions."""
    try:
        from PIL import Image, ImageDraw, ImageFont
        
        # Load image
        img = Image.open(img_path).convert('RGB')
        
        # Colors for each region
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
        
        # Create combined figure
        fig_width = img.width + 400
        fig_height = max(img.height, 600)
        fig = Image.new('RGB', (fig_width, fig_height), 'white')
        
        # Draw bboxes on left image
        img_with_boxes = img.copy()
        draw = ImageDraw.Draw(img_with_boxes)
        
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf", 14)
            font_small = ImageFont.truetype("/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf", 10)
        except:
            font = None
            font_small = None
        
        if 'region_bboxes' in landmarks:
            for region, bbox in landmarks['region_bboxes'].items():
                color = colors.get(region, 'white')
                
                draw.rectangle([
                    bbox['x_min'], bbox['y_min'],
                    bbox['x_max'], bbox['y_max']
                ], outline=color, width=3)
                
                label = region.replace('_', ' ').title()
                label_y = max(bbox['y_min'] - 20, 5)
                
                if font:
                    text_bbox = draw.textbbox((bbox['x_min'], label_y), label, font=font)
                    draw.rectangle(text_bbox, fill='black')
                    draw.text((bbox['x_min'], label_y), label, fill=color, font=font)
                else:
                    draw.text((bbox['x_min'], label_y), label, fill=color)
        
        title_text = f"{os.path.basename(img_path)}"
        draw.text((10, 10), title_text, fill='white', font=font)
        draw.text((9, 9), title_text, fill='black', font=font)
        
        fig.paste(img_with_boxes, (0, 0))
        
        # Extract and display regions on the right
        x_offset = img.width + 20
        y_offset = 20
        region_size = 80
        
        regions_draw = ImageDraw.Draw(fig)
        regions_draw.text((x_offset, 5), "Extracted Regions:", fill='black', font=font)
        
        if 'region_bboxes' in landmarks:
            for i, region in enumerate(region_order):
                if region in landmarks['region_bboxes']:
                    bbox = landmarks['region_bboxes'][region]
                    
                    region_img = img.crop((
                        bbox['x_min'], bbox['y_min'],
                        bbox['x_max'], bbox['y_max']
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
                    
                    label = region.replace('_', ' ').title()
                    size_text = f"{bbox['width']}×{bbox['height']}"
                    
                    regions_draw.text((x, y + border_img.height + 5), label, 
                                    fill=colors.get(region, 'black'), font=font_small)
                    regions_draw.text((x, y + border_img.height + 20), size_text, 
                                    fill='gray', font=font_small)
        
        fig.save(output_path)
        return True
        
    except Exception as e:
        print(f"Error visualizing: {e}")
        return False


def visualize_first_100(
    valid_json="petface_valid_images_with_ears.json",
    image_dir="/home/sagemaker-user/LostPet/PetFace/dog",
    landmarks_dir="/home/sagemaker-user/LostPet/dogface_landmark_estimation_hrcnn/petface_landmarks_json_all",
    output_dir="valid_images_check"
):
    """Visualize first 100 valid images for manual quality check."""
    
    # Load valid images JSON
    with open(valid_json, 'r') as f:
        data = json.load(f)
    
    dog_images = data['dog_images']
    
    print("Loaded valid images JSON")
    print(f"Total dogs: {len(dog_images)}")
    print(f"Total valid images: {data['metadata']['final_valid_images']}")
    print()
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Flatten to list of (dog_id, photo_id) pairs
    all_images = []
    for dog_id, photo_ids in sorted(dog_images.items()):
        for photo_id in photo_ids:
            all_images.append((dog_id, photo_id))
    
    # Take first 100
    images_to_visualize = all_images[:100]
    
    print(f"Visualizing first 100 valid images...")
    print(f"Output directory: {output_dir}/")
    print()
    
    success_count = 0
    for i, (dog_id, photo_id) in enumerate(images_to_visualize, 1):
        # Load image
        img_path = Path(image_dir) / dog_id / f"{photo_id}.png"
        
        # Load landmarks
        landmark_path = Path(landmarks_dir) / f"{dog_id}_{photo_id}.json"
        
        if not img_path.exists():
            print(f"[{i:3d}/100] {dog_id}/{photo_id} - Image not found")
            continue
        
        if not landmark_path.exists():
            print(f"[{i:3d}/100] {dog_id}/{photo_id} - Landmarks not found")
            continue
        
        with open(landmark_path, 'r') as f:
            landmarks = json.load(f)
        
        # Visualize
        output_path = os.path.join(output_dir, f"valid_{i:03d}_{dog_id}_{photo_id}.png")
        if visualize_image(img_path, landmarks, output_path):
            success_count += 1
            if i <= 10 or i % 10 == 0:
                print(f"[{i:3d}/100] {dog_id}/{photo_id} ✓")
        else:
            print(f"[{i:3d}/100] {dog_id}/{photo_id} - Visualization failed")
    
    print()
    print("="*60)
    print(f"✅ Created {success_count}/100 visualizations")
    print(f"📁 Check folder: {output_dir}/")
    print("="*60)


if __name__ == "__main__":
    visualize_first_100()
