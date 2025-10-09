"""
Test script for validating PetFace bounding boxes from landmark JSONs.
Creates visualizations to verify region extraction quality.
"""

import os
import json
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
import matplotlib.patches as patches


def load_sample_landmarks():
    """Load a sample landmark JSON to understand the structure."""
    sample_json = """
    {
      "image_path": "/home/sagemaker-user/LostPet/PetFace/dog/029364/02.png",
      "image_width": 224,
      "image_height": 224,
      "landmarks": [],
      "region_bboxes": {
        "left_eye": {
          "x_min": 147,
          "y_min": 88,
          "x_max": 169,
          "y_max": 92,
          "width": 22,
          "height": 4
        },
        "right_eye": {
          "x_min": 56,
          "y_min": 89,
          "x_max": 73,
          "y_max": 102,
          "width": 17,
          "height": 13
        },
        "nose": {
          "x_min": 81,
          "y_min": 135,
          "x_max": 141,
          "y_max": 188,
          "width": 60,
          "height": 53
        },
        "mouth": {
          "x_min": 49,
          "y_min": 104,
          "x_max": 178,
          "y_max": 207,
          "width": 129,
          "height": 103
        },
        "left_ear": {
          "x_min": 183,
          "y_min": 16,
          "x_max": 222,
          "y_max": 169,
          "width": 39,
          "height": 153
        },
        "right_ear": {
          "x_min": 3,
          "y_min": 4,
          "x_max": 48,
          "y_max": 200,
          "width": 45,
          "height": 196
        },
        "forehead": {
          "x_min": 30,
          "y_min": 14,
          "x_max": 201,
          "y_max": 97,
          "width": 171,
          "height": 83
        }
      },
      "avg_confidence": 0.5892922878265381,
      "visible_landmarks": 42
    }
    """
    return json.loads(sample_json)


def visualize_bboxes(image_path, landmarks_data, save_path=None):
    """Visualize bounding boxes on an image."""
    # Load image
    image = Image.open(image_path).convert('RGB')
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes = axes.flatten()
    
    # Define colors for each region
    region_colors = {
        'left_eye': '#FF6B6B',     # Red
        'right_eye': '#4ECDC4',    # Cyan
        'nose': '#95E1D3',         # Light green
        'mouth': '#FFE66D',        # Yellow
        'left_ear': '#C7CEEA',     # Purple
        'right_ear': '#FECA57',    # Orange
        'forehead': '#FF9FF3'      # Pink
    }
    
    # Plot 1: Original image with all bboxes
    ax = axes[0]
    ax.imshow(image)
    ax.set_title("All Regions", fontsize=14, fontweight='bold')
    ax.axis('off')
    
    # Draw all bboxes
    region_bboxes = landmarks_data.get('region_bboxes', {})
    for region_name, bbox in region_bboxes.items():
        rect = patches.Rectangle(
            (bbox['x_min'], bbox['y_min']),
            bbox['width'], bbox['height'],
            linewidth=2, 
            edgecolor=region_colors.get(region_name, 'black'),
            facecolor='none',
            alpha=0.8
        )
        ax.add_patch(rect)
        
        # Add label
        ax.text(
            bbox['x_min'], bbox['y_min'] - 3,
            region_name, 
            color=region_colors.get(region_name, 'black'),
            fontsize=9, 
            fontweight='bold',
            bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.7)
        )
    
    # Plot individual regions
    regions = ['left_eye', 'right_eye', 'nose', 'mouth', 'left_ear', 'right_ear', 'forehead']
    for i, region_name in enumerate(regions):
        ax = axes[i + 1]
        
        if region_name in region_bboxes:
            bbox = region_bboxes[region_name]
            
            # Crop region from image
            region_img = image.crop((
                bbox['x_min'], bbox['y_min'],
                bbox['x_max'], bbox['y_max']
            ))
            
            ax.imshow(region_img)
            ax.set_title(
                f"{region_name}\n{bbox['width']}×{bbox['height']}px",
                fontsize=11,
                color=region_colors.get(region_name, 'black')
            )
        else:
            ax.text(0.5, 0.5, 'Not found', ha='center', va='center', transform=ax.transAxes)
            ax.set_title(region_name, fontsize=11)
        
        ax.axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Saved visualization to: {save_path}")
    else:
        plt.show()


def analyze_bbox_statistics(landmarks_data):
    """Analyze bbox statistics from landmarks."""
    print("\nBounding Box Statistics:")
    print("-" * 50)
    
    region_bboxes = landmarks_data.get('region_bboxes', {})
    
    for region_name, bbox in region_bboxes.items():
        area = bbox['width'] * bbox['height']
        aspect_ratio = bbox['width'] / bbox['height'] if bbox['height'] > 0 else 0
        
        print(f"{region_name:12} - Size: {bbox['width']:3d} × {bbox['height']:3d}")
        print(f"{'':12}   Area: {area:6d} px²")
        print(f"{'':12}   Aspect ratio: {aspect_ratio:.2f}")
        print(f"{'':12}   Position: ({bbox['x_min']}, {bbox['y_min']})")
        print()


def test_with_actual_image():
    """Test bbox extraction with an actual PetFace image."""
    # Paths
    image_dir = Path("/Users/michael/Projects/LostPetTest/images")
    
    # Find a sample image
    train_images = list(image_dir.glob("train/*.jpg"))
    if not train_images:
        print("No training images found!")
        return
    
    # Use first available image
    image_path = train_images[0]
    print(f"Testing with image: {image_path}")
    
    # Load sample landmarks (since we don't have actual JSONs locally)
    landmarks_data = load_sample_landmarks()
    
    # Scale landmarks to match actual image size
    image = Image.open(image_path)
    scale_x = image.width / landmarks_data['image_width']
    scale_y = image.height / landmarks_data['image_height']
    
    print(f"Image size: {image.width}×{image.height}")
    print(f"Scale factors: {scale_x:.2f}×{scale_y:.2f}")
    
    # Scale all bboxes
    for region_name, bbox in landmarks_data['region_bboxes'].items():
        bbox['x_min'] = int(bbox['x_min'] * scale_x)
        bbox['y_min'] = int(bbox['y_min'] * scale_y)
        bbox['x_max'] = int(bbox['x_max'] * scale_x)
        bbox['y_max'] = int(bbox['y_max'] * scale_y)
        bbox['width'] = bbox['x_max'] - bbox['x_min']
        bbox['height'] = bbox['y_max'] - bbox['y_min']
    
    # Visualize
    save_path = "bbox_test_visualization.png"
    visualize_bboxes(image_path, landmarks_data, save_path)
    
    # Analyze statistics
    analyze_bbox_statistics(landmarks_data)


def main():
    """Main test function."""
    print("="*60)
    print("PetFace Bounding Box Test")
    print("="*60)
    
    # Test with actual image
    test_with_actual_image()
    
    print("\n" + "="*60)
    print("Next Steps:")
    print("="*60)
    print("1. Copy actual landmark JSONs from sagemaker")
    print("2. Run this script with real landmarks to verify quality")
    print("3. Check if any bboxes need adjustment")
    print("4. Proceed with regional feature extraction")


if __name__ == "__main__":
    main()
