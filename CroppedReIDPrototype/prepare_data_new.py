"""
Pre-crop images based on provided bounding boxes and save them along with original images.
"""

import csv
import json
import pandas as pd
from pathlib import Path
from PIL import Image

TRAIN_SPLIT = Path("subset_splits_petface/train_subset.csv")
QUERY_SPLIT = Path("subset_splits_petface/test_query_subset.csv")
GALLERY_SPLIT = Path("subset_splits_petface/test_gallery_subset.csv")

IMAGE_DIR = Path("dog")
LABEL_DIR = Path("dog_landmarks")
OUTPUT_DIR = Path("output")

def save_precropped_images_and_csv(split, images_dir, labels_dir, original_output_dir, crop_output_dir, csv_output_path):
    """
    For each image in the split, extract all region crops and save them to crop_output_dir.
    Also save the uncropped/original image to original_output_dir.
    Write a CSV with columns: path, pid, region
    """
    crop_output_dir = Path(crop_output_dir)
    crop_output_dir.mkdir(parents=True, exist_ok=True)
    original_output_dir = Path(original_output_dir)
    original_output_dir.mkdir(parents=True, exist_ok=True)
    split_df = pd.read_csv(split)
    print(f"Split file: {split}")
    print(f"Columns: {split_df.columns.tolist()}")
    # Use the correct image path column
    if 'img_rel_path' in split_df.columns:
        path_col = 'img_rel_path'
    elif 'path' in split_df.columns:
        path_col = 'path'
    else:
        # Use the first column as fallback
        path_col = split_df.columns[0]
        print(f"Warning: 'img_rel_path' or 'path' column not found. Using '{path_col}' as image path column.")
    if 'pid' not in split_df.columns:
        # Try to guess pid column
        pid_col = [c for c in split_df.columns if 'pid' in c.lower()]
        pid_col = pid_col[0] if pid_col else split_df.columns[1]
        print(f"Warning: 'pid' column not found. Using '{pid_col}' as pid column.")
    else:
        pid_col = 'pid'
    rows = []
    for idx, row in split_df.iterrows():
        img_path = Path(images_dir) / row[path_col]
        pid = row[pid_col]
        label_path = Path(labels_dir) / (img_path.parent.name + '_' + img_path.stem + '.json')
        if not label_path.exists():
            continue
        try:
            img = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"Could not open {img_path}: {e}")
            continue
        # Save the uncropped/original image
        orig_rel_path = f"{img_path.parent.name}_{img_path.stem}.png"
        orig_save_path = original_output_dir / orig_rel_path
        img.save(orig_save_path)
        with open(label_path) as f:
            label = json.load(f)
        region_bboxes = label.get('region_bboxes', {})
        width, height = img.size
        for region, bbox in region_bboxes.items():
            # Clip bbox to image bounds
            x_min = max(0, min(width, int(bbox['x_min'])))
            y_min = max(0, min(height, int(bbox['y_min'])))
            x_max = max(0, min(width, int(bbox['x_max'])))
            y_max = max(0, min(height, int(bbox['y_max'])))
            if x_max <= x_min or y_max <= y_min:
                print(f"Warning: Invalid bbox for {img_path} region {region}: {bbox}")
                continue
            crop = img.crop((x_min, y_min, x_max, y_max))
            crop_rel_path = f"{img_path.parent.name}_{img_path.stem}_{region}.png"
            crop_save_path = crop_output_dir / crop_rel_path
            crop.save(crop_save_path)
            rows.append({'path': crop_rel_path, 'pid': pid, 'region': region})
    # Write CSV
    with open(csv_output_path, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=['path', 'pid', 'region'])
        writer.writeheader()
        for r in rows:
            writer.writerow(r)
    print(f"Saved {len(rows)} crops and CSV to {csv_output_path}")

def main():
    # Pre-crop and save crops for each split
    save_precropped_images_and_csv(
        split=TRAIN_SPLIT,
        images_dir=IMAGE_DIR,
        labels_dir=LABEL_DIR,
        original_output_dir=OUTPUT_DIR / "train",
        crop_output_dir=OUTPUT_DIR / "train_crops",
        csv_output_path=OUTPUT_DIR / "train_crops.csv"
    )
    save_precropped_images_and_csv(
        split=QUERY_SPLIT,
        images_dir=IMAGE_DIR,
        labels_dir=LABEL_DIR,
        original_output_dir=OUTPUT_DIR / "query",
        crop_output_dir=OUTPUT_DIR / "query_crops",
        csv_output_path=OUTPUT_DIR / "query_crops.csv"
    )
    save_precropped_images_and_csv(
        split=GALLERY_SPLIT,
        images_dir=IMAGE_DIR,
        labels_dir=LABEL_DIR,
        original_output_dir=OUTPUT_DIR / "gallery",
        crop_output_dir=OUTPUT_DIR / "gallery_crops",
        csv_output_path=OUTPUT_DIR / "gallery_crops.csv"
    )

if __name__ == "__main__":
    main()