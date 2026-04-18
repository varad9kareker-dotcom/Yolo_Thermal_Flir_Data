# convert_flir_to_yolo_clean.py
import json
from pathlib import Path

# ── Only keep these 4 COCO category IDs ──────────────────────────────
# COCO ID -> YOLO class index
KEEP_CLASSES = {
    1:  0,   # person   -> 0
    2:  1,   # bicycle  -> 1
    3:  2,   # car      -> 2
    17: 3,   # dog      -> 3
}

CLASS_NAMES = {0: 'person', 1: 'bicycle', 2: 'car', 3: 'dog'}

def convert(coco_json_path, output_labels_dir):
    output_labels_dir = Path(output_labels_dir)
    output_labels_dir.mkdir(parents=True, exist_ok=True)

    with open(coco_json_path, 'r') as f:
        coco = json.load(f)

    # image_id -> image info
    images = {img['id']: img for img in coco['images']}

    # Group annotations by image_id, keeping only our 4 classes
    annotations_by_image = {}
    skipped = 0
    for ann in coco['annotations']:
        if ann['category_id'] not in KEEP_CLASSES:
            skipped += 1
            continue
        img_id = ann['image_id']
        if img_id not in annotations_by_image:
            annotations_by_image[img_id] = []
        annotations_by_image[img_id].append(ann)

    print(f"Skipped {skipped} annotations from unused classes")

    # Convert
    converted_images = 0
    converted_boxes  = 0

    for img_id, anns in annotations_by_image.items():
        img_info = images[img_id]
        W = img_info['width']
        H = img_info['height']

        img_stem = Path(img_info['file_name']).stem
        txt_path = output_labels_dir / f"{img_stem}.txt"

        with open(txt_path, 'w') as f:
            for ann in anns:
                cls_idx = KEEP_CLASSES[ann['category_id']]

                # COCO -> YOLO format
                x_min, y_min, bw, bh = ann['bbox']
                x_center = (x_min + bw / 2) / W
                y_center = (y_min + bh / 2) / H
                bw_norm  = bw / W
                bh_norm  = bh / H

                # Clamp to valid range
                x_center = max(0.0, min(1.0, x_center))
                y_center = max(0.0, min(1.0, y_center))
                bw_norm  = max(0.0, min(1.0, bw_norm))
                bh_norm  = max(0.0, min(1.0, bh_norm))

                f.write(f"{cls_idx} {x_center:.6f} {y_center:.6f} "
                        f"{bw_norm:.6f} {bh_norm:.6f}\n")
                converted_boxes += 1

        converted_images += 1

    # Print class breakdown
    print(f"\nConverted: {converted_images} images, {converted_boxes} boxes")
    print("Class breakdown:")
    class_counts = {0: 0, 1: 0, 2: 0, 3: 0}
    for anns in annotations_by_image.values():
        for ann in anns:
            cls_idx = KEEP_CLASSES[ann['category_id']]
            class_counts[cls_idx] += 1
    for idx, name in CLASS_NAMES.items():
        print(f"  {idx}: {name:<10} {class_counts[idx]} boxes")
    print(f"Output -> {output_labels_dir}\n")

# ── Run for train and val ─────────────────────────────────────────────
base = Path('D:/YOLO_Custom/Lesson_3/FLIR_YOLO/')

convert(
    coco_json_path    = base / 'labels/train/thermal_annotations.json',
    output_labels_dir = base / 'labels/train'
)

convert(
    coco_json_path    = base / 'labels/val/thermal_annotations.json',
    output_labels_dir = base / 'labels/val'
)

print("\nDone! All annotations converted.")