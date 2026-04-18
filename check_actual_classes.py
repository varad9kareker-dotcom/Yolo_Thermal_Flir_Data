# check_actual_classes.py
import json
from pathlib import Path
from collections import Counter

coco_path = Path('D:/YOLO_Custom/Lesson_3/FLIR_ADAS_1_3/train/thermal_annotations.json')

with open(coco_path, 'r') as f:
    coco = json.load(f)

# Build category id -> name lookup
cat_lookup = {cat['id']: cat['name'] for cat in coco['categories']}

# Count annotations per category
counts = Counter()
for ann in coco['annotations']:
    counts[ann['category_id']] += 1

# Print only classes that actually have annotations
print("Classes with actual annotations:")
print(f"{'ID':<6} {'Count':<10} {'Name'}")
print("-" * 35)

for cat_id, count in sorted(counts.items(), key=lambda x: -x[1]):
    name = cat_lookup.get(cat_id, 'unknown')
    print(f"{cat_id:<6} {count:<10} {name}")

print(f"\nTotal annotated classes: {len(counts)}")
print(f"Total annotations: {sum(counts.values())}")