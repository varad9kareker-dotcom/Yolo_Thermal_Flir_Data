# organize_dataset.py
import shutil
from pathlib import Path

base    = Path(r'D:\YOLO_Custom\Lesson_3\FLIR_ADAS_1_3')
out     = Path(r'D:\YOLO_Custom\Lesson_3\FLIR_YOLO')

# Define splits
splits = {
    'train': base / 'train',
    'val':   base / 'val',
}

for split_name, split_dir in splits.items():
    img_out = out / 'images' / split_name
    lbl_out = out / 'labels' / split_name
    img_out.mkdir(parents=True, exist_ok=True)
    lbl_out.mkdir(parents=True, exist_ok=True)

    # Copy images
    img_src = split_dir / 'data'
    count = 0
    for img_path in img_src.glob('*.jpeg'):
        shutil.copy(img_path, img_out / img_path.name)
        count += 1
    print(f"{split_name}: copied {count} images")

    # Copy labels
    lbl_src = split_dir / 'labels'
    count = 0
    for lbl_path in lbl_src.glob('*.txt'):
        shutil.copy(lbl_path, lbl_out / lbl_path.name)
        count += 1
    print(f"{split_name}: copied {count} labels")

print("\nFinal structure ready at D:/YOLO_Custom/FLIR_YOLO/")