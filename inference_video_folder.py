# inference_video_folder.py
import cv2
from pathlib import Path
from ultralytics import YOLO

# ── Load model ────────────────────────────────────────────────────────
model = YOLO('D:/YOLO_Custom/runs/detect/flir_4class_v13/weights/best.pt')

CLASS_NAMES = ['person', 'bicycle', 'car', 'dog']
COLORS = [
    ( 80,  80, 255),  # person  - red
    ( 80, 255,  80),  # bicycle - green
    (255, 150,  80),  # car     - blue
    (  0, 200, 255),  # dog     - yellow
]

# ── Folder ────────────────────────────────────────────────────────────
img_dir = Path('D:/YOLO_Custom/Lesson_3/FLIR_ADAS_1_3/video/thermal_8_bit')

# Collect all images (common thermal formats)
image_extensions = ['.jpeg', '.jpg', '.png', '.bmp', '.tiff', '.tif']
all_images = []
for ext in image_extensions:
    all_images.extend(img_dir.glob(f'*{ext}'))
all_images = sorted(all_images)  # sort by filename for sequential order

print(f"Found {len(all_images)} images in {img_dir}")
print("Controls: SPACE = next | Q = quit | S = save current frame")
print("-" * 55)

# ── Output folder for saved frames ───────────────────────────────────
save_dir = Path('D:/YOLO_Custom/Lesson_3/FLIR_results')
save_dir.mkdir(parents=True, exist_ok=True)

# ── Main loop ─────────────────────────────────────────────────────────
for idx, img_path in enumerate(all_images):

    img = cv2.imread(str(img_path))
    if img is None:
        print(f"  [SKIP] Cannot load: {img_path.name}")
        continue

    # Convert grayscale to BGR if needed
    if len(img.shape) == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    # ── Run detection ─────────────────────────────────────────────────
    results = model.predict(img, conf=0.4, verbose=False)
    boxes   = results[0].boxes

    # ── Draw detections ───────────────────────────────────────────────
    counts = {name: 0 for name in CLASS_NAMES}

    for box in boxes:
        cls      = int(box.cls)
        conf_val = float(box.conf)
        x1, y1, x2, y2 = map(int, box.xyxy[0])

        color = COLORS[cls % len(COLORS)]
        label = f"{CLASS_NAMES[cls]} {conf_val:.2f}"

        # Bounding box
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

        # Label background
        label_w = len(label) * 11
        cv2.rectangle(img, (x1, y1 - 22), (x1 + label_w, y1), color, -1)

        # Label text
        cv2.putText(img, label, (x1 + 2, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1, cv2.LINE_AA)

        counts[CLASS_NAMES[cls]] += 1

    # ── Top-left: image info ──────────────────────────────────────────
    cv2.rectangle(img, (0, 0), (260, 30), (0, 0, 0), -1)
    cv2.putText(img, f"[{idx+1}/{len(all_images)}] {img_path.name}",
                (6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)

    # ── Top-right: detection counts ───────────────────────────────────
    H, W = img.shape[:2]
    y_pos = 25
    for name, count in counts.items():
        if count > 0:
            color = COLORS[CLASS_NAMES.index(name)]
            cv2.putText(img, f"{name}: {count}",
                        (W - 150, y_pos),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)
            y_pos += 25

    # ── Bottom bar: controls hint ─────────────────────────────────────
    cv2.rectangle(img, (0, H - 25), (W, H), (30, 30, 30), -1)
    cv2.putText(img, "SPACE: next   S: save   Q: quit",
                (6, H - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1)

    # ── Console summary ───────────────────────────────────────────────
    detected = {k: v for k, v in counts.items() if v > 0}
    print(f"  [{idx+1:>4}/{len(all_images)}] {img_path.name:<40} "
          f"detections: {detected if detected else 'none'}")

    # ── Display ───────────────────────────────────────────────────────
    cv2.imshow('FLIR Thermal Detection', img)

    key = cv2.waitKey(200) & 0xFF

    if key == ord('q'):
        print("\nQuit by user.")
        break
    elif key == ord('s'):
        save_path = save_dir / f"result_{img_path.stem}.jpg"
        cv2.imwrite(str(save_path), img)
        print(f"         Saved -> {save_path}")

cv2.destroyAllWindows()
print(f"\nDone. Saved results in: {save_dir}")