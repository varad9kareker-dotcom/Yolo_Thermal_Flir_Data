import time
from ultralytics import YOLO

start_time = None

def on_train_start(trainer):
    global start_time
    start_time = time.time()

def on_train_epoch_end(trainer):
    elapsed = time.time() - start_time
    epoch = trainer.epoch + 1
    total_epochs = trainer.args.epochs

    # average time per epoch
    avg_time = elapsed / epoch

    # estimated remaining time
    remaining_epochs = total_epochs - epoch
    eta = avg_time * remaining_epochs

    # format ETA
    hrs = int(eta // 3600)
    mins = int((eta % 3600) // 60)
    secs = int(eta % 60)

    print(f"⏳ ETA: {hrs:02d}:{mins:02d}:{secs:02d}")

def main():
    model = YOLO('yolov8s.pt')

    model.add_callback("on_train_start", on_train_start)
    model.add_callback("on_train_epoch_end", on_train_epoch_end)

    model.train(
        data='D:/YOLO_Custom/Lesson_3/FLIR_YOLO/flir.yaml',
        epochs=100,
        imgsz=640,
        batch=8,
        name='flir_4class_v1',
        device=0,
        patience=20,
        workers=2,
        pretrained=True,
        verbose=True
    )

if __name__ == "__main__":
    main()