# FLIR Thermal Object Detection with YOLOv8

A complete object detection pipeline built on the Teledyne FLIR ADAS thermal 
dataset using Ultralytics YOLOv8. This project was developed as part of a 
structured OpenCV and deep learning learning journey, transitioning from 
MVTec Halcon to Python-based computer vision.

## What This Project Covers

- Converting FLIR ADAS annotations from COCO JSON format to YOLO txt format
- Cleaning and remapping class labels to 4 real annotated classes
- Organising the dataset into YOLOv8-compatible folder structure
- Training YOLOv8s on thermal grayscale images using a CUDA-enabled GPU
- Validating model performance using mAP, precision, and recall metrics
- Running inference on thermal images and video frame sequences
- Visualising training curves and per-class detection results

## Dataset

Teledyne FLIR ADAS Thermal Dataset — 7,860 training images, 1,366 val images.

| Class   | Annotations |
|---------|-------------|
| Car     | 41,260      |
| Person  | 22,372      |
| Bicycle | 3,986       |
| Dog     | 226         |

## Results

Trained for 53 epochs on a GTX 1050 Ti (4GB VRAM). Early stopping triggered 
at epoch 53, best weights saved at epoch 33.

| Class   | mAP@50 |
|---------|--------|
| Car     | 0.916  |
| Person  | 0.845  |
| Bicycle | 0.635  |
| Dog     | 0.000  |
| **Overall** | **0.599** |

## Environment

- Python 3.11.9
- OpenCV 4.x (contrib)
- PyTorch 2.5.1 + CUDA 12.1
- Ultralytics YOLOv8s
- GPU: NVIDIA GeForce GTX 1050 Ti

## Project Structure

FLIR_YOLO/
    images/
        train/        <- thermal JPEG images
        val/
    labels/
        train/        <- YOLO format .txt annotations
        val/
    flir.yaml         <- dataset configuration
    convert_flir_to_yolo_clean.py
    organize_clean.py
    verify_clean.py
    train_flir.py
    validate_flir.py
    inference_flir.py
    plot_results.py


Here's your complete requirements.txt for this project:
txt# requirements.txt
# FLIR Thermal Object Detection with YOLOv8
# Python 3.11.9

# ── Core Computer Vision ──────────────────────────────────────────────
opencv-contrib-python
numpy
matplotlib
pillow
scipy
scikit-image

# ── Deep Learning ─────────────────────────────────────────────────────
# NOTE: Install PyTorch separately using the command below
# pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
ultralytics

# ── Data Handling ─────────────────────────────────────────────────────
pandas
tifffile
pycocotools

Save this as requirements.txt in your project root, then add a note in your README under a new Installation section:
markdown## Installation

1. Create and activate virtual environment:
python -m venv venv
venv\Scripts\activate

2. Install PyTorch with CUDA support:
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

3. Install remaining dependencies:
pip install -r requirements.txt

PyTorch is kept separate from requirements.txt because the install URL depends on your CUDA version — someone with a different GPU or CPU-only setup would need a different command.