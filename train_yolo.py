"""Fine-tunes a YOLOv8 detector to find the green LEGO minifigure.

Run this after finishing the Roboflow steps (see Door2Door's setup notes):
take 50-100 photos of the minifig from varied angles (plus a few photos of
the scene with no minifig in them), label them in Roboflow as a single class
("Minifig"), and download the dataset in YOLOv8 format into dataset/ next to
this script -- it should contain a data.yaml plus train/valid/test folders.

Training progress and the final weights land under
runs/detect/green_minifig/weights/best.pt -- green_minifig_tracker.py loads
that file automatically if it exists, and only falls back to the generic
(untrained, color-matched) detector if it doesn't.
"""

from pathlib import Path

from ultralytics import YOLO

DATA_YAML = Path(__file__).parent / "dataset" / "data.yaml"
EPOCHS = 100  # small dataset (~50-100 images) benefits from more epochs than the usual 50
IMAGE_SIZE = 640
DEVICE = "mps"  # Apple Silicon GPU; falls back to CPU below if unavailable

if not DATA_YAML.exists():
    raise FileNotFoundError(
        f"{DATA_YAML} not found -- export your labeled dataset from Roboflow "
        "in YOLOv8 format and unzip it to dataset/ next to this script first."
    )

# Start from COCO-pretrained weights and fine-tune -- much faster than
# training from scratch, and works well with a few dozen images.
model = YOLO("yolov8n.pt")

try:
    model.train(
        data=DATA_YAML,
        epochs=EPOCHS,
        imgsz=IMAGE_SIZE,
        name="green_minifig",
        device=DEVICE,
    )
except Exception as e:
    print(f"Training on device={DEVICE!r} failed ({e}). Retrying on CPU...")
    model = YOLO("yolov8n.pt")
    model.train(
        data=DATA_YAML,
        epochs=EPOCHS,
        imgsz=IMAGE_SIZE,
        name="green_minifig",
        device="cpu",
    )

print("Done. Weights saved to runs/detect/green_minifig/weights/best.pt")
