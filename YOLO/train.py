"""
train.py - fine-tune YOLOv8 to find the green and blue LEGO minifigures.

Dataset: dataset/ (Roboflow YOLOv8 export, classes green_minifig and blue_minifig,
120 images split 84 train / 24 valid / 12 test).

We start from yolov8n.pt (the "nano" model, pretrained on the COCO dataset) and
fine-tune it on our own photos. That's much faster than training from scratch
and works well with only a hundred or so images.

Run from this folder:   python train.py
Result:                 best.pt  (copied here from runs/detect/minifigs/weights/)
"""
import shutil
from pathlib import Path

from ultralytics import YOLO

HERE = Path(__file__).parent
DATA_YAML = HERE / "dataset" / "data.yaml"
EPOCHS = 100        # small dataset, so more passes help
IMAGE_SIZE = 512    # matches the 512x512 resize we chose in Roboflow
DEVICE = "mps"      # Apple Silicon GPU; falls back to CPU below if unavailable


def train(device):
    model = YOLO("yolov8n.pt")
    model.train(
        data=str(DATA_YAML),
        epochs=EPOCHS,
        imgsz=IMAGE_SIZE,
        device=device,
        project=str(HERE / "runs" / "detect"),
        name="minifigs",
        exist_ok=True,
        patience=25,   # stop early if it stops improving
        hsv_h=0.005,   # keep color shifts tiny so green and blue don't get mixed up
    )


try:
    train(DEVICE)
except Exception as e:
    print(f"Training on device={DEVICE!r} failed ({e}). Retrying on CPU...")
    train("cpu")

best = HERE / "runs" / "detect" / "minifigs" / "weights" / "best.pt"
shutil.copy(best, HERE / "best.pt")
print(f"Done. Model saved to {HERE / 'best.pt'}")
