# YOLO

Trains a YOLOv8 nano model to detect two classes, `green_minifig` and
`blue_minifig`. It starts from `yolov8n.pt` (pretrained on COCO) and
fine-tunes it on our own photos.

## Dataset

Put the Roboflow YOLOv8 export in `YOLO/dataset/`, so that
`YOLO/dataset/data.yaml` exists next to the `train/`, `valid/` and
`test/` folders.

## Install

```
pip install -r requirements.txt
```

## Run

From this folder:

```
python train.py
```

It trains on the Apple Silicon GPU (`mps`) and falls back to the CPU if
that isn't available. When it finishes, the trained model is copied to
`YOLO/best.pt`. The full training output stays in `runs/`, which git
ignores.

## Detect and publish over MQTT

`detect_publish.py` runs `best.pt` on the laptop camera and publishes where
each minifig is to the `broker.hivemq.com` broker (port 1883):

| Minifig | Topic                  |
| ------- | ---------------------- |
| green   | `ME193/Mohammed/green` |
| blue    | `ME193/Mohammed/blue`  |

Each message is JSON, in the same format as the professor's MQTT Minifig
Monitor:

```
{"x": 412.0, "y": 230.5, "w": 640, "h": 480, "conf": 0.91}
```

`x`, `y` are the center of the minifig's box in pixels and `w`, `h` are
the camera frame size. Only the most confident box per color is sent, at
most 10 times a second. Nothing is sent for a minifig that isn't
detected, so the UNO Q decides what to do when messages stop arriving.

A video window shows the detections and a red line at the center of the
frame (the stopping point). Press `q` in that window to quit.

You must run it from inside this `YOLO` folder:

```
python detect_publish.py
```

Ultralytics can't load a model from a full path containing the apostrophe
in `Mohammed's Projects`, so the script loads `best.pt` by its short name.

If it opens the wrong camera (for example your iPhone), change `CAMERA`
at the top of the script to `1`.
