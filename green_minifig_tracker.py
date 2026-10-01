"""Green Minifig Tracker: finds a green LEGO minifigure in the webcam feed
and publishes its position over MQTT for the UnoQ to act on.

Detection prefers a fine-tuned YOLOv8 model (see train_yolo.py and its
docstring for the Roboflow labeling workflow this depends on):
  - If runs/detect/green_minifig/weights/best.pt exists (produced by
    train_yolo.py), it's loaded and used directly -- it was trained on a
    single "Minifig" class, so the highest-confidence detection above
    YOLO_CONFIDENCE is reported as-is, no color filtering needed.
  - Otherwise, this falls back to the generic COCO-pretrained yolov8n.pt.
    COCO has no "minifigure" class, so each proposed box is instead scored by
    how green it is (the fraction of its pixels inside an HSV green range);
    the greenest box at or above GREEN_FRACTION_THRESHOLD is reported. If no
    box is green enough, the single largest green blob in the whole frame is
    used instead -- still a real detection, just not YOLO-confirmed. This
    fallback exists so the script is runnable (if not very reliable) before
    you've trained a real model.
The preview window draws the match in a different color per source (see
DETECTION_COLORS) so you can tell which path found it.

MQTT topic and payload match ArduinoApps/mqtt-minifig-monitor (ported from
https://github.com/mohdalmheiri/ME193-Robotics) rather than this project's
own earlier format: {"x": <pixel x>, "y": <pixel y>, "w": <frame width>,
"h": <frame height>} -- x/y are the detection box's center in pixel
coordinates, and w/h are the camera frame's own pixel dimensions (used by
the receiver only to normalize x/y into its LED grid), not the detection
box's size. A message is only published when something is actually
detected -- the UnoQ side ages out a missing minifig on its own via how long
ago the last message arrived, rather than this script sending an explicit
"not detected" message.

This script also renders its own simulated LED grid (a blue dot at the
scaled cell) in a preview window so the scaling can be sanity-checked
without the UnoQ connected -- actually driving the physical display, and the
motors, from the published MQTT message is up to the UnoQ's own code (see
ArduinoApps/mqtt-minifig-drive), not this script.

Setup:
  pip install -r requirements.txt
  First run downloads the YOLOv8n weights (yolov8n.pt, ~6 MB) from
  Ultralytics -- needs internet access once.

Calibrating:
  CAMERA_INDEX: try 0 first, then 1, 2... while watching the preview window
    to find which index is actually your camera.
  H/S/V low/high trackbars (generic-model fallback only): watch the "green
    mask" preview window and adjust so only the minifig -- not the
    background -- shows up white. GREEN_HSV_LOWER/GREEN_HSV_UPPER below are
    just the starting values loaded into those trackbars.
  GREEN_FRACTION_THRESHOLD/MIN_GREEN_AREA (generic-model fallback only):
    placeholders -- tune against false positives/negatives on your own
    minifig and lighting.

Press 'q' or close the window to quit.
"""

import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import paho.mqtt.client as mqtt
from ultralytics import YOLO

# --- MQTT -------------------------------------------------------------------
MQTT_BROKER = "test.mosquitto.org"  # shared public broker used in class
MQTT_PORT = 1883
MQTT_TOPIC = "ME193/minifig"  # must match MQTT_TOPIC in ArduinoApps/mqtt-minifig-drive/python/main.py
PUBLISH_INTERVAL_S = 0.1  # publish at most 10x/sec -- test.mosquitto.org is a
                           # shared public broker, don't flood it at full
                           # camera frame rate

# --- Camera -------------------------------------------------------------
CAMERA_INDEX = 0  # try 0 first, then 1, 2... while watching the preview
                   # window to find which index is actually your camera

# --- LED display (this script's own debug preview only) ------------------
# Matches the UnoQ's onboard matrix: 8 rows x 13 columns, confirmed both by
# its pinout doc and by ArduinoApps/mqtt-minifig-monitor's FRAME_ROWS/COLS.
LED_GRID_COLS = 13
LED_GRID_ROWS = 8
LED_PREVIEW_CELL_PX = 40  # size of each simulated LED cell in the preview window

# --- YOLO object detection ------------------------------------------------
CUSTOM_MODEL_PATH = Path(__file__).parent / "runs" / "detect" / "green_minifig" / "weights" / "best.pt"
YOLO_MODEL = "yolov8n.pt"  # generic fallback if CUSTOM_MODEL_PATH doesn't exist yet --
                           # auto-downloaded on first run
YOLO_CONFIDENCE = 0.25     # minimum detection confidence to consider a box

# --- Green color matching (generic-model fallback only) ------------------
# Starting values only -- retune live with the trackbars below while
# watching the mask preview window.
GREEN_HSV_LOWER = np.array([40, 70, 70])
GREEN_HSV_UPPER = np.array([80, 255, 255])
GREEN_FRACTION_THRESHOLD = 0.15  # a YOLO box must be at least this green
                                  # (fraction of pixels inside it) to count
MIN_GREEN_AREA = 150  # pixels -- ignores tiny green specks/noise when falling
                       # back to whole-frame contour detection

DETECTION_COLORS = {
    "custom-yolo": (0, 255, 255),
    "yolo": (0, 255, 0),
    "color-fallback": (0, 165, 255),
}


def box_green_fraction(mask: np.ndarray, box: tuple[int, int, int, int]) -> float:
    x1, y1, x2, y2 = box
    region = mask[y1:y2, x1:x2]
    if region.size == 0:
        return 0.0
    return float(np.count_nonzero(region)) / region.size


def best_confidence_box(boxes: list[tuple[int, int, int, int]], confidences: list[float]):
    if not boxes:
        return None
    best_i = max(range(len(boxes)), key=lambda i: confidences[i])
    return boxes[best_i]


def find_minifig_by_color(mask: np.ndarray, boxes: list[tuple[int, int, int, int]]):
    """Generic-model fallback: returns ((x1, y1, x2, y2), source), or
    (None, None) if nothing green enough was found. source is "yolo" if a
    YOLO-proposed box was green enough, or "color-fallback" if the largest
    green blob in the whole frame was used instead."""
    best_box, best_fraction = None, 0.0
    for box in boxes:
        fraction = box_green_fraction(mask, box)
        if fraction > best_fraction:
            best_box, best_fraction = box, fraction
    if best_box is not None and best_fraction >= GREEN_FRACTION_THRESHOLD:
        return best_box, "yolo"

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None, None
    largest = max(contours, key=cv2.contourArea)
    if cv2.contourArea(largest) < MIN_GREEN_AREA:
        return None, None
    x, y, w, h = cv2.boundingRect(largest)
    return (x, y, x + w, y + h), "color-fallback"


def box_center(box: tuple[int, int, int, int]) -> tuple[int, int]:
    x1, y1, x2, y2 = box
    return (x1 + x2) // 2, (y1 + y2) // 2


def render_led_preview(col: int, row: int) -> np.ndarray:
    img = np.zeros((LED_GRID_ROWS * LED_PREVIEW_CELL_PX, LED_GRID_COLS * LED_PREVIEW_CELL_PX, 3), dtype=np.uint8)
    for gx in range(LED_GRID_COLS + 1):
        x = gx * LED_PREVIEW_CELL_PX
        cv2.line(img, (x, 0), (x, img.shape[0]), (40, 40, 40), 1)
    for gy in range(LED_GRID_ROWS + 1):
        y = gy * LED_PREVIEW_CELL_PX
        cv2.line(img, (0, y), (img.shape[1], y), (40, 40, 40), 1)
    center = (col * LED_PREVIEW_CELL_PX + LED_PREVIEW_CELL_PX // 2, row * LED_PREVIEW_CELL_PX + LED_PREVIEW_CELL_PX // 2)
    cv2.circle(img, center, LED_PREVIEW_CELL_PX // 3, (255, 0, 0), -1)
    return img


def main():
    if CUSTOM_MODEL_PATH.exists():
        model = YOLO(str(CUSTOM_MODEL_PATH))
        using_custom_model = True
        print(f"Loaded fine-tuned minifig detector from {CUSTOM_MODEL_PATH}")
    else:
        model = YOLO(YOLO_MODEL)
        using_custom_model = False
        print(
            f"No fine-tuned model found at {CUSTOM_MODEL_PATH} -- using generic "
            f"{YOLO_MODEL} with color-based filtering instead. Run train_yolo.py "
            "after labeling your dataset in Roboflow for real detection."
        )

    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        print(f"Could not open camera index {CAMERA_INDEX} -- try a different CAMERA_INDEX.")
        return 1

    client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
    client.connect(MQTT_BROKER, MQTT_PORT)
    client.loop_start()

    window = "Green Minifig Tracker (q to quit)"
    cv2.namedWindow(window)
    if not using_custom_model:
        cv2.createTrackbar("H low", window, int(GREEN_HSV_LOWER[0]), 179, lambda _: None)
        cv2.createTrackbar("H high", window, int(GREEN_HSV_UPPER[0]), 179, lambda _: None)
        cv2.createTrackbar("S low", window, int(GREEN_HSV_LOWER[1]), 255, lambda _: None)
        cv2.createTrackbar("S high", window, int(GREEN_HSV_UPPER[1]), 255, lambda _: None)
        cv2.createTrackbar("V low", window, int(GREEN_HSV_LOWER[2]), 255, lambda _: None)
        cv2.createTrackbar("V high", window, int(GREEN_HSV_UPPER[2]), 255, lambda _: None)

    last_publish_time = 0.0
    # Last detection's pixel center, for the preview window only -- starts
    # centered rather than at (0, 0) so the preview doesn't show a corner dot
    # before anything's ever been detected.
    preview_col, preview_row = LED_GRID_COLS // 2, LED_GRID_ROWS // 2

    print("Press 'q' or close the window to quit.")
    try:
        while True:
            if cv2.getWindowProperty(window, cv2.WND_PROP_VISIBLE) < 1:
                break  # window was closed via its titlebar, not 'q'

            ok, frame = cap.read()
            if not ok:
                print("Camera read failed -- stopping.")
                break

            results = model.predict(frame, conf=YOLO_CONFIDENCE, verbose=False)[0]
            boxes = [tuple(map(int, b)) for b in results.boxes.xyxy.tolist()]
            for box in boxes:
                cv2.rectangle(frame, box[:2], box[2:], (128, 128, 128), 1)

            if using_custom_model:
                confidences = results.boxes.conf.tolist()
                box = best_confidence_box(boxes, confidences)
                source = "custom-yolo" if box is not None else None
            else:
                hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
                lower = np.array([
                    cv2.getTrackbarPos("H low", window),
                    cv2.getTrackbarPos("S low", window),
                    cv2.getTrackbarPos("V low", window),
                ])
                upper = np.array([
                    cv2.getTrackbarPos("H high", window),
                    cv2.getTrackbarPos("S high", window),
                    cv2.getTrackbarPos("V high", window),
                ])
                mask = cv2.inRange(hsv, lower, upper)
                box, source = find_minifig_by_color(mask, boxes)
                cv2.imshow("green mask", mask)

            detected = box is not None
            if detected:
                frame_height, frame_width = frame.shape[:2]
                cx, cy = box_center(box)

                x1, y1, x2, y2 = box
                color = DETECTION_COLORS[source]
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.circle(frame, (cx, cy), 6, (255, 0, 0), -1)  # blue dot on the minifig
                cv2.putText(
                    frame, f"{source} -> ({cx}, {cy})", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2,
                )

                now = time.monotonic()
                if now - last_publish_time >= PUBLISH_INTERVAL_S:
                    last_publish_time = now
                    payload = json.dumps({"x": cx, "y": cy, "w": frame_width, "h": frame_height})
                    client.publish(MQTT_TOPIC, payload)

                preview_col = max(0, min(LED_GRID_COLS - 1, int(cx / frame_width * LED_GRID_COLS)))
                preview_row = max(0, min(LED_GRID_ROWS - 1, int(cy / frame_height * LED_GRID_ROWS)))
            else:
                cv2.putText(
                    frame, "no green minifig found", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2,
                )
                # Deliberately not publishing anything -- the UnoQ side ages
                # out a missing minifig on its own based on how long ago its
                # last message arrived, rather than us sending an explicit
                # "not detected" message. See mqtt-minifig-drive/python/main.py.

            cv2.imshow(window, frame)
            cv2.imshow("UnoQ LED preview", render_led_preview(preview_col, preview_row))

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        client.loop_stop()
        client.disconnect()
        cap.release()
        cv2.destroyAllWindows()

    return 0


if __name__ == "__main__":
    sys.exit(main())
