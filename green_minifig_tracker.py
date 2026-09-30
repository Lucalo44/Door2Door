"""Green Minifig Tracker: finds a green LEGO minifigure in the webcam feed
and publishes its position over MQTT for the UnoQ's LED display.

Detection is a two-stage pipeline:
  1. YOLO (ultralytics, pretrained on COCO) proposes candidate object boxes
     in the frame -- this is the neural net object detector.
  2. COCO has no "LEGO minifigure" class, so each YOLO box is instead scored
     by how green it is (the fraction of its pixels that fall inside an HSV
     green range). The greenest box at or above GREEN_FRACTION_THRESHOLD is
     reported as the minifig.
  If no YOLO box is green enough -- a small minifig often doesn't look like
  any COCO class at all -- this falls back to the single largest green blob
  in the whole frame instead. It's still a real detection, just not
  YOLO-confirmed; the preview window draws it in a different color (see
  DETECTION_COLORS) so you can tell which path found it.

Position is published over MQTT scaled to the UnoQ's LED_GRID_COLS x
LED_GRID_ROWS display, ready for the UnoQ to index straight into its LED
matrix. This script also draws its own simulated LED grid (a blue dot at the
scaled cell) in a preview window, so the scaling can be sanity-checked
without the UnoQ connected -- actually driving the physical LED matrix from
the published MQTT message is up to firmware running on the UnoQ, not this
script.

Setup:
  pip install -r requirements.txt
  First run downloads the YOLOv8n weights (yolov8n.pt, ~6 MB) from
  Ultralytics -- needs internet access once.

Calibrating:
  CAMERA_INDEX: try 0 first, then 1, 2... while watching the preview window
    to find which index is actually your camera.
  H/S/V low/high trackbars: watch the "green mask" preview window and adjust
    so only the minifig -- not the background -- shows up white. The
    GREEN_HSV_LOWER/GREEN_HSV_UPPER constants below are just the starting
    values loaded into those trackbars.
  GREEN_FRACTION_THRESHOLD/MIN_GREEN_AREA: placeholders -- tune against false
    positives/negatives on your own minifig and lighting.

Press 'q' or close the window to quit.
"""

import json
import sys
import time

import cv2
import numpy as np
import paho.mqtt.client as mqtt
from ultralytics import YOLO

# --- MQTT -------------------------------------------------------------------
MQTT_BROKER = "test.mosquitto.org"  # shared public broker used in class
MQTT_PORT = 1883
MQTT_TOPIC = "ME193/Door2Door"
# Payload: {"detected": bool, "x": int, "y": int} -- x/y are already scaled to
# [0, LED_GRID_COLS) x [0, LED_GRID_ROWS), ready for the UnoQ to index
# directly into its LED matrix. When detected is false, x/y repeat the last
# known position rather than jumping to (0, 0).
PUBLISH_INTERVAL_S = 0.1  # publish at most 10x/sec -- test.mosquitto.org is a
                           # shared public broker, don't flood it at full
                           # camera frame rate

# --- Camera -------------------------------------------------------------
CAMERA_INDEX = 0  # try 0 first, then 1, 2... while watching the preview
                   # window to find which index is actually your camera

# --- LED display --------------------------------------------------------
LED_GRID_COLS = 12  # UnoQ LED display width, in LEDs
LED_GRID_ROWS = 8   # UnoQ LED display height, in LEDs
LED_PREVIEW_CELL_PX = 40  # size of each simulated LED cell in the preview window

# --- YOLO object detection ------------------------------------------------
YOLO_MODEL = "yolov8n.pt"  # smallest/fastest pretrained COCO model --
                           # auto-downloaded on first run
YOLO_CONFIDENCE = 0.25     # minimum detection confidence to consider a box

# --- Green color matching -------------------------------------------------
# COCO has no "LEGO minifigure" class, so YOLO alone won't reliably find one --
# it can only propose object-shaped boxes. These boxes (and, as a fallback,
# the whole frame) are then checked for "greenness" via an HSV color mask to
# actually pick out the minifig. Starting values only -- retune live with the
# trackbars below while watching the mask preview window.
GREEN_HSV_LOWER = np.array([40, 70, 70])
GREEN_HSV_UPPER = np.array([80, 255, 255])
GREEN_FRACTION_THRESHOLD = 0.15  # a YOLO box must be at least this green
                                  # (fraction of pixels inside it) to count
MIN_GREEN_AREA = 150  # pixels -- ignores tiny green specks/noise when falling
                       # back to whole-frame contour detection

DETECTION_COLORS = {"yolo": (0, 255, 0), "color-fallback": (0, 165, 255)}


def box_green_fraction(mask: np.ndarray, box: tuple[int, int, int, int]) -> float:
    x1, y1, x2, y2 = box
    region = mask[y1:y2, x1:x2]
    if region.size == 0:
        return 0.0
    return float(np.count_nonzero(region)) / region.size


def find_minifig(mask: np.ndarray, boxes: list[tuple[int, int, int, int]]):
    """Returns ((x1, y1, x2, y2), source), or (None, None) if nothing green
    enough was found. source is "yolo" if a YOLO-proposed box was green
    enough, or "color-fallback" if the largest green blob in the whole frame
    was used instead."""
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


def scale_to_led_grid(cx: int, cy: int, frame_width: int, frame_height: int) -> tuple[int, int]:
    col = int(cx / frame_width * LED_GRID_COLS)
    row = int(cy / frame_height * LED_GRID_ROWS)
    col = max(0, min(LED_GRID_COLS - 1, col))
    row = max(0, min(LED_GRID_ROWS - 1, row))
    return col, row


def render_led_preview(col: int, row: int, detected: bool) -> np.ndarray:
    img = np.zeros((LED_GRID_ROWS * LED_PREVIEW_CELL_PX, LED_GRID_COLS * LED_PREVIEW_CELL_PX, 3), dtype=np.uint8)
    for gx in range(LED_GRID_COLS + 1):
        x = gx * LED_PREVIEW_CELL_PX
        cv2.line(img, (x, 0), (x, img.shape[0]), (40, 40, 40), 1)
    for gy in range(LED_GRID_ROWS + 1):
        y = gy * LED_PREVIEW_CELL_PX
        cv2.line(img, (0, y), (img.shape[1], y), (40, 40, 40), 1)
    center = (col * LED_PREVIEW_CELL_PX + LED_PREVIEW_CELL_PX // 2, row * LED_PREVIEW_CELL_PX + LED_PREVIEW_CELL_PX // 2)
    # Full blue when currently detected, dim blue when showing a stale
    # last-known position -- same distinction as the "detected" flag published
    # over MQTT.
    dot_color = (255, 0, 0) if detected else (90, 60, 0)
    cv2.circle(img, center, LED_PREVIEW_CELL_PX // 3, dot_color, -1)
    return img


def main():
    model = YOLO(YOLO_MODEL)

    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        print(f"Could not open camera index {CAMERA_INDEX} -- try a different CAMERA_INDEX.")
        return 1

    client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
    client.connect(MQTT_BROKER, MQTT_PORT)
    client.loop_start()

    window = "Green Minifig Tracker (q to quit)"
    cv2.namedWindow(window)
    cv2.createTrackbar("H low", window, int(GREEN_HSV_LOWER[0]), 179, lambda _: None)
    cv2.createTrackbar("H high", window, int(GREEN_HSV_UPPER[0]), 179, lambda _: None)
    cv2.createTrackbar("S low", window, int(GREEN_HSV_LOWER[1]), 255, lambda _: None)
    cv2.createTrackbar("S high", window, int(GREEN_HSV_UPPER[1]), 255, lambda _: None)
    cv2.createTrackbar("V low", window, int(GREEN_HSV_LOWER[2]), 255, lambda _: None)
    cv2.createTrackbar("V high", window, int(GREEN_HSV_UPPER[2]), 255, lambda _: None)

    # Start centered rather than at (0, 0), so a never-yet-detected minifig
    # doesn't make the UnoQ light up a corner LED by default.
    last_col, last_row = LED_GRID_COLS // 2, LED_GRID_ROWS // 2
    last_publish_time = 0.0

    print("Press 'q' or close the window to quit.")
    try:
        while True:
            if cv2.getWindowProperty(window, cv2.WND_PROP_VISIBLE) < 1:
                break  # window was closed via its titlebar, not 'q'

            ok, frame = cap.read()
            if not ok:
                print("Camera read failed -- stopping.")
                break

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

            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            mask = cv2.inRange(hsv, lower, upper)

            results = model.predict(frame, conf=YOLO_CONFIDENCE, verbose=False)[0]
            boxes = [tuple(map(int, b)) for b in results.boxes.xyxy.tolist()]
            for box in boxes:
                cv2.rectangle(frame, box[:2], box[2:], (128, 128, 128), 1)

            box, source = find_minifig(mask, boxes)
            detected = box is not None
            if detected:
                x1, y1, x2, y2 = box
                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                frame_height, frame_width = frame.shape[:2]
                last_col, last_row = scale_to_led_grid(cx, cy, frame_width, frame_height)

                cv2.rectangle(frame, (x1, y1), (x2, y2), DETECTION_COLORS[source], 2)
                cv2.circle(frame, (cx, cy), 6, (255, 0, 0), -1)  # blue dot on the minifig
                cv2.putText(
                    frame, f"{source} -> grid ({last_col}, {last_row})", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, DETECTION_COLORS[source], 2,
                )
            else:
                cv2.putText(
                    frame, "no green minifig found", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2,
                )

            now = time.monotonic()
            if now - last_publish_time >= PUBLISH_INTERVAL_S:
                last_publish_time = now
                payload = json.dumps({"detected": detected, "x": last_col, "y": last_row})
                client.publish(MQTT_TOPIC, payload)

            cv2.imshow(window, frame)
            cv2.imshow("green mask", mask)
            cv2.imshow("UnoQ LED preview", render_led_preview(last_col, last_row, detected))

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
