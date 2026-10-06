"""
detect_publish.py - watch the laptop camera with our trained YOLO model and
publish where the minifig is over MQTT.

Single-class model (just "Minifig") -- publishes to one topic:
  ME193/Luca/green

Message (JSON), same format as the professor's MQTT Minifig Monitor:
  {"x": 412.0, "y": 230.5, "w": 640, "h": 480, "conf": 0.91}
  x, y = center of the minifig's box in pixels; w, h = camera frame size.
Nothing is sent for a frame with no detection - the UNO Q decides what to
do when messages stop arriving.

Run from this folder:   python detect_publish.py
Press q in the video window to quit.
"""
import json
import os
import time
from pathlib import Path

import cv2
import paho.mqtt.client as mqtt
from ultralytics import YOLO

# Run from this folder and load the model by its short name -- also makes
# "python detect_publish.py" work regardless of your current directory.
os.chdir(Path(__file__).parent)

# ---------------- settings ----------------
MODEL_FILE = "best.pt"
CAMERA = 0              # 0 = built-in camera; try 1 if it opens your iPhone instead
CONFIDENCE = 0.5        # ignore detections less sure than this
SEND_RATE = 10          # MQTT messages per second, max

BROKER = "broker.hivemq.com"
PORT = 1883
TOPIC = "ME193/Luca/green"
BOX_COLOR = (0, 200, 0)  # BGR -- green
# ------------------------------------------


def best_detection(result):
    """Keep only the single most confident box in the frame, or None."""
    best = None
    for box in result.boxes:
        conf = float(box.conf)
        if best is None or conf > best[0]:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            best = (conf, (x1 + x2) / 2, (y1 + y2) / 2, (x1, y1, x2, y2))
    return best


def main():
    model = YOLO(MODEL_FILE)
    print("Model classes:", model.names)

    camera = cv2.VideoCapture(CAMERA)
    if not camera.isOpened():
        raise SystemExit(f"Could not open camera {CAMERA}. Check camera permission for VS Code.")

    client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
    client.connect(BROKER, PORT, keepalive=30)
    client.loop_start()
    print(f"Connected to {BROKER}. Publishing to {TOPIC}")

    last_send = 0.0
    while True:
        ok, frame = camera.read()
        if not ok:
            print("Camera stopped sending frames.")
            break
        h, w = frame.shape[:2]

        result = model(frame, conf=CONFIDENCE, verbose=False)[0]
        found = best_detection(result)

        # Publish (rate-limited so we don't flood the broker)
        now = time.time()
        if found is not None and now - last_send >= 1 / SEND_RATE:
            last_send = now
            conf, cx, cy, _ = found
            msg = {"x": round(cx, 1), "y": round(cy, 1), "w": w, "h": h, "conf": round(conf, 2)}
            client.publish(TOPIC, json.dumps(msg))

        # Draw what we see: center line (the stopping point) and the detection
        cv2.line(frame, (w // 2, 0), (w // 2, h), (0, 0, 255), 1)
        if found is not None:
            conf, cx, cy, (x1, y1, x2, y2) = found
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), BOX_COLOR, 2)
            cv2.circle(frame, (int(cx), int(cy)), 5, BOX_COLOR, -1)
            cv2.putText(frame, f"minifig {conf:.2f}", (int(x1), int(y1) - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, BOX_COLOR, 2)
        else:
            cv2.putText(frame, "no minifig detected", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        cv2.imshow("Door-to-door: YOLO minifig detector (q to quit)", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    camera.release()
    cv2.destroyAllWindows()
    client.loop_stop()
    client.disconnect()


if __name__ == "__main__":
    main()
