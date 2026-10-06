"""Runs on the UNO Q's Linux/MPU side as the Python half of this Arduino App
Lab project. Subscribes to MQTT (published by Door2Door/YOLO/detect_publish.py)
and acts on it: shows the minifig's position on the LED matrix as a dot, and
forwards already-computed motor speeds to the sketch. sketch/sketch.ino is a
thin executor -- it has no control logic of its own.

The PD control math (centering the minifig, tuning gains) used to live here,
but now runs on the laptop in detect_publish.py instead -- that way, tuning
is just editing a constant (or dragging a trackbar) and rerunning a Python
script, not redeploying to this board through App Lab. See that file's
docstring for the reasoning and the message format:
  {"x", "y", "w", "h", "conf", "left", "right"}
x/y/w/h are used for the LED display (same as mqtt-minifig-monitor); left/
right are the signed motor speeds (-255..255) to forward as-is.

This board still owns one piece of control logic: if no message arrives for
CONTROL_TIMEOUT seconds (laptop crashed, MQTT dropped, camera froze), the
motors are hard-stopped here regardless of what the last message said --
driving on stale data is a safety issue a display isn't.

mqtt-minifig-monitor never goes stale on the LED: once a first reading has
arrived, the marker shrinks (see FRESH_MARKER_SIZE/STALE_MARKER_SIZE/
STALE_TIMEOUT) but never disappears, so you can always see the last known
position.
"""

import json
import threading
import time

import numpy as np
import paho.mqtt.client as mqtt

from arduino.app_utils import App, Bridge, Frame

# --- MQTT feed -------------------------------------------------------------
# Must match the broker/topic detect_publish.py (in Door2Door/YOLO/) actually
# publishes to.
MQTT_BROKER = "broker.hivemq.com"
MQTT_PORT = 1883
MQTT_TOPIC = "ME193/Luca/green"

# --- Display (copied from mqtt-minifig-monitor) -----------------------------
FRAME_ROWS = 8
FRAME_COLS = 13
PIXEL_BRIGHTNESS = 7  # 0-7, max brightness
REFRESH_INTERVAL = 0.05  # seconds -- LED redraw / drive-forwarding rate

# The marker is a FRESH_MARKER_SIZE x FRESH_MARKER_SIZE block right after a
# reading, shrinking to a single STALE_MARKER_SIZE x STALE_MARKER_SIZE pixel
# once older than STALE_TIMEOUT -- but it never fully disappears once a first
# reading has arrived, so you can always see the last known position.
FRESH_MARKER_SIZE = 3
STALE_MARKER_SIZE = 1
STALE_TIMEOUT = 0.75  # seconds

# --- Motor safety -----------------------------------------------------------
CONTROL_TIMEOUT = 0.75  # seconds -- hard-stop the motors if no fresher
                         # message than this arrives. Separate from
                         # STALE_TIMEOUT above on principle (driving on stale
                         # data is a safety issue a display isn't), even
                         # though they currently share the same value.

_state_lock = threading.Lock()
_last_x = None
_last_y = None
_last_w = None
_last_h = None
_last_left = 0
_last_right = 0
_last_seen = 0.0


def on_connect(client, userdata, flags, rc):
    print(f"[mqtt] connected (rc={rc}), subscribing to {MQTT_TOPIC!r}")
    client.subscribe(MQTT_TOPIC)


def on_disconnect(client, userdata, rc):
    print(f"[mqtt] disconnected (rc={rc})")


def on_message(client, userdata, msg):
    # Keep this handler fast: just parse and stash the latest message.
    # Messages can arrive at a high rate from a laptop-side YOLO detector.
    global _last_x, _last_y, _last_w, _last_h, _last_left, _last_right, _last_seen
    try:
        text = msg.payload.decode("utf-8", errors="replace")
    except Exception:
        return
    print(f"[mqtt] {msg.topic}: {text}")

    try:
        data = json.loads(text)
        x, y, w, h = data["x"], data["y"], data["w"], data["h"]
        left, right = int(data["left"]), int(data["right"])
    except (TypeError, ValueError, KeyError, json.JSONDecodeError):
        return
    if not w or not h:
        return

    with _state_lock:
        _last_x, _last_y, _last_w, _last_h = x, y, w, h
        _last_left, _last_right = left, right
        _last_seen = time.monotonic()


client = mqtt.Client()
client.on_connect = on_connect
client.on_disconnect = on_disconnect
client.on_message = on_message
client.reconnect_delay_set(min_delay=1, max_delay=30)
client.connect(MQTT_BROKER, MQTT_PORT, keepalive=60)
client.loop_start()


def build_frame():
    """Identical to mqtt-minifig-monitor's build_frame()."""
    array = np.zeros((FRAME_ROWS, FRAME_COLS), dtype=np.uint8)
    with _state_lock:
        x, y, w, h, last_seen = _last_x, _last_y, _last_w, _last_h, _last_seen

    if x is None:
        return array  # blank: no reading has ever arrived yet

    col = max(0, min(FRAME_COLS - 1, int(x / w * FRAME_COLS)))
    row = max(0, min(FRAME_ROWS - 1, int(y / h * FRAME_ROWS)))

    fresh = (time.monotonic() - last_seen) <= STALE_TIMEOUT
    size = FRESH_MARKER_SIZE if fresh else STALE_MARKER_SIZE
    half = size // 2

    for dr in range(-half, half + 1):
        for dc in range(-half, half + 1):
            r, c = row + dr, col + dc
            if 0 <= r < FRAME_ROWS and 0 <= c < FRAME_COLS:
                array[r, c] = PIXEL_BRIGHTNESS
    return array


def loop():
    with _state_lock:
        left, right, last_seen = _last_left, _last_right, _last_seen

    have_fresh_command = (time.monotonic() - last_seen) <= CONTROL_TIMEOUT
    if have_fresh_command:
        Bridge.call("drive", left, right)
    else:
        # Hard stop -- the laptop already slew-limits normal speed changes,
        # so there's no "ease down" case to handle here, just "is there a
        # fresh command or not."
        Bridge.call("drive", 0, 0)

    Bridge.call("draw", Frame(build_frame()).to_board_bytes())
    time.sleep(REFRESH_INTERVAL)


App.run(user_loop=loop)
