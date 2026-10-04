import json
import threading
import time

import numpy as np
import paho.mqtt.client as mqtt

from arduino.app_utils import App, Bridge, Frame

# --- MQTT feed -------------------------------------------------------------
# Must match the broker/topic detect_publish.py (in Door2Door/YOLO/) actually
# publishes to -- that script and this one were ported from two different
# people's code and didn't originally agree on either.
MQTT_BROKER = "broker.hivemq.com"
MQTT_PORT = 1883
MQTT_TOPIC = "ME193/Luca/green"

# --- Heartbeat ---------------------------------------------------------------
# Published so anyone watching the broker (e.g. an instructor dashboard) can
# tell this board is alive, independent of whether a target is being tracked.
MQTT_HEARTBEAT_TOPIC = "ME193/heartbeat"
HEARTBEAT_INTERVAL = 60  # seconds
DEVICE_ID = "Luca"  # App Lab's name for this board - update if renamed

# --- Display ----------------------------------------------------------------
FRAME_ROWS = 8
FRAME_COLS = 13
PIXEL_BRIGHTNESS = 7  # 0-7, max brightness

# How often the LED matrix is refreshed from the latest known position.
# Decoupled from the MQTT message rate (which can be much higher).
REFRESH_INTERVAL = 0.05  # seconds

# The marker is a FRESH_MARKER_SIZE x FRESH_MARKER_SIZE block right after a
# reading, shrinking to a single STALE_MARKER_SIZE x STALE_MARKER_SIZE pixel
# once older than STALE_TIMEOUT - but it never fully disappears once a first
# reading has arrived, so you can always see the last known position.
FRESH_MARKER_SIZE = 3
STALE_MARKER_SIZE = 1
STALE_TIMEOUT = 0.75  # seconds

_state_lock = threading.Lock()
_last_col = None
_last_row = None
_last_seen = 0.0


def on_connect(client, userdata, flags, rc):
    print(f"[mqtt] connected (rc={rc}), subscribing to {MQTT_TOPIC!r}")
    client.subscribe(MQTT_TOPIC)
    send_heartbeat(force=True)  # confirm liveness as soon as we're connected


def on_disconnect(client, userdata, rc):
    print(f"[mqtt] disconnected (rc={rc})")


def on_message(client, userdata, msg):
    # Keep this handler fast: just parse, print, and stash the latest
    # position. Messages can arrive at a high rate from a laptop-side YOLO
    # detector.
    global _last_col, _last_row, _last_seen
    try:
        text = msg.payload.decode("utf-8", errors="replace")
    except Exception:
        return
    print(f"[mqtt] {msg.topic}: {text}")

    try:
        data = json.loads(text)
        x, y, w, h = data["x"], data["y"], data["w"], data["h"]
    except (TypeError, ValueError, KeyError, json.JSONDecodeError):
        return
    if not w or not h:
        return

    col = int(x / w * FRAME_COLS)
    row = int(y / h * FRAME_ROWS)
    col = max(0, min(FRAME_COLS - 1, col))
    row = max(0, min(FRAME_ROWS - 1, row))

    with _state_lock:
        _last_col, _last_row = col, row
        _last_seen = time.monotonic()


def build_frame():
    array = np.zeros((FRAME_ROWS, FRAME_COLS), dtype=np.uint8)
    with _state_lock:
        col, row, last_seen = _last_col, _last_row, _last_seen

    if col is None:
        return array  # blank: no reading has ever arrived yet

    fresh = (time.monotonic() - last_seen) <= STALE_TIMEOUT
    size = FRESH_MARKER_SIZE if fresh else STALE_MARKER_SIZE
    half = size // 2

    for dr in range(-half, half + 1):
        for dc in range(-half, half + 1):
            r, c = row + dr, col + dc
            if 0 <= r < FRAME_ROWS and 0 <= c < FRAME_COLS:
                array[r, c] = PIXEL_BRIGHTNESS
    return array


client = mqtt.Client()
client.on_connect = on_connect
client.on_disconnect = on_disconnect
client.on_message = on_message
client.reconnect_delay_set(min_delay=1, max_delay=30)
client.connect(MQTT_BROKER, MQTT_PORT, keepalive=60)
client.loop_start()

# The connect-time heartbeat is sent from on_connect (force=True); this just
# seeds the periodic check in loop() so it doesn't also fire on iteration 1.
_last_heartbeat = time.monotonic()


def send_heartbeat(force=False):
    global _last_heartbeat
    now = time.monotonic()
    if not force and now - _last_heartbeat < HEARTBEAT_INTERVAL:
        return
    _last_heartbeat = now
    payload = json.dumps({"device": DEVICE_ID, "ts": time.time()})
    result = client.publish(MQTT_HEARTBEAT_TOPIC, payload)
    status = "ok" if result.rc == mqtt.MQTT_ERR_SUCCESS else f"failed (rc={result.rc})"
    print(f"[mqtt] heartbeat -> {MQTT_HEARTBEAT_TOPIC} [{status}]: {payload}")


def loop():
    send_heartbeat()
    frame = Frame(build_frame())
    Bridge.call("draw", frame.to_board_bytes())
    time.sleep(REFRESH_INTERVAL)


App.run(user_loop=loop)
