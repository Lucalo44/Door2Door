"""Runs on the UNO Q's Linux/MPU side as the Python half of this Arduino App
Lab project. Subscribes to the minifig position over MQTT (published by
green_minifig_tracker.py), shows it on the LED matrix as a dot, and
PD-controls two Cytron Maker Drive motors to keep it horizontally centered.
sketch/sketch.ino is a thin executor: all of this logic lives here.

Topic and payload convention, and the LED frame/staleness handling, are
copied from mqtt-minifig-monitor (ported from
https://github.com/mohdalmheiri/ME193-Robotics) rather than this project's
own earlier format: {"x": <pixel x>, "y": <pixel y>, "w": <frame width>,
"h": <frame height>} -- x/y are the detection center in pixel coordinates,
w/h are the camera frame's own dimensions (used only to normalize x/y into
the LED grid), not the detection box's size. One consequence: there's no
box-size field to use as a distance proxy, so unlike this project's earlier
control loop, motor speed isn't scaled by how close the minifig looks --
only the horizontal centering error.

mqtt-minifig-monitor never goes stale on the LED: once a first reading has
arrived, the marker shrinks (see FRESH_MARKER_SIZE/STALE_MARKER_SIZE/
STALE_TIMEOUT) but never disappears, so you can always see the last known
position. The motors behave differently on purpose -- driving on a
stale reading is a safety issue a display isn't, so CONTROL_TIMEOUT stops
them separately (see control_step()/stop_control()).
"""

import json
import threading
import time

import numpy as np
import paho.mqtt.client as mqtt

from arduino.app_utils import App, Bridge, Frame

# --- MQTT feed -------------------------------------------------------------
MQTT_BROKER = "test.mosquitto.org"
MQTT_PORT = 1883
MQTT_TOPIC = "ME193/minifig"  # must match MQTT_TOPIC in green_minifig_tracker.py

# --- Display (copied from mqtt-minifig-monitor) -----------------------------
FRAME_ROWS = 8
FRAME_COLS = 13
PIXEL_BRIGHTNESS = 7  # 0-7, max brightness
REFRESH_INTERVAL = 0.05  # seconds -- LED redraw / control loop rate

# The marker is a FRESH_MARKER_SIZE x FRESH_MARKER_SIZE block right after a
# reading, shrinking to a single STALE_MARKER_SIZE x STALE_MARKER_SIZE pixel
# once older than STALE_TIMEOUT -- but it never fully disappears once a first
# reading has arrived, so you can always see the last known position.
FRESH_MARKER_SIZE = 3
STALE_MARKER_SIZE = 1
STALE_TIMEOUT = 0.75  # seconds

# --- Motor control -----------------------------------------------------------
CONTROL_TIMEOUT = 0.75  # seconds -- stop the motors if no fresher reading
                         # than this arrives. Separate from STALE_TIMEOUT
                         # above on principle (driving on stale data is a
                         # safety issue a display isn't), even though they
                         # currently share the same value.
MAX_SPEED = 255          # analogWrite ceiling, must match MAX_SPEED in sketch.ino
MIN_SPEED = 60           # smallest PWM that reliably overcomes the motors'
                          # own static friction -- below this they just
                          # stall instead of creeping closer. Raise if it
                          # still stalls short of center; lower if it
                          # overshoots.
DEADZONE_PIXELS = 20      # horizontal error smaller than this (in pixels)
                          # counts as "centered" -> stop
MAX_SPEED_STEP = 15       # max change in commanded PWM per loop -- caps how
                          # fast speed can ramp so it glides instead of
                          # jumping
KP = 1.2    # PWM per pixel of horizontal error -- placeholder, retune live
KD = 0.15   # PWM per (pixel/second) of error's rate of change -- placeholder
D_SMOOTHING = 0.15  # low-pass filter weight on the derivative term, same
                     # reasoning as apriltag_seek_tracker.py's D_SMOOTHING
                     # elsewhere in this repo

_state_lock = threading.Lock()
_last_x = None
_last_y = None
_last_w = None
_last_h = None
_last_seen = 0.0


def on_connect(client, userdata, flags, rc):
    print(f"[mqtt] connected (rc={rc}), subscribing to {MQTT_TOPIC!r}")
    client.subscribe(MQTT_TOPIC)


def on_disconnect(client, userdata, rc):
    print(f"[mqtt] disconnected (rc={rc})")


def on_message(client, userdata, msg):
    # Keep this handler fast: just parse and stash the latest position.
    # Messages can arrive at a high rate from a laptop-side YOLO detector.
    global _last_x, _last_y, _last_w, _last_h, _last_seen
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

    with _state_lock:
        _last_x, _last_y, _last_w, _last_h = x, y, w, h
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


_prev_error = 0.0
_have_prev_error = False
_smoothed_d_error = 0.0
_last_left_speed = 0
_last_right_speed = 0


def _slew(desired_left, desired_right):
    global _last_left_speed, _last_right_speed
    left_step = max(-MAX_SPEED_STEP, min(MAX_SPEED_STEP, desired_left - _last_left_speed))
    right_step = max(-MAX_SPEED_STEP, min(MAX_SPEED_STEP, desired_right - _last_right_speed))
    _last_left_speed += left_step
    _last_right_speed += right_step
    return int(_last_left_speed), int(_last_right_speed)


def control_step(x, w, dt):
    """Returns (left_speed, right_speed): a PD controller centering x within
    a frame of width w, same structure as apriltag_seek_tracker.py's control
    loop elsewhere in this repo (minus the distance-factor term, which needs
    the detection box's own size -- not available in this payload)."""
    global _prev_error, _have_prev_error, _smoothed_d_error

    frame_center_x = w / 2
    error = frame_center_x - x  # positive -> target is left of center

    raw_d_error = 0.0
    if _have_prev_error and dt > 0:
        raw_d_error = (error - _prev_error) / dt
    _smoothed_d_error += D_SMOOTHING * (raw_d_error - _smoothed_d_error)
    _prev_error = error
    _have_prev_error = True

    desired = 0.0
    if abs(error) > DEADZONE_PIXELS:
        raw = KP * error + KD * _smoothed_d_error
        magnitude = max(MIN_SPEED, min(MAX_SPEED, abs(raw)))
        desired = magnitude if raw >= 0 else -magnitude

    # Turn toward the target (rotate in place): left/right motors get
    # opposite signs. Swap these if it turns the wrong way.
    return _slew(-desired, desired)


def stop_control():
    global _have_prev_error, _smoothed_d_error
    _have_prev_error = False
    _smoothed_d_error = 0.0
    return _slew(0, 0)  # still slew-limited, so it eases to a stop


_prev_time = time.monotonic()


def loop():
    global _prev_time
    now = time.monotonic()
    dt = now - _prev_time
    _prev_time = now

    with _state_lock:
        x, w, last_seen = _last_x, _last_w, _last_seen

    have_target = x is not None and (now - last_seen) <= CONTROL_TIMEOUT
    if have_target:
        left, right = control_step(x, w, dt)
    else:
        left, right = stop_control()

    Bridge.call("drive", left, right)
    Bridge.call("draw", Frame(build_frame()).to_board_bytes())

    time.sleep(REFRESH_INTERVAL)


App.run(user_loop=loop)
