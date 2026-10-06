"""
detect_publish.py - watch the laptop camera with our trained YOLO model,
compute a motor speed to park the car at the camera's center, and publish
both over MQTT.

The camera is stationary (not mounted on the car): it watches the car --
which carries the green minifig as its own marker -- drive back and forth
along a straight track. This closes the loop the same way
aprilTags/apriltag_pd_tracker.py does elsewhere in this repo: the marker's
horizontal pixel position is the measurement, the frame's horizontal center
is the fixed setpoint, and PD control turns "how far off-center" into a
single forward/backward speed applied to both motors (sign-corrected for
mirror-mounting via RIGHT_MOTOR_SIGN) -- there is no left/right steering at
all, same as that script.

Single-class model (just "Minifig") -- publishes to one topic:
  ME193/Luca/green

Message (JSON):
  {"x": 412.0, "y": 230.5, "w": 640, "h": 480, "conf": 0.91,
   "left": -80, "right": 80}
  x, y = center of the minifig's box in pixels; w, h = camera frame size;
  left, right = the same PD-controlled forward/backward speed
  (-255..255), applied to both motors (right sign-corrected for
  mirror-mounting -- see RIGHT_MOTOR_SIGN).
Nothing is sent for a frame with no detection - the UnoQ stops the motors
on its own if messages stop arriving for too long (see
ArduinoApps/mqtt-minifig-drive/python/main.py's CONTROL_TIMEOUT).

Why the control math lives here instead of on the UnoQ: this way, tuning
the gains below is just editing this file and rerunning it -- no App Lab
redeploy needed. The Kp/Kd/Min Speed/Deadzone/Max Step/Smoothing trackbars
on the preview window let you retune live, without even restarting the
script.

If it still overshoots with Kd at 0 and Kp low, the cause usually is not
the gains at all -- it's noise: YOLO's detected box center jitters a few
pixels frame to frame even for a stationary target, and/or there's real
lag between a motor command and the camera actually seeing its effect
(inference time + MQTT round trip + physical motor response). The
Smoothing trackbar low-pass-filters the raw detected x position itself,
every camera frame, before it ever reaches the controller -- this damps
jitter-driven overshoot in a way that lowering Kp/Kd alone cannot, since
those gains don't distinguish real motion from noise.

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
SEND_RATE_INIT = 10    # MQTT messages per second -- also paces the control
SEND_RATE_MAX = 60     # loop, since a speed is only computed right before
                        # being sent, and is itself a trackbar (see "Send
                        # Rate Hz" below): between control updates, the
                        # motor just holds whatever speed it was last told,
                        # continuously, for the whole 1/rate gap -- too low
                        # a rate means the car can overshoot clear past the
                        # target (and the camera's entire field of view)
                        # before the next correction ever arrives. Raising
                        # it shrinks that blind window. There's a natural
                        # ceiling, though: requesting faster than the
                        # camera/YOLO pipeline can actually deliver frames
                        # just makes every frame trigger a control step --
                        # it can't go faster than that regardless of the
                        # slider.

BROKER = "broker.hivemq.com"
PORT = 1883
TOPIC = "ME193/Luca/green"
BOX_COLOR = (0, 200, 0)  # BGR -- green

WINDOW_NAME = "Door-to-door: YOLO minifig detector (q to quit)"

# --- Motor control gains -----------------------------------------------
# These are just the trackbars' starting positions -- drag the sliders on
# the preview window to retune live.
MAX_SPEED_CEILING = 255  # absolute hardware limit, matches sketch.ino's
                          # analogWrite range -- not itself a slider,
                          # just the upper bound for the Max Speed one below
MAX_SPEED_INIT = 80  # starting ceiling well below the full 255 range, so a
                      # runaway can't reach full physical speed before the
                      # UnoQ's own CONTROL_TIMEOUT safety stop (see
                      # mqtt-minifig-drive/python/main.py) has a chance to
                      # catch it. Raise this once low-speed control actually
                      # works -- capping it low doesn't fix a bad gain or
                      # sign, it just limits how far a mistake can travel
                      # before the car drives off the track/out of frame.
DIRECTION_SIGN = 1     # flip to -1 if the car drives away from center
                       # instead of toward it
RIGHT_MOTOR_SIGN = 1  # placeholder, unconfirmed for this chassis -- the two
                       # motors are likely mirror-mounted (same as
                       # whistle_soccer.py/apriltag_pd_tracker.py elsewhere
                       # in this repo), so sending the same speed to both
                       # would spin the car in place instead of driving it
                       # straight; this inverts the right side to cancel
                       # that out. If the car spins in place instead of
                       # driving straight, flip this to +1.
KP_INIT, KP_MAX = 1.2, 5.0        # speed per pixel of horizontal error
KD_INIT, KD_MAX = 0.15, 2.0       # speed per (pixel/second) of error's rate of change
MIN_SPEED_INIT = 60    # smallest PWM that reliably overcomes the motors' own
                        # static friction -- below this they just stall
                        # instead of creeping closer. Raise if it still
                        # stalls short of center; lower if it overshoots.
DEADZONE_PIXELS_INIT = 20  # horizontal error smaller than this (in pixels)
                            # counts as "centered" -> stop
MAX_SPEED_STEP_INIT = 15   # max change in commanded PWM per control step --
                            # caps how fast speed can ramp so it glides
                            # instead of jumping
SMOOTHING_INIT = 30  # out of 100 -- weight on each new raw x reading when
                      # low-pass-filtering it (see update_smoothed_cx()).
                      # Lower = heavier smoothing (less jitter, more lag);
                      # higher = less smoothing (more responsive, more
                      # jitter passed through to the controller). 100 = off.
D_SMOOTHING = 0.15  # low-pass filter weight on the derivative term, same
                     # reasoning as apriltag_pd_tracker.py's D_SMOOTHING
                     # elsewhere in this repo -- not exposed as a slider,
                     # rarely needs retuning

# The minifig's real size is fixed, so its apparent box width in pixels is a
# proxy for the car's distance from the camera -- bigger on screen means
# closer. This scales the PD output by (reference width / apparent width),
# so a car that looks small (far away) drives faster and one that looks big
# (close) drives slower, on top of the existing centering behavior. Same
# idea as apriltag_pd_tracker.py's TAG_SIZE_REF_PIXELS.
MINIFIG_WIDTH_REF_PIXELS = 120  # apparent box width, in pixels, considered
                                 # "neutral" distance (factor = 1x) -- measure
                                 # this at your track's typical distance
DISTANCE_FACTOR_MIN = 0.5       # clamp so an extreme distance can't overwhelm Kp/Kd
DISTANCE_FACTOR_MAX = 2.0
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


_smoothed_cx = None


def update_smoothed_cx(cx):
    """Low-pass filters the detected center-x to damp frame-to-frame jitter
    before it ever reaches the controller. Called every camera frame (not
    gated by SEND_RATE like control_step()/publishing are), so it has the
    full frame rate's worth of samples to average over -- smoothing against
    only the slower, throttled control-step rate would be much weaker for
    the same slider value."""
    global _smoothed_cx
    smoothing = cv2.getTrackbarPos("Smoothing x100", WINDOW_NAME) / 100.0
    if _smoothed_cx is None:
        _smoothed_cx = cx  # snap to the first-ever reading, no artificial ramp-in
    else:
        _smoothed_cx += smoothing * (cx - _smoothed_cx)
    return _smoothed_cx


_prev_error = 0.0
_have_prev_error = False
_smoothed_d_error = 0.0
_last_speed = 0.0
_prev_control_time = time.monotonic()
_last_distance_factor = 1.0  # just for the on-screen readout


def control_step(cx, box_width, frame_width):
    """Returns (left_speed, right_speed) ints: a single PD-controlled
    forward/backward speed applied to both motors (sign-corrected for
    mirror-mounting via RIGHT_MOTOR_SIGN) to park the car -- which carries
    the minifig as its own marker -- centered in this stationary camera's
    view. Same structure as apriltag_pd_tracker.py's control loop elsewhere
    in this repo, just with a YOLO-detected box instead of an AprilTag.
    Gains are read live from the trackbars on the preview window."""
    global _prev_error, _have_prev_error, _smoothed_d_error, _prev_control_time
    global _last_speed, _last_distance_factor

    kp = cv2.getTrackbarPos("Kp x100", WINDOW_NAME) / 100.0
    kd = cv2.getTrackbarPos("Kd x100", WINDOW_NAME) / 100.0
    min_speed = cv2.getTrackbarPos("Min Speed", WINDOW_NAME)
    max_speed = cv2.getTrackbarPos("Max Speed", WINDOW_NAME)
    deadzone_pixels = cv2.getTrackbarPos("Deadzone px", WINDOW_NAME)
    max_speed_step = max(1, cv2.getTrackbarPos("Max Step", WINDOW_NAME))

    now = time.monotonic()
    dt = now - _prev_control_time
    _prev_control_time = now

    frame_center_x = frame_width / 2
    error = frame_center_x - cx  # positive -> car needs to drive toward +x

    raw_d_error = 0.0
    if _have_prev_error and dt > 0:
        raw_d_error = (error - _prev_error) / dt
    _smoothed_d_error += D_SMOOTHING * (raw_d_error - _smoothed_d_error)
    _prev_error = error
    _have_prev_error = True

    distance_factor = MINIFIG_WIDTH_REF_PIXELS / max(1, box_width)
    distance_factor = max(DISTANCE_FACTOR_MIN, min(DISTANCE_FACTOR_MAX, distance_factor))
    _last_distance_factor = distance_factor

    desired_speed = 0.0
    if abs(error) > deadzone_pixels:
        raw = (kp * error + kd * _smoothed_d_error) * distance_factor
        # Faster the farther off-center it is, slower as it nears the line --
        # but floored at min_speed so it doesn't stall out before actually
        # getting there.
        magnitude = max(min_speed, min(max_speed, abs(raw)))
        desired_speed = DIRECTION_SIGN * (magnitude if raw >= 0 else -magnitude)

    # Slew-limit so the actual command glides toward desired_speed instead of
    # jumping straight there.
    step = max(-max_speed_step, min(max_speed_step, desired_speed - _last_speed))
    _last_speed += step

    left = int(round(_last_speed))
    right = int(round(RIGHT_MOTOR_SIGN * _last_speed))
    return left, right


def reset_control():
    global _have_prev_error, _smoothed_d_error, _smoothed_cx
    _have_prev_error = False
    _smoothed_d_error = 0.0
    _smoothed_cx = None  # don't drag a stale average into the next detection


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

    cv2.namedWindow(WINDOW_NAME)
    cv2.createTrackbar("Kp x100", WINDOW_NAME, int(KP_INIT * 100), int(KP_MAX * 100), lambda _: None)
    cv2.createTrackbar("Kd x100", WINDOW_NAME, int(KD_INIT * 100), int(KD_MAX * 100), lambda _: None)
    cv2.createTrackbar("Min Speed", WINDOW_NAME, MIN_SPEED_INIT, MAX_SPEED_CEILING, lambda _: None)
    cv2.createTrackbar("Max Speed", WINDOW_NAME, MAX_SPEED_INIT, MAX_SPEED_CEILING, lambda _: None)
    cv2.createTrackbar("Deadzone px", WINDOW_NAME, DEADZONE_PIXELS_INIT, 100, lambda _: None)
    cv2.createTrackbar("Max Step", WINDOW_NAME, MAX_SPEED_STEP_INIT, 50, lambda _: None)
    cv2.createTrackbar("Smoothing x100", WINDOW_NAME, SMOOTHING_INIT, 100, lambda _: None)
    cv2.createTrackbar("Send Rate Hz", WINDOW_NAME, SEND_RATE_INIT, SEND_RATE_MAX, lambda _: None)

    last_send = 0.0
    left, right = 0, 0
    while True:
        ok, frame = camera.read()
        if not ok:
            print("Camera stopped sending frames.")
            break
        h, w = frame.shape[:2]

        result = model(frame, conf=CONFIDENCE, verbose=False)[0]
        found = best_detection(result)

        # Compute control + publish together (rate-limited so we don't flood
        # the broker -- this also paces how often the control loop steps).
        now = time.time()
        if found is not None:
            conf, cx, cy, (x1, y1, x2, y2) = found
            smoothed_cx = update_smoothed_cx(cx)  # every frame, not gated by SEND_RATE
            box_width = x2 - x1

            send_rate = max(1, cv2.getTrackbarPos("Send Rate Hz", WINDOW_NAME))
            if now - last_send >= 1 / send_rate:
                last_send = now
                left, right = control_step(smoothed_cx, box_width, w)
                msg = {
                    "x": round(smoothed_cx, 1), "y": round(cy, 1), "w": w, "h": h,
                    "conf": round(conf, 2), "left": left, "right": right,
                }
                client.publish(TOPIC, json.dumps(msg))

            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), BOX_COLOR, 2)
            cv2.circle(frame, (int(cx), int(cy)), 5, BOX_COLOR, -1)  # raw detection
            cv2.circle(frame, (int(smoothed_cx), int(cy)), 5, (0, 255, 255), 2)  # smoothed (hollow)
            cv2.putText(frame, f"minifig {conf:.2f}  L{left:+d} R{right:+d}", (int(x1), int(y1) - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, BOX_COLOR, 2)
            cv2.putText(frame, f"box width: {box_width:.0f}px  distance factor: x{_last_distance_factor:.2f}",
                        (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, BOX_COLOR, 2)
        else:
            reset_control()
            cv2.putText(frame, "no minifig detected", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        cv2.line(frame, (w // 2, 0), (w // 2, h), (0, 0, 255), 1)
        cv2.imshow(WINDOW_NAME, frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    camera.release()
    cv2.destroyAllWindows()
    client.loop_stop()
    client.disconnect()


if __name__ == "__main__":
    main()
