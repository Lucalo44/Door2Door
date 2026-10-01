"""Runs on the UNO Q's Linux/MPU side (the QRB2210) as the Python half of
this Arduino App Lab project. The motor-control sketch lives in
sketch/sketch.ino and runs on the STM32 MCU side.

Why split across two processors: per the UNO Q's own pinout doc, WiFi/BT is
wired to the QRB2210 MPU, not the STM32 MCU -- there's no radio on the MCU
side at all. MQTT needs a network connection, so this Python process (MPU
side) is the only place it can live; it then hands each parsed detection to
the MCU sketch, which does the actual real-time motor PWM and (eventually)
LED matrix output.

How the two sides talk -- READ THIS:
  Arduino App Lab reportedly ships an official Python<->sketch bridge for
  UNO Q projects, but I don't have a confirmed, current reference for its
  exact API, and I'd rather not invent import/class names that might not
  exist and fail silently. So this uses a plain serial connection instead
  (MCU_SERIAL_PORT below) -- it's a mechanism that's certain to work on any
  Arduino-compatible MCU, at the cost of being lower-level than whatever
  official bridge App Lab provides.
  If your App Lab project template already generated a working bridge
  example (e.g. something imported from an `arduino` package), prefer that:
  replace the body of send_to_sketch() below with the bridge call instead of
  the serial write, and you can delete the serial/pyserial parts entirely.

Setup:
  pip install -r requirements.txt
  MCU_SERIAL_PORT: the serial device this Python process uses to reach the
    MCU sketch. Placeholder -- confirm the actual device node for your UNO Q
    (e.g. check `ls /dev/tty*` before and after the board is recognized).
"""

import json
import time

import paho.mqtt.client as mqtt
import serial

# --- MQTT -------------------------------------------------------------------
# Must match MQTT_BROKER/MQTT_TOPIC in green_minifig_tracker.py exactly --
# this is the other end of that connection.
MQTT_BROKER = "test.mosquitto.org"
MQTT_PORT = 1883
MQTT_TOPIC = "ME193/Door2Door"

# --- MCU link ---------------------------------------------------------------
MCU_SERIAL_PORT = "/dev/ttyACM0"  # placeholder -- confirm the actual device
                                   # node for the UNO Q's MCU side
MCU_BAUD_RATE = 115200             # must match Serial.begin(...) in sketch.ino
RECONNECT_DELAY_S = 2.0            # pause between reconnect attempts on
                                    # either the MQTT or serial side


def open_serial():
    while True:
        try:
            return serial.Serial(MCU_SERIAL_PORT, MCU_BAUD_RATE, timeout=1)
        except serial.SerialException as exc:
            print(f"Could not open {MCU_SERIAL_PORT} ({exc}); retrying...")
            time.sleep(RECONNECT_DELAY_S)


def send_to_sketch(mcu, detected: bool, x: float, y: float, width: float, height: float):
    """One line per detection: "detected,x,y,width,height\\n" -- matches the
    parsing in sketch.ino's readLine(). All of x/y/width/height are already
    normalized to [0, 1] (fraction of the camera frame) by
    green_minifig_tracker.py."""
    line = f"{int(detected)},{x:.4f},{y:.4f},{width:.4f},{height:.4f}\n"
    mcu.write(line.encode("ascii"))


def main():
    mcu = open_serial()
    print(f"Connected to MCU sketch on {MCU_SERIAL_PORT}")

    def on_connect(client, userdata, flags, reason_code, properties):
        print(f"Connected to {MQTT_BROKER}, subscribing to {MQTT_TOPIC!r}")
        client.subscribe(MQTT_TOPIC)

    def on_message(client, userdata, msg):
        nonlocal mcu
        try:
            data = json.loads(msg.payload.decode())
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            print(f"Bad MQTT payload ({exc}): {msg.payload!r}")
            return

        try:
            send_to_sketch(
                mcu,
                bool(data["detected"]),
                float(data["x"]),
                float(data["y"]),
                float(data["width"]),
                float(data["height"]),
            )
        except KeyError as exc:
            print(f"MQTT payload missing field {exc}: {data!r}")
        except serial.SerialException as exc:
            print(f"Lost connection to MCU sketch ({exc}); reconnecting...")
            mcu.close()
            mcu = open_serial()

    client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
    client.on_connect = on_connect
    client.on_message = on_message

    while True:
        try:
            client.connect(MQTT_BROKER, MQTT_PORT)
            client.loop_forever()
        except OSError as exc:
            print(f"MQTT connection failed ({exc}); retrying...")
            time.sleep(RECONNECT_DELAY_S)


if __name__ == "__main__":
    main()
