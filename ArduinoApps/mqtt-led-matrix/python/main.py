import threading
import time

import paho.mqtt.client as mqtt

from arduino.app_utils import App, Bridge

MQTT_BROKER = "test.mosquitto.org"
MQTT_PORT = 1883
MQTT_TOPIC = "ME193/minifig"

MATRIX_ROWS = 8
MATRIX_COLS = 13
SCROLL_DELAY_SECONDS = 0.08

# 3x5 pixel font: each glyph is 5 rows of 3 columns. Unsupported characters
# fall back to a blank space.
FONT = {
    "0": ["###", "#.#", "#.#", "#.#", "###"],
    "1": [".#.", "##.", ".#.", ".#.", "###"],
    "2": ["###", "..#", "###", "#..", "###"],
    "3": ["###", "..#", "###", "..#", "###"],
    "4": ["#.#", "#.#", "###", "..#", "..#"],
    "5": ["###", "#..", "###", "..#", "###"],
    "6": ["###", "#..", "###", "#.#", "###"],
    "7": ["###", "..#", "..#", "..#", "..#"],
    "8": ["###", "#.#", "###", "#.#", "###"],
    "9": ["###", "#.#", "###", "..#", "###"],
    "A": [".#.", "#.#", "###", "#.#", "#.#"],
    "B": ["##.", "#.#", "##.", "#.#", "##."],
    "C": [".##", "#..", "#..", "#..", ".##"],
    "D": ["##.", "#.#", "#.#", "#.#", "##."],
    "E": ["###", "#..", "##.", "#..", "###"],
    "F": ["###", "#..", "##.", "#..", "#.."],
    "G": [".##", "#..", "#.#", "#.#", ".##"],
    "H": ["#.#", "#.#", "###", "#.#", "#.#"],
    "I": ["###", ".#.", ".#.", ".#.", "###"],
    "J": ["..#", "..#", "..#", "#.#", ".#."],
    "K": ["#.#", "#.#", "##.", "#.#", "#.#"],
    "L": ["#..", "#..", "#..", "#..", "###"],
    "M": ["#.#", "###", "#.#", "#.#", "#.#"],
    "N": ["#.#", "##.", "#.#", ".##", "#.#"],
    "O": [".#.", "#.#", "#.#", "#.#", ".#."],
    "P": ["##.", "#.#", "##.", "#..", "#.."],
    "Q": [".#.", "#.#", "#.#", ".#.", "..#"],
    "R": ["##.", "#.#", "##.", "#.#", "#.#"],
    "S": [".##", "#..", ".#.", "..#", "##."],
    "T": ["###", ".#.", ".#.", ".#.", ".#."],
    "U": ["#.#", "#.#", "#.#", "#.#", ".#."],
    "V": ["#.#", "#.#", "#.#", "#.#", ".#."],
    "W": ["#.#", "#.#", "#.#", "###", "#.#"],
    "X": ["#.#", "#.#", ".#.", "#.#", "#.#"],
    "Y": ["#.#", "#.#", ".#.", ".#.", ".#."],
    "Z": ["###", "..#", ".#.", "#..", "###"],
    ".": ["...", "...", "...", "...", ".#."],
    ",": ["...", "...", "...", ".#.", "#.."],
    "!": [".#.", ".#.", ".#.", "...", ".#."],
    "?": ["##.", "..#", ".#.", "...", ".#."],
    ":": ["...", ".#.", "...", ".#.", "..."],
    "'": [".#.", ".#.", "...", "...", "..."],
    "/": ["..#", ".#.", ".#.", "#..", "#.."],
    "-": ["...", "...", "###", "...", "..."],
    "_": ["...", "...", "...", "...", "###"],
    " ": ["...", "...", "...", "...", "..."],
}
CHAR_ROWS = 5
CHAR_COLS = 3
CHAR_GAP = 1
ROW_OFFSET = (MATRIX_ROWS - CHAR_ROWS) // 2


def render_text(text):
    """Returns a list of columns (left to right); each column is a list of
    MATRIX_ROWS pixel values (0 or 1), top to bottom."""
    columns = []
    for ch in text.upper():
        glyph = FONT.get(ch, FONT[" "])
        for c in range(CHAR_COLS):
            col = [0] * MATRIX_ROWS
            for r in range(CHAR_ROWS):
                if glyph[r][c] == "#":
                    col[ROW_OFFSET + r] = 1
            columns.append(col)
        for _ in range(CHAR_GAP):
            columns.append([0] * MATRIX_ROWS)
    return columns


def frame_bits(window_columns):
    """window_columns: MATRIX_COLS columns, each MATRIX_ROWS pixels tall.
    Returns a row-major '0'/'1' string of length MATRIX_ROWS * MATRIX_COLS."""
    bits = [0] * (MATRIX_ROWS * MATRIX_COLS)
    for c, col in enumerate(window_columns):
        for r, v in enumerate(col):
            bits[r * MATRIX_COLS + c] = v
    return "".join(str(b) for b in bits)


_lock = threading.Lock()
_incoming_message = {"text": None}


def on_connect(client, userdata, flags, reason_code, properties=None):
    print(f"MQTT connected to {MQTT_BROKER}:{MQTT_PORT} ({reason_code})")
    client.subscribe(MQTT_TOPIC)


def on_message(client, userdata, msg):
    text = msg.payload.decode("utf-8", errors="replace").strip()
    print(f"MQTT message on {msg.topic}: {text!r}")
    if text:
        with _lock:
            _incoming_message["text"] = text


mqtt_client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
mqtt_client.on_connect = on_connect
mqtt_client.on_message = on_message
mqtt_client.reconnect_delay_set(min_delay=1, max_delay=30)
mqtt_client.connect_async(MQTT_BROKER, MQTT_PORT, keepalive=60)
mqtt_client.loop_start()

state = {
    "message": f"WAITING FOR {MQTT_TOPIC}...     ",
    "columns": None,
    "total": 0,
    "pos": 0,
}


def loop():
    with _lock:
        incoming = _incoming_message["text"]
        _incoming_message["text"] = None

    if incoming is not None:
        state["message"] = incoming + "     "
        state["columns"] = None

    if state["columns"] is None or state["pos"] >= state["total"]:
        columns = render_text(state["message"])
        columns += [[0] * MATRIX_ROWS for _ in range(MATRIX_COLS)]
        state["columns"] = columns
        state["total"] = len(columns)
        state["pos"] = 0

    columns = state["columns"]
    total = state["total"]
    pos = state["pos"]
    window = [columns[(pos + i) % total] for i in range(MATRIX_COLS)]
    Bridge.notify("draw_frame", frame_bits(window))

    state["pos"] += 1
    time.sleep(SCROLL_DELAY_SECONDS)


App.run(user_loop=loop)
