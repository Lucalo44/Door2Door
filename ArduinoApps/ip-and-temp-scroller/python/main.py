import json
import os
import socket
import time
import urllib.request

from arduino.app_utils import App, Bridge

LATITUDE = 42.36
LONGITUDE = -71.06
WEATHER_URL = (
    "https://api.open-meteo.com/v1/forecast"
    f"?latitude={LATITUDE}&longitude={LONGITUDE}"
    "&current=temperature_2m&temperature_unit=fahrenheit"
)

MATRIX_ROWS = 8
MATRIX_COLS = 13
SCROLL_DELAY_SECONDS = 0.08
TEMP_REFRESH_SECONDS = 60

# 3x5 pixel font: each glyph is 5 rows of 3 columns.
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
    ".": ["...", "...", "...", "...", ".#."],
    "-": ["...", "...", "###", "...", "..."],
    " ": ["...", "...", "...", "...", "..."],
    "I": ["###", ".#.", ".#.", ".#.", "###"],
    "P": ["###", "#.#", "###", "#..", "#.."],
    "F": ["###", "#..", "###", "#..", "#.."],
}
CHAR_ROWS = 5
CHAR_COLS = 3
CHAR_GAP = 1
ROW_OFFSET = (MATRIX_ROWS - CHAR_ROWS) // 2


def get_ip():
    # The app container runs on its own Docker network, so its own socket's
    # address isn't the board's LAN IP; App Lab exposes the real one here.
    host_ip = os.environ.get("HOST_IP")
    if host_ip:
        return host_ip

    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("8.8.8.8", 80))
        return s.getsockname()[0]
    except OSError:
        return "0.0.0.0"
    finally:
        s.close()


def get_temperature_f():
    with urllib.request.urlopen(WEATHER_URL, timeout=10) as resp:
        data = json.loads(resp.read())
    return data["current"]["temperature_2m"]


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


state = {
    "columns": None,
    "total": 0,
    "pos": 0,
    "cached_temp_f": None,
    "last_temp_fetch": 0.0,
}


def build_message():
    ip = get_ip()

    now = time.time()
    if state["cached_temp_f"] is None or now - state["last_temp_fetch"] > TEMP_REFRESH_SECONDS:
        try:
            state["cached_temp_f"] = get_temperature_f()
            state["last_temp_fetch"] = now
        except Exception as e:
            print(f"Open-Meteo fetch failed: {e}")

    if state["cached_temp_f"] is None:
        temp_str = "--F"
    else:
        temp_str = f"{round(state['cached_temp_f'])}F"

    return f"IP {ip}     {temp_str}     "


def loop():
    if state["columns"] is None or state["pos"] >= state["total"]:
        columns = render_text(build_message())
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
