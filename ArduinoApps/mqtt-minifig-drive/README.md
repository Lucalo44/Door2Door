# 🚗 MQTT Minifig Drive

Shows a tracked minifig's position on the LED matrix (same approach as
mqtt-minifig-monitor) and drives two motors through an M1A/M1B + M2A/M2B
dual-PWM-per-motor driver.

All control logic (PD centering, gain tuning) runs on the laptop in
`Door2Door/YOLO/detect_publish.py`, not here -- this app just forwards the
`left`/`right` speeds it's sent straight to the motors, and hard-stops them
if messages stop arriving for too long. See that script's docstring for why.
