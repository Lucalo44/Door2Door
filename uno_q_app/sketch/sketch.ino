/*
  Door2Door motor control sketch for the UNO Q (MCU/STM32 side of this
  Arduino App Lab project).

  Receives a line per detection over Serial from main.py (the Linux/MPU
  side, which holds the actual MQTT connection -- see that file's docstring
  for why the split exists): "detected,x,y,width,height\n", where x/y are
  the minifig's center and width/height its size, each normalized to [0, 1]
  as a fraction of the camera frame.

  Drives two motors via a Cytron Maker Drive, which takes one PWM pin
  (speed magnitude) plus one DIR pin (direction, HIGH/LOW) per motor. Centers
  the minifig horizontally with a PD controller, same structure as
  apriltag_seek_tracker.py's control loop
  elsewhere in this repo: proportional + derivative on the horizontal error,
  slew-limited output, a speed floor so it doesn't stall right at the
  deadzone edge, and a distance factor from the box's width (bigger box =
  closer = drive slower) -- see that file for the reasoning behind each of
  these if it's unfamiliar. This only turns the car to keep the minifig
  centered -- there's no forward/backward "approach" component, since that
  wasn't specified.

  Safety: if no line arrives within LINK_TIMEOUT_MS (main.py crashed, MQTT
  dropped, the camera froze upstream), the motors stop rather than keep
  running on a stale command.

  LED matrix: NOT IMPLEMENTED. The UNO Q's pinout doc only documents the
  8x13 matrix's electrical pin numbering (1-104), not a software library for
  addressing it -- I don't have a confirmed API to drive it with, so
  setMatrixPixel() below is a stub. ledIndexFor() does the position math
  (which of the 104 LEDs corresponds to a given normalized x/y) so only the
  actual pixel-write call needs filling in once you've confirmed that API
  (check Arduino's UNO Q documentation/examples for a matrix library,
  possibly similar to the UNO R4 WiFi's Arduino_LED_Matrix).

  Calibrating:
    Motor pins: placeholders below -- confirm against your actual Maker
      Drive wiring. The two PWM pins must be PWM-capable per the UNO Q
      pinout; the two DIR pins can be any plain digital pin.
    If a motor spins the wrong way, swap HIGH/LOW in driveMotor() below (or
      swap that motor's two wires at the driver).
    KP/KD, DEADZONE, MIN_SPEED, MAX_SPEED_STEP: same tuning approach as
      apriltag_seek_tracker.py -- start with KP alone, add KD to damp
      oscillation, raise DEADZONE or lower MIN_SPEED if it jitters at center.
    WIDTH_REF: the minifig's normalized box width considered "neutral"
      distance (factor = 1x) -- measure this at your track's typical
      distance between the camera and the minifig.
    If it turns away from the target instead of toward it, swap the signs
    on desiredLeft/desiredRight below.
*/

#include <Arduino.h>

// --- Motor driver pins (Cytron Maker Drive -- PWM + DIR per motor) --------
// placeholders -- confirm against your actual Maker Drive wiring
const int LEFT_MOTOR_PWM_PIN = 5;   // ~D5
const int LEFT_MOTOR_DIR_PIN = 4;   // D4
const int RIGHT_MOTOR_PWM_PIN = 9;  // ~D9
const int RIGHT_MOTOR_DIR_PIN = 7;  // D7
// ~D3, ~D6, ~D10, ~D11 are spare PWM-capable pins, unused here.

// --- Serial link to main.py (the MPU/Linux side) ---------------------------
const long SERIAL_BAUD_RATE = 115200;        // must match MCU_BAUD_RATE in main.py
const unsigned long LINK_TIMEOUT_MS = 500;   // stop the motors if no new
                                              // detection line arrives within
                                              // this long

// --- Control ----------------------------------------------------------------
const int MAX_SPEED = 255;      // analogWrite ceiling
const int MIN_SPEED = 60;       // smallest PWM that reliably overcomes the
                                 // motors' own static friction -- below this
                                 // they just stall instead of creeping
                                 // closer. Raise if it still stalls short of
                                 // center; lower if it overshoots.
const float DEADZONE = 0.03;         // horizontal error smaller than this (as
                                      // a fraction of frame width) counts as
                                      // "centered" -> stop
const int MAX_SPEED_STEP = 15;       // max change in commanded PWM per loop --
                                      // caps how fast speed can ramp so it
                                      // glides instead of jumping
const float KP = 400.0;    // PWM per unit of normalized horizontal error
const float KD = 60.0;     // PWM per unit of error's rate of change (1/s)
const float D_SMOOTHING = 0.15;   // low-pass filter weight on the derivative
                                   // term, same reasoning as
                                   // apriltag_seek_tracker.py's D_SMOOTHING

const float WIDTH_REF = 0.15;          // normalized box width considered
                                        // "neutral" distance (factor = 1x) --
                                        // measure this at your typical
                                        // tracking distance
const float DISTANCE_FACTOR_MIN = 0.5;
const float DISTANCE_FACTOR_MAX = 2.0;

// --- LED matrix (8 rows x 13 columns, numbered 1-104, row-major) -----------
const int MATRIX_ROWS = 8;
const int MATRIX_COLS = 13;

// --- State -------------------------------------------------------------
bool linkDetected = false;
float linkX = 0.5, linkY = 0.5, linkWidth = 0.0, linkHeight = 0.0;
unsigned long lastLineMillis = 0;

float prevError = 0.0;
bool havePrevError = false;
float smoothedDError = 0.0;
unsigned long prevMicros = 0;
int lastLeftSpeed = 0, lastRightSpeed = 0;

String serialBuffer = "";

void setup() {
  Serial.begin(SERIAL_BAUD_RATE);

  pinMode(LEFT_MOTOR_PWM_PIN, OUTPUT);
  pinMode(LEFT_MOTOR_DIR_PIN, OUTPUT);
  pinMode(RIGHT_MOTOR_PWM_PIN, OUTPUT);
  pinMode(RIGHT_MOTOR_DIR_PIN, OUTPUT);

  prevMicros = micros();
}

// Parses one "detected,x,y,width,height" line into the link* globals.
// Malformed lines are ignored (lastLineMillis simply doesn't get refreshed,
// so the link-timeout logic in loop() still catches a persistently bad feed).
void parseLine(const String &line) {
  int fieldStart = 0;
  float fields[5];
  int fieldCount = 0;

  for (int i = 0; i <= (int)line.length() && fieldCount < 5; i++) {
    if (i == (int)line.length() || line[i] == ',') {
      fields[fieldCount++] = line.substring(fieldStart, i).toFloat();
      fieldStart = i + 1;
    }
  }
  if (fieldCount != 5) {
    return;  // malformed -- drop it
  }

  linkDetected = fields[0] != 0.0;
  linkX = fields[1];
  linkY = fields[2];
  linkWidth = fields[3];
  linkHeight = fields[4];
  lastLineMillis = millis();
}

void readSerial() {
  while (Serial.available() > 0) {
    char c = (char)Serial.read();
    if (c == '\n') {
      parseLine(serialBuffer);
      serialBuffer = "";
    } else if (c != '\r') {
      serialBuffer += c;
    }
  }
}

// Sets one motor's signed speed (-MAX_SPEED..+MAX_SPEED) via its Cytron
// Maker Drive PWM + DIR pin pair: PWM carries the magnitude, DIR selects
// direction. If a motor spins the wrong way, swap HIGH/LOW here (or swap
// that motor's two wires at the driver).
void driveMotor(int pwmPin, int dirPin, int speed) {
  speed = constrain(speed, -MAX_SPEED, MAX_SPEED);
  digitalWrite(dirPin, speed >= 0 ? HIGH : LOW);
  analogWrite(pwmPin, abs(speed));
}

// TODO: fill in once the UNO Q's actual LED matrix API is confirmed -- see
// the file header. ledIndexFor() already does the position math.
void setMatrixPixel(int index, bool on) {
  (void)index;
  (void)on;
}

int ledIndexFor(float x, float y) {
  int col = constrain((int)(x * MATRIX_COLS), 0, MATRIX_COLS - 1);
  int row = constrain((int)(y * MATRIX_ROWS), 0, MATRIX_ROWS - 1);
  return row * MATRIX_COLS + col + 1;  // matrix is numbered 1-104, row-major
}

void loop() {
  readSerial();

  unsigned long now = micros();
  float dt = (now - prevMicros) / 1000000.0;
  prevMicros = now;

  bool linkAlive = (millis() - lastLineMillis) <= LINK_TIMEOUT_MS;
  bool haveTarget = linkAlive && linkDetected;

  int desiredLeft = 0, desiredRight = 0;

  if (haveTarget) {
    float error = 0.5 - linkX;  // positive -> target is left of center

    float rawDError = 0.0;
    if (havePrevError && dt > 0) {
      rawDError = (error - prevError) / dt;
    }
    smoothedDError += D_SMOOTHING * (rawDError - smoothedDError);
    prevError = error;
    havePrevError = true;

    float distanceFactor = WIDTH_REF / max(linkWidth, 0.01f);
    distanceFactor = constrain(distanceFactor, DISTANCE_FACTOR_MIN, DISTANCE_FACTOR_MAX);

    if (fabs(error) > DEADZONE) {
      float raw = (KP * error + KD * smoothedDError) * distanceFactor;
      float magnitude = constrain(fabs(raw), MIN_SPEED, MAX_SPEED);
      float turn = (raw >= 0) ? magnitude : -magnitude;
      // Turn toward the target (rotate in place): left/right motors get
      // opposite signs. Swap these if it turns the wrong way.
      desiredLeft = (int)-turn;
      desiredRight = (int)turn;
    }
  } else {
    havePrevError = false;
    smoothedDError = 0.0;
  }

  // Slew-limit so actual motor speed glides toward desired instead of
  // jumping straight there, same reasoning as apriltag_seek_tracker.py.
  int leftStep = constrain(desiredLeft - lastLeftSpeed, -MAX_SPEED_STEP, MAX_SPEED_STEP);
  int rightStep = constrain(desiredRight - lastRightSpeed, -MAX_SPEED_STEP, MAX_SPEED_STEP);
  lastLeftSpeed += leftStep;
  lastRightSpeed += rightStep;

  driveMotor(LEFT_MOTOR_PWM_PIN, LEFT_MOTOR_DIR_PIN, lastLeftSpeed);
  driveMotor(RIGHT_MOTOR_PWM_PIN, RIGHT_MOTOR_DIR_PIN, lastRightSpeed);

  setMatrixPixel(ledIndexFor(linkX, linkY), haveTarget);
}
