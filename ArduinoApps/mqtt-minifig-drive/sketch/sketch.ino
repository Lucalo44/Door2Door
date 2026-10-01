// Mirrors mqtt-minifig-monitor's sketch (Arduino_RouterBridge +
// Arduino_LED_Matrix) and adds a second Bridge.provide() for driving two
// motors. All position/control logic lives in python/main.py -- this sketch
// is a thin executor: it draws whatever frame Python last sent, and drives
// whatever signed left/right speeds Python last computed.

#include <Arduino_RouterBridge.h>
#include <Arduino_LED_Matrix.h>
#include <vector>

Arduino_LED_Matrix matrix;

const uint8_t FRAME_ROWS = 8;
const uint8_t FRAME_COLS = 13;
const uint8_t FRAME_SIZE = FRAME_ROWS * FRAME_COLS;

uint8_t frame[FRAME_SIZE] = {0};

// Cytron Maker Drive -- one PWM pin (speed magnitude) plus one DIR pin
// (direction, HIGH/LOW) per motor. Placeholders -- confirm against your
// actual Maker Drive wiring. If a motor spins the wrong way, swap HIGH/LOW
// in driveMotor() below (or swap that motor's two wires at the driver).
const int LEFT_MOTOR_PWM_PIN = 5;   // ~D5
const int LEFT_MOTOR_DIR_PIN = 4;   // D4
const int RIGHT_MOTOR_PWM_PIN = 9;  // ~D9
const int RIGHT_MOTOR_DIR_PIN = 7;  // D7
const int MAX_SPEED = 255;           // must match MAX_SPEED in main.py

void setup() {
  matrix.begin();
  matrix.setGrayscaleBits(3);
  matrix.clear();

  pinMode(LEFT_MOTOR_PWM_PIN, OUTPUT);
  pinMode(LEFT_MOTOR_DIR_PIN, OUTPUT);
  pinMode(RIGHT_MOTOR_PWM_PIN, OUTPUT);
  pinMode(RIGHT_MOTOR_DIR_PIN, OUTPUT);

  Bridge.begin();
  Bridge.provide("draw", draw);
  Bridge.provide("drive", drive);
}

void loop() {
  matrix.draw(frame);
  delay(10);
}

// Called from Python with a new frame to display whenever the tracked
// position updates -- identical to mqtt-minifig-monitor's draw().
void draw(std::vector<uint8_t> newFrame) {
  size_t len = min(newFrame.size(), (size_t)FRAME_SIZE);
  memcpy(frame, newFrame.data(), len);
}

void driveMotor(int pwmPin, int dirPin, int speed) {
  speed = constrain(speed, -MAX_SPEED, MAX_SPEED);
  digitalWrite(dirPin, speed >= 0 ? HIGH : LOW);
  analogWrite(pwmPin, abs(speed));
}

// Called from Python with already-PD-controlled signed speeds
// (-MAX_SPEED..MAX_SPEED) for each motor -- see main.py's control_step().
void drive(int leftSpeed, int rightSpeed) {
  driveMotor(LEFT_MOTOR_PWM_PIN, LEFT_MOTOR_DIR_PIN, leftSpeed);
  driveMotor(RIGHT_MOTOR_PWM_PIN, RIGHT_MOTOR_DIR_PIN, rightSpeed);
}
