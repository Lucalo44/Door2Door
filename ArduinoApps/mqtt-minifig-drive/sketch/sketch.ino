// Mirrors mqtt-minifig-monitor's sketch (Arduino_RouterBridge +
// Arduino_LED_Matrix) and adds a second Bridge.provide() for driving two
// motors. This sketch is a thin executor with no control logic of its own:
// it draws whatever frame Python last sent, and drives whatever signed
// left/right speeds Python last sent. The PD control math itself runs even
// further upstream, on the laptop in Door2Door/YOLO/detect_publish.py --
// python/main.py here just forwards those speeds (and hard-stops them if
// messages stop arriving).
//
// Motor driver interface (confirmed against the board's own silkscreen,
// not a datasheet guess): each motor gets two PWM-capable pins and no
// separate direction pin -- PWM one pin for forward at that speed (the
// other held at 0), PWM the other pin instead for reverse. See
// driveMotor() below. Screw terminals on the driver board (not sketch
// concerns): each motor's two leads go to that channel's motor output
// terminal, and the motor battery goes to VB+/VB-.

#include <Arduino_RouterBridge.h>
#include <Arduino_LED_Matrix.h>
#include <vector>

Arduino_LED_Matrix matrix;

const uint8_t FRAME_ROWS = 8;
const uint8_t FRAME_COLS = 13;
const uint8_t FRAME_SIZE = FRAME_ROWS * FRAME_COLS;

uint8_t frame[FRAME_SIZE] = {0};

// Driver board pins -- placeholders, confirm against your actual wiring.
// Motor 1 = left wheel, Motor 2 = right wheel; swap here if that is backwards.
const int MOTOR1_A_PIN = 5;   // M1A -- ~D5
const int MOTOR1_B_PIN = 6;   // M1B -- ~D6
const int MOTOR2_A_PIN = 9;   // M2A -- ~D9
const int MOTOR2_B_PIN = 10;  // M2B -- ~D10
// ~D3 and ~D11 are spare PWM-capable pins, unused here.
const int MAX_SPEED = 255;    // must match MAX_SPEED in main.py

void setup() {
  matrix.begin();
  matrix.setGrayscaleBits(3);
  matrix.clear();

  pinMode(MOTOR1_A_PIN, OUTPUT);
  pinMode(MOTOR1_B_PIN, OUTPUT);
  pinMode(MOTOR2_A_PIN, OUTPUT);
  pinMode(MOTOR2_B_PIN, OUTPUT);

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

// speed: -MAX_SPEED..MAX_SPEED. Exactly one of pinA/pinB is ever nonzero:
// PWM pinA for forward at that magnitude, PWM pinB instead for reverse. If
// a motor spins the wrong way, swap pinA/pinB in the two calls in drive()
// below (or swap that motor's two wires at the driver's screw terminal).
void driveMotor(int pinA, int pinB, int speed) {
  speed = constrain(speed, -MAX_SPEED, MAX_SPEED);
  if (speed >= 0) {
    analogWrite(pinA, speed);
    analogWrite(pinB, 0);
  } else {
    analogWrite(pinA, 0);
    analogWrite(pinB, -speed);
  }
}

// Called from Python with already-PD-controlled signed speeds
// (-MAX_SPEED..MAX_SPEED) for each motor -- computed on the laptop in
// detect_publish.py's control_step(), forwarded as-is by main.py here.
void drive(int leftSpeed, int rightSpeed) {
  driveMotor(MOTOR1_A_PIN, MOTOR1_B_PIN, leftSpeed);
  driveMotor(MOTOR2_A_PIN, MOTOR2_B_PIN, rightSpeed);
}
