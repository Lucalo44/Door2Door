#include "Arduino_LED_Matrix.h"
#include "Arduino_RouterBridge.h"

const int MATRIX_ROWS = 8;
const int MATRIX_COLS = 13;

Arduino_LED_Matrix matrix;
uint8_t frame[MATRIX_ROWS * MATRIX_COLS];

void setup() {
    matrix.begin();
    matrix.clear();

    Bridge.begin();
    Bridge.provide("draw_frame", draw_frame);
}

void loop() {
}

// bits is a string of '0'/'1', one per pixel, row-major (row*MATRIX_COLS + col)
void draw_frame(String bits) {
    int count = MATRIX_ROWS * MATRIX_COLS;
    for (int i = 0; i < count; i++) {
        frame[i] = (i < bits.length() && bits.charAt(i) == '1') ? 7 : 0;
    }
    matrix.draw(frame);
}
