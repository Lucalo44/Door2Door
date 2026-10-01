#include "Arduino_RouterBridge.h"

const int LED_BLUE = LED_BUILTIN + 2;

void setup() {
    pinMode(LED_BLUE, OUTPUT);

    Bridge.begin();
    Bridge.provide("set_led_state", set_led_state);
}

void loop() {
}

void set_led_state(bool state) {
    // LOW state means LED is ON
    digitalWrite(LED_BLUE, state ? LOW : HIGH);
}
