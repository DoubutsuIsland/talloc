#include <Servo.h>

Servo servos[14];

void setup() {
  Serial.begin(115200);
}

void loop() {
  while (Serial.available() == 0) {};
  byte command = Serial.read();
  if (command == '\x00') {
    while (Serial.available() == 0) {};
    int pin = Serial.read();
    pinSetting(pin);
  } else if (command == '\x01') {
    while (Serial.available() == 0) {};
    int pin = Serial.read();
    pinWrite(pin);
  } else if (command == '\x02') {
    while (Serial.available() == 0) {};
    int pin = Serial.read();
    pinRead(pin);
  }
}

void pinSetting(byte pin) {
  if (pin <= 13) {
    pinMode(pin, INPUT);
  } else if (pin <= 25) {
    pinMode(pin -  12, INPUT_PULLUP);
  } else if (pin <= 37) {
    pinMode(pin -  24, OUTPUT);
  } else if (pin <= 49) {
    servos[pin - 36].attach(pin - 36);
  }
}

void pinWrite(byte pin) {
  if (pin <= 13) {
    digitalWrite(pin, LOW);
  } else if (pin <= 25) {
    digitalWrite(pin - 12, HIGH);
  } else if (pin <= 37) {
    while (Serial.available() == 0) {};
    unsigned int v = Serial.read();
    analogWrite(pin - 24, v);
  } else if (pin <= 49) {
    while (Serial.available() == 0) {};
    unsigned int angle = Serial.read();
    servos[pin - 36].write(angle);
  }
}

void pinRead(byte pin) {
  if (pin <= 13) {
    int state = digitalRead(pin);
    Serial.write(state);
  } else if (pin <= 25) {
    int v = analogRead(pin - 12);
    Serial.write(v);
  }
}
