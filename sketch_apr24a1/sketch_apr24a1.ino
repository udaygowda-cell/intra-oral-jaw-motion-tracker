#include <Wire.h>
#include <Adafruit_Sensor.h>
#include <Adafruit_LSM303_U.h>

// Create sensor objects
Adafruit_LSM303_Mag_Unified mag = Adafruit_LSM303_Mag_Unified(12345);
Adafruit_LSM303_Accel_Unified accel = Adafruit_LSM303_Accel_Unified(54321);

void setup() {
  Serial.begin(115200);
  Wire.begin();

  // Initialize accelerometer
  if (!accel.begin()) {
    Serial.println("No LSM303 accelerometer detected!");
    while (1);
  }

  // Initialize magnetometer
  if (!mag.begin()) {
    Serial.println("No LSM303 magnetometer detected!");
    while (1);
  }

  Serial.println("LSM303 Initialized!");
}

void loop() {
  // Get accelerometer data
  sensors_event_t accelEvent;
  accel.getEvent(&accelEvent);

  // Get magnetometer data
  sensors_event_t magEvent;
  mag.getEvent(&magEvent);

  // Send only magnetometer data (X, Y, Z) to match Python program
  Serial.print(magEvent.magnetic.x);
  Serial.print(",");
  Serial.print(magEvent.magnetic.y);
  Serial.print(",");
  Serial.println(magEvent.magnetic.z);

  delay(50); // ~20 Hz update rate
}
