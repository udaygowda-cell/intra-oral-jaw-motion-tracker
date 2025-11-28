/*******************************************************
 *  PART 0 — LIBRARIES & GLOBALS
 *******************************************************/
#include <Wire.h>
#include <WiFi.h>
#include <HTTPClient.h>
#include <Adafruit_Sensor.h>
#include <Adafruit_LSM303_U.h>

const char* ssid = "udaykumar";
const char* password = "12345678";

const char* googleScriptURL =
"https://script.google.com/macros/s/AKfycbzdvplDKRuwVOrPwcMXgUaqN-IfOPwstGR5_PjyciANSCGSnRiCqUfNjC5BZTO7uUb2/exec";

Adafruit_LSM303_Accel_Unified accel(12345);
Adafruit_LSM303_Mag_Unified   mag(54321);

// Sampling 50 Hz
const unsigned long SAMPLE_INTERVAL = 20;

// Send every 2 seconds
const unsigned long SEND_INTERVAL = 2000;

unsigned long lastSample = 0;
unsigned long lastSend = 0;

// Raw IMU values
float ax, ay, az;
float mx, my, mz;

/*******************************************************
 *  FILTER CONSTANTS & STATE
 *******************************************************/
const float ACC_ALPHA = 0.36;   // Low-pass for accelerometer (10 Hz)
const float MAG_ALPHA = 0.10;   // Light smoothing for magnetometer

float fax = 0, fay = 0, faz = 0;
float fmx = 0, fmy = 0, fmz = 0;

/*******************************************************
 *  FILTER FUNCTION
 *******************************************************/
void applyFilters() {
  // Accelerometer low-pass filter (IIR)
  fax = ACC_ALPHA * ax + (1 - ACC_ALPHA) * fax;
  fay = ACC_ALPHA * ay + (1 - ACC_ALPHA) * fay;
  faz = ACC_ALPHA * az + (1 - ACC_ALPHA) * faz;

  // Magnetometer smoothing
  fmx = MAG_ALPHA * mx + (1 - MAG_ALPHA) * fmx;
  fmy = MAG_ALPHA * my + (1 - MAG_ALPHA) * fmy;
  fmz = MAG_ALPHA * mz + (1 - MAG_ALPHA) * fmz;
}

/*******************************************************
 *  READ IMU
 *******************************************************/
void readIMU() {
  sensors_event_t accEvent, magEvent;

  accel.getEvent(&accEvent);
  mag.getEvent(&magEvent);

  ax = accEvent.acceleration.x;
  ay = accEvent.acceleration.y;
  az = accEvent.acceleration.z;

  mx = magEvent.magnetic.x;
  my = magEvent.magnetic.y;
  mz = magEvent.magnetic.z;
}

/*******************************************************
 *  WIFI RECONNECT
 *******************************************************/
void ensureWiFi() {
  if (WiFi.status() == WL_CONNECTED) return;

  WiFi.begin(ssid, password);
  unsigned long start = millis();
  while (WiFi.status() != WL_CONNECTED && millis() - start < 5000) {
    delay(200);
  }
}

/*******************************************************
 *  URL ENCODER
 *******************************************************/
String urlEncode(const String &s) {
  String out;
  for (int i = 0; i < s.length(); i++) {
    char c = s[i];
    if (isalnum(c)) out += c;
    else if (c == ' ') out += '+';
    else {
      char hex[4];
      sprintf(hex, "%%%02X", (uint8_t)c);
      out += hex;
    }
    yield();
  }
  return out;
}

/*******************************************************
 *  SEND FILTERED DATA
 *******************************************************/
void sendToGoogleSheets(String data) {
  ensureWiFi();
  if (WiFi.status() != WL_CONNECTED) return;

  HTTPClient http;
  http.setTimeout(8000);

  String url = String(googleScriptURL) + "?data=" + urlEncode(data);
  http.begin(url);
  http.GET();
  http.end();
}

/*******************************************************
 *  SETUP
 *******************************************************/
void setup() {
  Serial.begin(115200);
  Wire.begin();

  accel.begin();
  mag.begin();

  WiFi.begin(ssid, password);
  while (WiFi.status() != WL_CONNECTED) {
    delay(300);
  }
}

/*******************************************************
 *  LOOP — SAMPLE & SEND EVERY 2 SEC
 *******************************************************/
void loop() {
  unsigned long now = millis();

  // Read IMU at 50 Hz
  if (now - lastSample >= SAMPLE_INTERVAL) {
    readIMU();
    applyFilters();   // IMPORTANT
    lastSample = now;

    Serial.printf("ACC: %.2f %.2f %.2f | MAG: %.2f %.2f %.2f\n",
                  fax, fay, faz, fmx, fmy, fmz);
  }

  // Send filtered data every 2 seconds
  if (now - lastSend >= SEND_INTERVAL) {

    String packet =
      String(now) + "," +
      String(fax, 2) + "," +
      String(fay, 2) + "," +
      String(faz, 2) + "," +
      String(fmx, 2) + "," +
      String(fmy, 2) + "," +
      String(fmz, 2);

    Serial.println("SEND: " + packet);

    sendToGoogleSheets(packet);

    lastSend = now;
  }
}
