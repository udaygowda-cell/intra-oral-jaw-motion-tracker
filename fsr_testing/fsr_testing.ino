const int ADC_PIN = 34;
const float VCC = 3.3;
const int ADC_MAX = 4095;
const float R_FIXED = 10000.0;

// NEW CALIBRATION constants based on your 1kg stone (Rfsr = 10883.66 Ω)
const float A_cal = 7000.0;
const float B_cal = -0.76;

void setup() {
  Serial.begin(115200);
  Serial.println("=== FSR Calibrated for 1kg Stone ===");
}

float readForceNewton() {
  long sum = 0;
  const int SAMPLES = 20;
  
  for (int i = 0; i < SAMPLES; i++) {
    sum += analogRead(ADC_PIN);
    delay(2);
  }
  
  float adc_avg = sum / (float)SAMPLES;
  if (adc_avg <= 10) return 0.0;
  
  float Vout = VCC * (adc_avg / ADC_MAX);
  float Rfsr = R_FIXED * (VCC / Vout - 1.0);
  float forceN = A_cal * pow(Rfsr, B_cal);
  
  return forceN;
}

void loop() {
  float force = readForceNewton();
  
  // Clear previous output with carriage return
  Serial.print("\rForce: ");
  
  if (force < 0.1) {
    Serial.println("0.0 N        "); // Extra spaces to clear previous text
  } else {
    Serial.print(force, 1); // Display with 1 decimal place
    Serial.println(" N        "); // Extra spaces to clear previous text
  }
  
  delay(200); // Update 5 times per second
}