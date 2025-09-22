#include <Wire.h>
#include <WiFi.h>
#include <WiFiClientSecure.h>
#include <HTTPClient.h>
#include <ArduinoJson.h>
#include <Adafruit_INA219.h>
#include <Adafruit_BME280.h>

// ========== WiFi Credentials ==========
const char* ssid = "";
const char* password = "";

//Thingspeak API
const char* THINGSPEAK_WRITE_API_KEY = "";
const char* THINGSPEAK_URL = "http://api.thingspeak.com/update";
unsigned long lastThingSpeak = 0;
const unsigned long thingSpeakInterval = 15000; // 15 sec

// ========== Supabase Config ==========
const char* supabaseUrl = "";
const char* apiKey = "";
const char* bearerToken = "";  // keep full token here

// ========== Sensors ==========
Adafruit_INA219 ina219;
Adafruit_BME280 bme;

// ========== Energy Tracking ==========
float energy_Wh = 0.0;              // cumulative energy
const float interval_s = 5.0;       // measurement interval (seconds)

// ========== Functions ==========
void sendData(float voltage, float current, float power, float energy,
              float temperature, float humidity, float pressure) {
  if (WiFi.status() == WL_CONNECTED) {
    WiFiClientSecure client;
    client.setInsecure();  // ✅ disable cert verification
    HTTPClient http;

    if (http.begin(client, supabaseUrl)) {
      http.addHeader("apikey", apiKey);
      http.addHeader("Authorization", String("Bearer ") + bearerToken);
      http.addHeader("Content-Type", "application/json");

      // JSON payload
      String payload;
      StaticJsonDocument<512> doc;
      doc["voltage"] = voltage;
      doc["current"] = current;
      doc["power"] = power;
      doc["energy"] = energy;
      doc["temperature"] = temperature;
      doc["humidity"] = humidity;
      doc["pressure"] = pressure;
      serializeJson(doc, payload);

      int httpResponseCode = http.POST(payload);

      if (httpResponseCode > 0) {
        Serial.print("POST Response: ");
        Serial.println(httpResponseCode);
      } else {
        Serial.print("POST Error: ");
        Serial.println(http.errorToString(httpResponseCode).c_str());
      }
      http.end();
    }
  } else {
    Serial.println("WiFi Disconnected");
  }
}

// =========================================================
// Send data to ThingSpeak
// =========================================================
void sendToThingSpeak(float voltage, float current, float power, float energy,
                      float temperature, float humidity, float pressure) {
  if (WiFi.status() == WL_CONNECTED) {
    WiFiClient client;
    HTTPClient http;

    String postData = String("api_key=") + THINGSPEAK_WRITE_API_KEY +
                      "&field1=" + String(voltage, 2) +
                      "&field2=" + String(current, 3) +
                      "&field3=" + String(power, 3) +
                      "&field4=" + String(energy, 3) +
                      "&field5=" + String(temperature, 2) +
                      "&field6=" + String(humidity, 2) +
                      "&field7=" + String(pressure, 2);

    http.begin(client, THINGSPEAK_URL);
    http.addHeader("Content-Type", "application/x-www-form-urlencoded");

    int httpResponseCode = http.POST(postData);
    if (httpResponseCode > 0) {
      Serial.printf("ThingSpeak Response: %d\n", httpResponseCode);
    } else {
      Serial.printf("ThingSpeak Error: %s\n", http.errorToString(httpResponseCode).c_str());
    }

    http.end();
  }
}

void setup() {
  Serial.begin(115200);
  WiFi.begin(ssid, password);

  Serial.print("Connecting to WiFi");
  while (WiFi.status() != WL_CONNECTED) {
    delay(500);
    Serial.print(".");
  }
  Serial.println("\n Connected to WiFi");

  // Init INA219
  if (!ina219.begin()) {
    Serial.println("Failed to find INA219 chip");
    while (1) { delay(10); }
  }
  Serial.println("INA219 initialized");

  // Init BME280
  if (!bme.begin(0x76)) {
    Serial.println("Could not find BME280 sensor!");
    while (1) { delay(10); }
  }
  Serial.println("BME280 initialized");
}

void loop() {
  // Read INA219
  float shuntVoltage = ina219.getShuntVoltage_mV() / 1000.0;  // V
  float busVoltage   = ina219.getBusVoltage_V();
  float current_A    = ina219.getCurrent_mA() / 1000.0;       // A
  float power_W      = ina219.getPower_mW() / 1000.0;         // W
  float loadVoltage  = busVoltage + shuntVoltage;

  // Read BME280
  float temperature = bme.readTemperature();   // °C
  float humidity    = bme.readHumidity();      // %
  float pressure    = bme.readPressure() / 100.0; // hPa

  // Update energy (Wh)
  energy_Wh += power_W * (interval_s / 3600.0);

  // Debug log
  Serial.println("===== Sensor Readings =====");
  Serial.printf("Voltage: %.3f V\n", loadVoltage);
  Serial.printf("Current: %.3f A\n", current_A);
  Serial.printf("Power: %.3f W\n", power_W);
  Serial.printf("Energy: %.3f Wh\n", energy_Wh);
  Serial.printf("Temp: %.2f °C | Humidity: %.2f %% | Pressure: %.2f hPa\n\n",
                temperature, humidity, pressure);

  // Send data
  sendData(loadVoltage, current_A, power_W, energy_Wh, temperature, humidity, pressure);

  // Send to ThingSpeak (every 15s)
  if (millis() - lastThingSpeak >= thingSpeakInterval) {
    sendToThingSpeak(loadVoltage, current_A, power_W, energy_Wh, temperature, humidity, pressure);
    lastThingSpeak = millis();
  }

  delay(interval_s * 1000); // send every interval
}