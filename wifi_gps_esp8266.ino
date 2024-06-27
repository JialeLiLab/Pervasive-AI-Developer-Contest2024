//Server
#include "ESP8266WiFi.h"
#include <WifiLocation.h>
 
const char* ssid = "";//WiFi name
const char* password =  "";//wifi pwd
const char* googleApiKey = "";// google api key
 
WiFiServer wifiServer(8080);//port
WifiLocation location (googleApiKey);

// Set time via NTP, as required for x.509 validation
void setClock () {
    configTime (0, 0, "pool.ntp.org", "time.nist.gov");

    // Serial.print ("Waiting for NTP time sync: ");
    time_t now = time (nullptr);
    while (now < 8 * 3600 * 2) {
        delay (200); // net delay
        Serial.print (".");
        now = time (nullptr);
    }
    struct tm timeinfo;
    gmtime_r (&now, &timeinfo);
    // Serial.print ("\n");
    // Serial.print ("Current time: ");
    // Serial.print (asctime (&timeinfo));
}


void setup() {
  Serial.begin(115200);
  delay(1000);
  WiFi.begin(ssid, password);
  while (WiFi.status() != WL_CONNECTED) {
    delay(1000);
    Serial.println("Connecting..");
  }
  Serial.print("Connected to WiFi. IP:");
  Serial.println(WiFi.localIP());
  wifiServer.begin();
}
 
void loop() {
  WiFiClient client = wifiServer.available();
  String msgs = "";
  
  if (client) {
    while (client.connected()) {

      while (Serial.available() > 0) {
        char s = Serial.read();
        Serial.write(s);
        if (s == '\n') {
          // msgs += s;
          break;  // end of message
        }
        msgs += s;
        delay(10);
      }

      while (client.available() > 0) {
        char c = client.read();
        Serial.write(c);
        if (c == '\n') {
          msgs += c;
          break;  // end of message
        }
        msgs += c;
      }

      if (msgs == "sendLocation") { // the signal of sending the location
          setClock ();
          location_t loc = location.getGeoFromWiFi();
          client.write((String (loc.lat, 7) + " " + String (loc.lon, 7)).c_str()); // the actual location
          msgs = "";
      }
      // clear buff
      else if (msgs.length() > 0) {
        client.write(msgs.c_str());
        msgs = "";
      }
    }
    client.stop();
    Serial.println("Client disconnected");
  }
}
