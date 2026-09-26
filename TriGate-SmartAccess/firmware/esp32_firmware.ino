/*
 * esp32_firmware.ino - TriGate ESP32-S3 firmware
 *
 * Corrected in the TriGate code review - Document 5 of 6.
 * Full explanation, tests and installation steps: TriGate_Doc5_esp32_firmware.pdf
 *
 * Same pins, same commands, same screens as the original. Every change is
 * marked next to the lines it touches:
 *
 *   FIX W1  OLED fails to start -> carry on without the screen
 *           (was for(;;): the whole board froze and looked dead)
 *   FIX W2  lockout: keep reading serial - RESET works, anything else is
 *           dropped (was deaf for 60 s, then ran every stored command -
 *           a GRANTED sent during the lockout opened the gate ~53 s later);
 *           heartbeat keeps going; screen redrawn once a second, not
 *           nonstop (the likely cause of the OLED corruption)
 *   FIX W3  GRANTED no longer blocks for 5 s; the gate closes itself after
 *           6 s only if the Pi never sent CLOSE
 *   FIX W4  DENIED no longer blocks for 3 s; the lockout starts at once
 *   FIX W5  I2C at 400 kHz (was the 100 kHz default)
 *   FIX W6  CLOSE also turns the green light off and resets the screen
 *   FIX W7  new command IDLE: stop scanning (sent by the fixed
 *           mqtt_client.py; the original firmware ignored it)
 *   FIX W8  RESET lets an "ACCESS DENIED" message that is already showing
 *           finish (the original Pi software sends RESET right after
 *           every DENIED)
 *   FIX W9  GRANTED (and CLOSE) now clear deniedShowing too. Without this,
 *           a GRANTED arriving within 3 s of a prior DENIED left
 *           deniedShowing set to true; loop()'s denied-timeout code then
 *           fired ~3 s later, saw isScanning==false && isLockedOut==false
 *           (both true right after GRANTED), and silently repainted the
 *           screen from "ACCESS GRANTED / Gate Opening..." to "READY" -
 *           even though the gate was still physically open for up to 6 s
 *           more. RESET already guarded against this (FIX W8); GRANTED
 *           did not.
 *
 * Upload exactly as before - no library or board setting changes.
 */
#include <Wire.h>
#include <Adafruit_GFX.h>
#include <Adafruit_SSD1306.h>
#include <ESP32Servo.h>

#define SERVO_PIN    4
#define BUZZER_PIN   5
#define RED_PIN      6
#define YELLOW_PIN   7
#define GREEN_PIN    10
#define I2C_SDA      8
#define I2C_SCL      9

#define SCREEN_WIDTH 128
#define SCREEN_HEIGHT 64
Adafruit_SSD1306 display(SCREEN_WIDTH, SCREEN_HEIGHT, &Wire, -1);

Servo myServo;

int failCount = 0;
bool isLockedOut = false;
unsigned long lockoutStart = 0;
unsigned long lastHeartbeat = 0;
unsigned long yellowBlinkTimer = 0;
bool yellowBlinkState = false;
bool isScanning = false;
bool hasDisplay = false;            // FIX W1
bool gateOpen = false;              // FIX W3
unsigned long gateOpenedAt = 0;     // FIX W3
bool deniedShowing = false;         // FIX W4
unsigned long deniedAt = 0;         // FIX W4
int lastLockoutShown = -1;          // FIX W2

void setup() {
  Serial.begin(115200);
  pinMode(BUZZER_PIN, OUTPUT);
  pinMode(RED_PIN, OUTPUT);
  pinMode(YELLOW_PIN, OUTPUT);
  pinMode(GREEN_PIN, OUTPUT);
  Wire.begin(I2C_SDA, I2C_SCL);
  Wire.setClock(400000);   // FIX W5: default 100 kHz made every screen update ~4x slower
  hasDisplay = display.begin(SSD1306_SWITCHCAPVCC, 0x3C);
  if (!hasDisplay) {
    // FIX W1: was for(;;) - the board froze here forever: servo never
    // attached, serial never read, looked completely dead. Carry on
    // without the screen instead.
    Serial.println("OLED FAILED - continuing without display");
  }
  for (int i = 0; i < 3; i++) {
    digitalWrite(RED_PIN, HIGH); delay(200); digitalWrite(RED_PIN, LOW);
    digitalWrite(YELLOW_PIN, HIGH); delay(200); digitalWrite(YELLOW_PIN, LOW);
    digitalWrite(GREEN_PIN, HIGH); delay(200); digitalWrite(GREEN_PIN, LOW);
  }
  myServo.setPeriodHertz(50);
  myServo.attach(SERVO_PIN, 500, 2400);
  myServo.write(0);
  showScreen("GARAGE SYSTEM", "READY", "");
  Serial.println("ESP32_READY");
}

void loop() {
  // FIX W2: heartbeat moved up so it keeps going during a lockout too
  if (millis() - lastHeartbeat > 10000) {
    Serial.println("HEARTBEAT");
    lastHeartbeat = millis();
  }

  // FIX W3: the gate closes itself after 6 s ONLY if the Pi never sent
  // CLOSE (the Pi sends it at 5 s). This used to be delay(5000) inside
  // handleCommand, which stopped serial being read for 5 s.
  if (gateOpen && millis() - gateOpenedAt > 6000) {
    closeGate();
  }

  // FIX W4: end of the 3 s "ACCESS DENIED" screen, without delay(3000)
  if (deniedShowing && millis() - deniedAt > 3000) {
    deniedShowing = false;
    digitalWrite(RED_PIN, LOW);
    if (!isScanning && !isLockedOut) showScreen("GARAGE SYSTEM", "READY", "");
  }

  if (isLockedOut) {
    // FIX W2: the old code did `return` here before ever reaching
    // Serial.available(), so for 60 s the board was deaf: RESET could not
    // reach it, and every command sent meanwhile piled up and ran all at
    // once when the lockout ended. Now RESET works and anything else is
    // read and dropped.
    if (Serial.available()) {
      String cmd = Serial.readStringUntil('\n');
      cmd.trim();
      if (cmd == "RESET") { handleCommand(cmd); return; }
      if (cmd.length() > 0) Serial.println("IGNORED_LOCKOUT " + cmd);
    }
    unsigned long elapsed = (millis() - lockoutStart) / 1000;
    int remaining = 60 - elapsed;
    if (remaining <= 0) {
      isLockedOut = false;
      failCount = 0;
      allLedsOff();
      showScreen("GARAGE SYSTEM", "READY", "");
      Serial.println("LOCKOUT_CLEARED");
    } else {
      digitalWrite(RED_PIN, (millis() / 500) % 2);
      // FIX W2: was redrawn on EVERY loop pass - 1024 bytes over I2C each
      // time, nonstop for 60 s, while the text changes once per second.
      // That flooding is the most likely cause of the OLED corruption.
      if (remaining != lastLockoutShown) {
        lastLockoutShown = remaining;
        showScreen("!! LOCKOUT !!", String(remaining) + "s remaining", "");
      }
    }
    return;
  }

  if (isScanning) {
    if (millis() - yellowBlinkTimer > 400) {
      yellowBlinkState = !yellowBlinkState;
      digitalWrite(YELLOW_PIN, yellowBlinkState);
      yellowBlinkTimer = millis();
    }
  }

  if (Serial.available()) {
    String cmd = Serial.readStringUntil('\n');
    cmd.trim();
    handleCommand(cmd);
  }
}

void handleCommand(String cmd) {
  if (cmd == "SCANNING") {
    isScanning = true;
    allLedsOff();
    showScreen("SCANNING...", "Please wait", "");
  } else if (cmd == "GRANTED") {
    isScanning = false;
    failCount = 0;
    deniedShowing = false;    // FIX W9: see note above handleCommand's header
    allLedsOff();
    digitalWrite(GREEN_PIN, HIGH);
    myServo.write(90);
    beep(100); delay(100); beep(100);
    showScreen("ACCESS GRANTED", "Welcome!", "Gate Opening...");
    gateOpen = true;            // FIX W3: was delay(5000) + close
    gateOpenedAt = millis();
  } else if (cmd == "DENIED") {
    isScanning = false;
    failCount++;
    allLedsOff();
    digitalWrite(RED_PIN, HIGH);
    beep(1000);
    showScreen("ACCESS DENIED", "Unauthorized!", "Attempt: " + String(failCount) + "/3");
    if (failCount >= 3) {
      triggerLockout();
    } else {
      deniedShowing = true;     // FIX W4: was delay(3000)
      deniedAt = millis();
    }
  } else if (cmd == "LOCKOUT") {
    triggerLockout();
  } else if (cmd == "OPEN") {
    myServo.write(90);
    Serial.println("GATE_OPENED");
  } else if (cmd == "CLOSE") {
    closeGate();                // FIX W6: also turns the green LED off and
    Serial.println("GATE_CLOSED");  // resets the screen (it used to rely on
                                    // GRANTED's delay(5000) doing that)
  } else if (cmd == "IDLE") {       // FIX W7: sent by the fixed mqtt_client.py
    isScanning = false;             // stop scanning: yellow light off, and
    digitalWrite(YELLOW_PIN, LOW);  // back to READY - unless a GRANTED or
    if (!gateOpen && !deniedShowing) {   // DENIED screen is still showing
      showScreen("GARAGE SYSTEM", "READY", "");
    }
  } else if (cmd == "RESET") {
    isScanning = false;
    isLockedOut = false;
    failCount = 0;
    gateOpen = false;           // FIX W3
    lastLockoutShown = -1;      // FIX W2
    myServo.write(0);
    // FIX W8: the original Pi software sends RESET straight after every
    // DENIED. Now that DENIED no longer blocks (W4), that RESET would wipe
    // the "ACCESS DENIED" message at once. Let a denial that is already
    // showing finish its 3 s; everything else about RESET is unchanged.
    if (deniedShowing) {
      digitalWrite(YELLOW_PIN, LOW);
      digitalWrite(GREEN_PIN, LOW);
    } else {
      allLedsOff();
      showScreen("GARAGE SYSTEM", "READY", "");
    }
    Serial.println("RESET_OK");
  }
}

void triggerLockout() {
  isLockedOut = true;
  isScanning = false;
  lockoutStart = millis();
  lastLockoutShown = -1;        // FIX W2
  deniedShowing = false;        // FIX W4
  allLedsOff();
  for (int i = 0; i < 5; i++) {
    beep(200); delay(100);
  }
  Serial.println("LOCKOUT_ACTIVE");
}

void beep(int duration) {
  digitalWrite(BUZZER_PIN, HIGH);
  delay(duration);
  digitalWrite(BUZZER_PIN, LOW);
}

// FIX W3/W6/W9: one place that closes the gate and tidies up
void closeGate() {
  gateOpen = false;
  deniedShowing = false;   // FIX W9: keep this in sync with GRANTED's reset
  myServo.write(0);
  digitalWrite(GREEN_PIN, LOW);
  showScreen("GARAGE SYSTEM", "READY", "");
}

void allLedsOff() {
  digitalWrite(RED_PIN, LOW);
  digitalWrite(YELLOW_PIN, LOW);
  digitalWrite(GREEN_PIN, LOW);
}

void showScreen(String line1, String line2, String line3) {
  if (!hasDisplay) return;      // FIX W1
  display.clearDisplay();
  display.setTextColor(SSD1306_WHITE);
  display.setTextSize(1);
  display.setCursor(0, 0);
  display.println("===================");
  display.setCursor(0, 16);
  display.println(line1);
  display.setCursor(0, 32);
  display.println(line2);
  display.setCursor(0, 48);
  display.println(line3);
  display.display();
}
