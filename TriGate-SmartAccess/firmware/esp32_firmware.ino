/* =======================================================================
   esp32_firmware.ino  -- PATCHED

   This is YOUR file. Same pins, same commands, same screens, same
   structure. Six defects fixed, nothing redesigned. Every change is
   marked  // [FIX n]  and listed in DIFFS.md with the original line
   numbers.

   FIX 1  line 37   for(;;) on OLED failure froze the whole board
   FIX 2  line 63   showScreen() every loop pass during lockout was
                    saturating the I2C bus -- the likely OLED corruption
   FIX 3  line 65   `return` during lockout made the board deaf to serial
                    for 60 seconds; RESET could not reach it
   FIX 4  line 101  delay(5000) in GRANTED blocked serial reads
   FIX 5  line 112  delay(3000) in DENIED blocked serial reads
   FIX 6  line 144  1.5 s of blocking beeps in triggerLockout

   Everything the Pi sends still works exactly as before.
   ======================================================================= */

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

// [FIX 1] the board keeps running when the display is missing
bool hasDisplay = false;
unsigned long lastOledTry = 0;

// [FIX 2] remember what is on screen so we only redraw on change
String scr1 = "", scr2 = "", scr3 = "";

// [FIX 4/5/6] deferred actions, replacing the blocking delay() calls
unsigned long gateCloseAt   = 0;   // 0 = nothing pending
unsigned long deniedClearAt = 0;
int      beepsLeft   = 0;
int      beepOnMs    = 0;
int      beepOffMs   = 0;
unsigned long beepPhase = 0;
bool     beepHigh    = false;

void setup() {
  Serial.begin(115200);
  pinMode(BUZZER_PIN, OUTPUT);
  pinMode(RED_PIN, OUTPUT);
  pinMode(YELLOW_PIN, OUTPUT);
  pinMode(GREEN_PIN, OUTPUT);

  Wire.begin(I2C_SDA, I2C_SCL);
  Wire.setClock(400000);           // [FIX 2] 100 kHz default made every
                                   // full-screen update cost ~90 ms
  hasDisplay = display.begin(SSD1306_SWITCHCAPVCC, 0x3C);
  if (!hasDisplay) {
    Serial.println("OLED FAILED - continuing without display");
    // [FIX 1] was: for (;;);
    // That single line is why the board looked completely dead whenever
    // the I2C handshake failed. It froze inside setup(), so the servo was
    // never attached and the serial command loop was never reached.
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
  lastOledTry = millis();
}

void loop() {
  unsigned long now = millis();

  // ---- [FIX 4/5/6] service the deferred work first ------------------
  serviceBeep();

  if (gateCloseAt && now >= gateCloseAt) {
    gateCloseAt = 0;
    myServo.write(0);
    digitalWrite(GREEN_PIN, LOW);
    showScreen("GARAGE SYSTEM", "READY", "");
    Serial.println("GATE_CLOSED");
  }

  if (deniedClearAt && now >= deniedClearAt) {
    deniedClearAt = 0;
    digitalWrite(RED_PIN, LOW);
    if (failCount >= 3) triggerLockout();
    else                showScreen("GARAGE SYSTEM", "READY", "");
  }

  // [FIX 1] retry the display instead of giving up for good
  if (!hasDisplay && now - lastOledTry > 5000) {
    lastOledTry = now;
    i2cRecover();
    Wire.begin(I2C_SDA, I2C_SCL);
    Wire.setClock(400000);
    hasDisplay = display.begin(SSD1306_SWITCHCAPVCC, 0x3C);
    if (hasDisplay) {
      Serial.println("OLED_RECOVERED");
      scr1 = ""; showScreen(scr1, scr2, scr3);
    }
  }

  // ---- [FIX 3] serial is read BEFORE the lockout branch -------------
  // In the original, lockout hit `return` at line 65 and never reached
  // Serial.available() at line 81. The board was deaf for a full 60 s:
  // RESET could not clear it, and every command the Pi sent in that
  // window queued up and then executed in a burst when lockout expired.
  if (Serial.available()) {
    String cmd = Serial.readStringUntil('\n');
    cmd.trim();
    handleCommand(cmd);
  }

  if (isLockedOut) {
    unsigned long elapsed = (now - lockoutStart) / 1000;
    int remaining = 60 - (int)elapsed;
    if (remaining <= 0) {
      isLockedOut = false;
      failCount = 0;
      allLedsOff();
      showScreen("GARAGE SYSTEM", "READY", "");
      Serial.println("LOCKOUT_CLEARED");
    } else {
      digitalWrite(RED_PIN, (now / 500) % 2);
      // [FIX 2] was an unconditional showScreen() on EVERY loop pass.
      // display.display() pushes 1024 bytes over I2C; at the old 100 kHz
      // that is ~90 ms, so the bus ran at 100% duty for 60 s straight.
      // showScreen() now returns immediately unless the text changed, so
      // this redraws once per second instead of ten times per second.
      showScreen("!! LOCKOUT !!", String(remaining) + "s remaining", "");
    }
    return;
  }

  if (isScanning) {
    if (now - yellowBlinkTimer > 400) {
      yellowBlinkState = !yellowBlinkState;
      digitalWrite(YELLOW_PIN, yellowBlinkState);
      yellowBlinkTimer = now;
    }
  }

  if (now - lastHeartbeat > 10000) {
    Serial.println("HEARTBEAT");
    lastHeartbeat = now;
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
    allLedsOff();
    digitalWrite(GREEN_PIN, HIGH);
    myServo.write(90);
    beep(2, 100, 100);                      // [FIX 4] was blocking
    showScreen("ACCESS GRANTED", "Welcome!", "Gate Opening...");

    // [FIX 4] was: delay(5000); myServo.write(0); ...
    // 5.3 s during which Serial was never read. The Pi's CLOSE (and any
    // SCANNING or RESET) sat in the RX buffer and fired late.
    //
    // This is now a FAILSAFE, not normal operation: the Pi owns the door
    // and should send CLOSE. If the Pi dies mid-cycle the gate still
    // shuts, which is the behaviour you want from a crashed controller.
    gateCloseAt = millis() + 6000;          // 6 s > the Pi's 5 s, so the
                                            // Pi closes first in normal
                                            // operation and the race is
                                            // gone

  } else if (cmd == "DENIED") {
    isScanning = false;
    failCount++;
    allLedsOff();
    digitalWrite(RED_PIN, HIGH);
    beep(1, 1000, 0);                       // [FIX 5] was blocking
    showScreen("ACCESS DENIED", "Unauthorized!",
               "Attempt: " + String(failCount) + "/3");
    deniedClearAt = millis() + 3000;        // [FIX 5] was delay(3000)

  } else if (cmd == "LOCKOUT") {
    triggerLockout();

  } else if (cmd == "OPEN") {
    myServo.write(90);
    gateCloseAt = 0;                        // manual open stays open
    Serial.println("GATE_OPENED");

  } else if (cmd == "CLOSE") {
    myServo.write(0);
    gateCloseAt = 0;                        // the Pi closed it; cancel the
                                            // failsafe so it cannot fire
                                            // twice
    digitalWrite(GREEN_PIN, LOW);
    showScreen("GARAGE SYSTEM", "READY", "");
    Serial.println("GATE_CLOSED");

  } else if (cmd == "RESET") {
    isScanning = false;
    isLockedOut = false;
    failCount = 0;
    gateCloseAt = 0;
    deniedClearAt = 0;
    beepsLeft = 0;
    digitalWrite(BUZZER_PIN, LOW);
    allLedsOff();
    myServo.write(0);
    showScreen("GARAGE SYSTEM", "READY", "");
    Serial.println("RESET_OK");
  }
}

void triggerLockout() {
  isLockedOut = true;
  isScanning = false;
  lockoutStart = millis();
  deniedClearAt = 0;
  allLedsOff();
  beep(5, 200, 100);        // [FIX 6] was 5 x (beep(200)+delay(100)),
                            // i.e. 1.5 s of blocking inside a handler
  Serial.println("LOCKOUT_ACTIVE");
}

// ---- [FIX 4/5/6] non-blocking buzzer --------------------------------
void beep(int times, int onMs, int offMs) {
  beepsLeft = times; beepOnMs = onMs; beepOffMs = offMs;
  beepPhase = millis(); beepHigh = true;
  digitalWrite(BUZZER_PIN, HIGH);
}

void serviceBeep() {
  if (beepsLeft <= 0) return;
  unsigned long now = millis();
  if (beepHigh && now - beepPhase >= (unsigned long)beepOnMs) {
    digitalWrite(BUZZER_PIN, LOW);
    beepHigh = false; beepPhase = now;
    beepsLeft--;
    if (beepsLeft <= 0) beepsLeft = 0;
  } else if (!beepHigh && beepsLeft > 0 &&
             now - beepPhase >= (unsigned long)beepOffMs) {
    digitalWrite(BUZZER_PIN, HIGH);
    beepHigh = true; beepPhase = now;
  }
}

void allLedsOff() {
  digitalWrite(RED_PIN, LOW);
  digitalWrite(YELLOW_PIN, LOW);
  digitalWrite(GREEN_PIN, LOW);
}

// ---- [FIX 2] I2C bus recovery ---------------------------------------
// If a transfer was cut mid-byte -- which is exactly what a replug does
// -- the slave can hold SDA low forever and every later begin() fails.
// Clocking SCL until it lets go, then issuing a STOP, frees the bus.
void i2cRecover() {
  pinMode(I2C_SDA, INPUT_PULLUP);
  pinMode(I2C_SCL, OUTPUT_OPEN_DRAIN);
  digitalWrite(I2C_SCL, HIGH);
  delayMicroseconds(5);
  for (int i = 0; i < 9 && digitalRead(I2C_SDA) == LOW; i++) {
    digitalWrite(I2C_SCL, LOW);  delayMicroseconds(5);
    digitalWrite(I2C_SCL, HIGH); delayMicroseconds(5);
  }
  pinMode(I2C_SDA, OUTPUT_OPEN_DRAIN);
  digitalWrite(I2C_SDA, LOW);  delayMicroseconds(5);
  digitalWrite(I2C_SCL, HIGH); delayMicroseconds(5);
  digitalWrite(I2C_SDA, HIGH); delayMicroseconds(5);
}

void showScreen(String line1, String line2, String line3) {
  // [FIX 2] skip the transfer entirely when nothing changed
  if (line1 == scr1 && line2 == scr2 && line3 == scr3) return;
  scr1 = line1; scr2 = line2; scr3 = line3;
  if (!hasDisplay) return;                  // [FIX 1]

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
