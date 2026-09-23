# TriGate — Triple-Authentication Smart Garage Door System

Checks a vehicle's license plate, its make/type, and the driver's face — plus a liveness check against photo/screen spoofing — before opening the garage door. Built on a Raspberry Pi 5 (vision models) paired with an ESP32-S3 (servo, buzzer, traffic light, OLED), with a single USB webcam.

## How it works

| Layer | Method | File |
|---|---|---|
| 1 — Plate | OCR (Tesseract, multi-variant + voting) vs. allowlist | `layer1_plate.py` |
| 2 — Vehicle | YOLOv8 detection + color histogram + ORB match | `layer2_vehicle.py` |
| 3 — Face | `face_recognition` vs. allowlist, largest face only | `layer3_face.py` |
| Liveness | Blink detection + motion + screen-replay check | `liveness.py` |

Checks run cheapest-first with short-circuit (plate → vehicle → face), then a liveness gate before the door is granted.

## Hardware

- **Raspberry Pi 5** — runs all vision models, powered by its own wall charger
- **ESP32-S3** — drives the servo, buzzer, traffic light, and OLED; talks to the Pi over USB serial
- **USB webcam** (Rapoo C260)

| Component | Wiring | Notes |
|---|---|---|
| Servo (MG90S) | Signal → GPIO4, VCC → external 5V, GND → GND | This unit: Yellow=signal, Orange=VCC, Brown=GND (non-standard — verify before assuming) |
| Buzzer | GPIO5 → resistor → transistor base; emitter → GND; collector → buzzer(−); buzzer(+) → external 5V | Passive buzzer — needs a tone/square-wave drive, plain HIGH/LOW won't make sound |
| Traffic light (4-pin) | R → GPIO6, Y → GPIO7, G → GPIO10, Common → GND | Common-cathode, active-HIGH |
| OLED (SSD1306, I2C) | SDA → GPIO8, SCL → GPIO9, VCC → ESP32 3V3, GND → GND | — |

**Status:** OLED + traffic light confirmed working off ESP32 power. Servo + buzzer wiring is on hold until an external 5V supply (batteries) is in place — see Known Issues.

## Repo structure

```
.
├── main.py                  # Entry point — camera loop + layer orchestration
├── layer1_plate.py          # Layer 1 — plate OCR
├── layer2_vehicle.py        # Layer 2 — vehicle detection
├── layer3_face.py           # Layer 3 — face recognition
├── liveness.py              # Anti-spoofing check
├── mqtt_client.py           # Serial bridge to the ESP32
├── doctor.py                # ESP32 link diagnostic — run before main.py
├── test_all.py              # Offline test suite — no hardware needed
├── allowlist.example.json   # Template — copy to allowlist.json and fill in
├── firmware/
│   └── esp32_firmware.ino   # ESP32 firmware
├── face_db/                 # Enrolled face photos (gitignored — personal data)
└── vehicle_db/
    └── accent/              # Reference vehicle photos (gitignored — personal data)
```

## Getting started

1. Clone the repo, create a venv, activate it.
2. Install dependencies:
   ```bash
   sudo apt install -y tesseract-ocr
   pip install opencv-python face_recognition ultralytics pyserial numpy
   ```
3. Add your own photos: `face_db/<name>/*.jpg` and `vehicle_db/accent/*.jpg` (8-15 shots, a few distances/angles).
4. `cp allowlist.example.json allowlist.json` and fill in real names/plate/vehicle.
5. Flash `firmware/esp32_firmware.ino` in the Arduino IDE — Board: **ESP32S3 Dev Module**, USB CDC On Boot: **Enabled**, USB Mode: **Hardware CDC and JTAG**.
6. `python3 test_all.py` — should pass with no hardware attached.
7. `python3 doctor.py` — checks the ESP32 link specifically.
8. `python3 main.py` — prints how many layers started, e.g. `3/3 layers active`.

## Known issues

- **Servo + buzzer** not wired in yet — waiting on an external 5V supply (batteries on order).
- **Layer 2 (Vehicle)** fails every attempt until `vehicle_db/accent/` has real reference photos.
- **Liveness thresholds** were tuned on generated images — run `python3 liveness.py --calibrate` on the real camera before demoing.
- **OLED corruption on ESP32 replug** — firmware now includes an I2C bus-recovery routine; believed fixed, not yet confirmed on hardware.

## Contributing

- Branch per feature/fix, open a PR against `main`.
- Run `python3 test_all.py` before opening a PR — it needs no hardware.
- Keep hardware-dependent code behind a graceful fallback (see `_HAS_TESSERACT` in `layer1_plate.py`) so the test suite keeps working without hardware attached.
- Never commit `allowlist.json`, `face_db/`, or `vehicle_db/` — they contain real personal data. Use the `.example` files instead.

## Team

- Mahdi Ahmed Fouad — project lead
- *(add teammates here)*

## License

MIT — see `LICENSE`.
