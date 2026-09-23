# TriGate — Triple-Authentication Smart Garage Access System

A physical access-control system that verifies a vehicle and its driver through three independent identity checks — license plate, vehicle identity, and face — plus a dedicated anti-spoofing pass, before it will open a garage door. Designed and built end to end: the computer vision pipeline, the embedded firmware, the serial protocol between them, and the hardware itself, on a Raspberry Pi 5 paired with an ESP32-S3.

## Why

Most consumer access systems rely on a single factor — a remote, a code, a badge. TriGate treats entry as a multi-factor identification problem: it has to agree the plate, the vehicle, and the driver all match before it even considers opening, and it screens for spoofing before granting access — closer to how a real security checkpoint reasons about a vehicle than a typical single-credential smart-home product.

## Engineering highlights

- **Short-circuit multi-layer pipeline** — cheapest checks run first (plate → vehicle → face), each layer independently gateable so a missing camera angle or absent hardware degrades gracefully instead of crashing the system.
- **Plate OCR built from scratch** — a perspective-correcting locator, multi-PSM Tesseract voting instead of trusting a single unreliable confidence score, and a locale-specific character whitelist.
- **Vehicle re-identification beyond color** — a color histogram alone can't tell two same-color cars apart; added ORB keypoint matching (grille shape, badge placement, panel lines) to identify a specific vehicle, not just its color.
- **Closed a real auth gap** — face matching originally authorized whichever face appeared first in frame; fixed to select the largest (nearest-camera) face and check only against the allowlist, closing a passenger-can-authorize hole.
- **Liveness / anti-spoofing layer** — blink detection, non-rigid motion analysis, and screen-replay scoring, so a printed photo or a phone screen held to the camera can't open the gate.
- **Root-caused a persistent embedded systems bug** — pyserial's default port-open behavior toggles DTR/RTS, which resets an ESP32-S3 (or traps it in its own bootloader) on every connection. That was the actual cause behind a long-standing "works standalone, dead over serial" failure mode.
- **I2C bus-recovery routine** — manual 9-clock SDA release and retry, resolving display corruption caused by a stuck I2C bus after a hot-unplug.
- **31-check automated test suite** that runs with zero hardware attached, plus a standalone `doctor.py` diagnostic that walks and ranks 8 possible causes of an ESP32 link failure.

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

**Status:** OLED + traffic light confirmed working off ESP32 power. Servo + buzzer wiring completes once the external 5V supply is connected — see Roadmap.

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

## Roadmap

- **Actuation hardware** — servo and buzzer wiring completes once the external 5V supply is connected; control logic and firmware are already implemented and tested in isolation.
- **Vehicle reference set** — add real photos to `vehicle_db/accent/` to bring Layer 2 fully online end-to-end.
- **Liveness calibration** — run `python3 liveness.py --calibrate` against the deployed camera to lock in final thresholds.
- **OLED stability validation** — I2C bus-recovery routine is implemented; final confirmation pending a full hardware burn-in pass.

## Contributing

- Branch per feature/fix, open a PR against `main`.
- Run `python3 test_all.py` before opening a PR — it needs no hardware.
- Keep hardware-dependent code behind a graceful fallback (see `_HAS_TESSERACT` in `layer1_plate.py`) so the test suite keeps working without hardware attached.
- Never commit `allowlist.json`, `face_db/`, or `vehicle_db/` — they contain real personal data. Use the `.example` files instead.

## Team

- Mahdi Ahmed Fouad 
- Anmar Nihad Walid
- Abdullah Sharhabil Sahij

## License

MIT — see `LICENSE`.
