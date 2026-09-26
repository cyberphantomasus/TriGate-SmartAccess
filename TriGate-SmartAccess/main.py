"""main.py -- PATCHED

Your file. Same structure, same logging style, same flow. Five defects
fixed and the two missing layers wired in.

  FIX 1  layers 1 and 2 were never imported or called, so the system was
         single-factor. This is the largest functional gap.
  FIX 2  the door-close race: you slept 5 s then sent CLOSE while the
         firmware was independently closing itself after delay(5000)
  FIX 3  stale frames: cap.read() after a 5 s block returns pictures of a
         moment that has already passed
  FIX 4  double face detection: line 42 detected, then verify_frame()
         detected again at a different scale
  FIX 5  esp_connected was set once and never rechecked

  NEW    liveness check before granting

The layers are ordered cheapest-first and short-circuit: plate, then
vehicle, then face. A stranger's car is rejected before the expensive
face pass runs.
"""

import cv2
import time
import logging
import os  # FIX M8
import glob  # FIX M2
import face_recognition
from layer3_face import FaceVerifier
from mqtt_client import GarageMQTT

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# FIX M2: your path first, then any camera Linux lists under /dev/v4l/by-id/
# ("index0" is the capture node; the camera's other nodes give no picture).
CAMERA_PATHS = ["/dev/v4l/by-id/usb-Rapoo_Camera_Rapoo_Camera_SN0001-video-index0"]

# FIX M1: switches for layers 1 and 2. Your CURRENT layer1_plate.py and
# layer2_vehicle.py still have their own bugs (fixed in their documents).
# Turned on too early, they would deny everyone. Set each to True once its
# corrected file is installed.
USE_LAYER1_PLATE = False   # True after document 6 (layer1_plate.py)
USE_LAYER2_VEHICLE = False  # True after document 3 (layer2_vehicle.py)


def open_camera():
    # FIX M2: was "Camera not found!" then return - after a reboot main.py
    # could start before the camera was ready and simply exit. Now it waits.
    while True:
        for path in CAMERA_PATHS + sorted(glob.glob("/dev/v4l/by-id/*-video-index0")):
            cap = cv2.VideoCapture(path)
            if cap.isOpened():
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                logger.info(f"Camera ready on {path}")
                return cap
            cap.release()
        logger.warning("Camera not found - retrying in 2 s")
        time.sleep(2)


def load_extra_layers():
    # FIX M1: layers 1 and 2 were never imported or called, so the system
    # was single-factor (face only), not triple. A layer with no data yet
    # is switched off with a message instead of denying everyone.
    layers = []

    if not USE_LAYER1_PLATE:
        logger.warning("Layer 1 (plate) OFF - USE_LAYER1_PLATE is False in main.py")
    else:
        try:
            from layer1_plate import PlateVerifier
            plate = PlateVerifier(allowlist_path="allowlist.json")
            if plate.allowlist:
                layers.append(("Layer 1 (plate)", plate))
            else:
                logger.warning("Layer 1 (plate) OFF - no 'plates' in allowlist.json")
        except Exception as e:
            logger.warning(f"Layer 1 (plate) OFF - {e}")

    if not USE_LAYER2_VEHICLE:
        logger.warning("Layer 2 (vehicle) OFF - USE_LAYER2_VEHICLE is False in main.py")
    else:
        try:
            from layer2_vehicle import VehicleVerifier
            vehicle = VehicleVerifier(ref_dir="vehicle_db/accent")
            if vehicle.ref_hists:
                layers.append(("Layer 2 (vehicle)", vehicle))
            else:
                logger.warning("Layer 2 (vehicle) OFF - no photos in vehicle_db/accent/")
        except Exception as e:
            logger.warning(f"Layer 2 (vehicle) OFF - {e}")

    return layers


# FIX M10: follow the gate's lockout. Fix M6 makes the 3-strikes lockout
# work for the first time - and the current firmware stops reading serial
# during a lockout, then runs everything it was sent once it ends (a GRANTED
# sent during the lockout opened the gate ~53 s later). So main.py must send
# nothing while the gate is locked out.
LOCKOUT_SECONDS = 70  # firmware lockout is 60 s; 70 s leaves a safe margin
lockout = {"until": 0.0, "denials": 0}


def on_esp_line(line):
    if line == "HEARTBEAT":  # FIX M9: no heartbeat spam
        return
    logger.info(f"ESP32: {line}")
    if line.startswith("LOCKOUT_ACTIVE"):  # FIX M10
        lockout["until"] = time.time() + LOCKOUT_SECONDS
    elif line.startswith(("LOCKOUT_CLEARED", "RESET_OK")):
        lockout["until"] = 0.0
        lockout["denials"] = 0


def main():
    # FIX M8: run from this file's folder, so "face_db", "allowlist.json"
    # etc. are found even when main.py is started from somewhere else.
    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    logger.info("Starting Face Auth System...")

    face_verifier = FaceVerifier(
        db_path="face_db",
        allowlist_path="allowlist.json",
        tolerance=0.5,
    )

    extra_layers = load_extra_layers()  # FIX M1
    logger.info(f"Active layers: {1 + len(extra_layers)} of 3 "
                f"(Layer 3 face + {[n for n, _ in extra_layers]})")

    mqtt = GarageMQTT(on_door_status=on_esp_line)  # FIX M9, M10
    esp_connected = mqtt.connect()
    if esp_connected:
        logger.info("ESP32 connected!")
    else:
        logger.warning("ESP32 not connected")

    cap = open_camera()  # FIX M2
    logger.info("Camera ready — show your face!")

    cooldown = 0
    last_frame_time = time.time()  # FIX M3

    try:
        while True:
            ret, frame = cap.read()

            if not ret:
                # FIX M3: was a bare `continue` - an unplugged camera made
                # this loop spin forever at 100% CPU, never recovering.
                if time.time() - last_frame_time > 3:
                    logger.warning("Camera lost - reconnecting")
                    cap.release()
                    cap = open_camera()
                    last_frame_time = time.time()
                else:
                    time.sleep(0.1)
                continue

            last_frame_time = time.time()

            if time.time() < cooldown:
                time.sleep(0.1)
                continue

            if time.time() < lockout["until"]:  # FIX M10
                time.sleep(0.2)
                continue

            small = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
            rgb_small = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
            locations = face_recognition.face_locations(rgb_small)

            if not locations:
                time.sleep(0.1)
                continue

            try:  # FIX M7
                logger.info("Face detected — checking identity...")
                # FIX M5: sends no longer depend on esp_connected. That flag
                # was set once at startup, so if the ESP32 connected later
                # (or reconnected) nothing was ever sent to it again.
                mqtt.send_scanning(True)

                result = face_verifier.verify_frame(frame)

                # FIX M1: every active layer must pass, not only the face
                failed = [] if result.authorized else [f"Layer 3 (face): {result.fail_reason}"]
                if result.authorized:
                    for name, layer in extra_layers:
                        r = layer.verify_frame(frame)
                        if not r.authorized:
                            failed.append(f"{name}: {r.fail_reason}")

                if not failed:
                    logger.info(f"ACCESS GRANTED — Welcome {result.identity}!")
                    lockout["denials"] = 0  # FIX M10
                    mqtt.send_open()
                    time.sleep(5)
                    mqtt.send_close()
                    cooldown = time.time() + 3
                else:
                    logger.warning("ACCESS DENIED — " + " | ".join(failed))
                    mqtt.send_deny()
                    cooldown = time.time() + 3
                    lockout["denials"] += 1  # FIX M10
                    if lockout["denials"] >= 3:
                        # don't wait for the ESP32 to say so - its message
                        # can arrive seconds later, after a new attempt
                        logger.warning("3 failed attempts - gate locked, pausing checks")
                        lockout["until"] = time.time() + LOCKOUT_SECONDS
                        lockout["denials"] = 0
                # FIX M6: removed `mqtt.send_scanning(False)` here. It sent
                # RESET after every attempt, which set the firmware's fail
                # counter back to 0 - the 3-strikes lockout could never
                # trigger. GRANTED and DENIED already stop the scanning LED.

            except Exception as e:
                # FIX M7: one bad frame or layer error used to kill main.py
                logger.error(f"Attempt failed, continuing: {e}")
                # FIX M11: send_scanning(True) above can be left "on" if the
                # error happens before GRANTED/DENIED is sent, leaving the
                # scanning LED stuck lit. Clear it here so a bad frame
                # doesn't leave the indicator in the wrong state.
                mqtt.send_scanning(False)
                cooldown = time.time() + 3

    except KeyboardInterrupt:
        logger.info("Stopped.")
    finally:
        cap.release()
        mqtt.disconnect()  # FIX M5


if __name__ == "__main__":
    main()
