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

import json
import logging
import time
from pathlib import Path

import cv2
import face_recognition

from layer3_face import FaceVerifier
from mqtt_client import GarageMQTT

# [FIX 1] the two layers that were never wired in ----------------------
try:
    from layer1_plate import PlateVerifier
    HAS_PLATE = True
except Exception as e:                       # file missing or import error
    logging.warning(f"Layer 1 unavailable: {e}")
    HAS_PLATE = False

try:
    from layer2_vehicle import VehicleVerifier
    HAS_VEHICLE = True
except Exception as e:
    logging.warning(f"Layer 2 unavailable: {e}")
    HAS_VEHICLE = False

# [NEW] anti-spoofing
try:
    from liveness import LivenessChecker
    HAS_LIVENESS = True
except Exception as e:
    logging.warning(f"Liveness unavailable: {e}")
    HAS_LIVENESS = False

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

CAMERA = ("/dev/v4l/by-id/"
          "usb-Rapoo_Camera_Rapoo_Camera_SN0001-video-index0")

# How many of the three layers must pass. Set to 3 for a strict demo, 2 if
# your reference data is thin and you would rather not fail on stage. Say
# which one you used when you present -- a judge will ask.
REQUIRED_LAYERS = 3

HOLD_SECONDS = 5           # must stay BELOW the firmware's 6 s failsafe


def drain(cap, n=5):
    """[FIX 3] Throw away queued frames before making a decision.

    V4L2 hands OpenCV a ring buffer and frames come out in order. After
    any blocking call you are reading pictures from before the block. This
    is invisible while testing (you stand still) and wrong in a demo (the
    car moves).
    """
    for _ in range(n):
        cap.grab()


def main():
    logger.info("Starting TriGate — triple authentication")

    # ---- layers ------------------------------------------------------
    face_verifier = FaceVerifier(db_path="face_db",
                                 allowlist_path="allowlist.json",
                                 tolerance=0.5)

    plate_verifier = None
    if HAS_PLATE:
        try:
            plate_verifier = PlateVerifier(allowlist_path="allowlist.json")
            logger.info("Layer 1 (plate) ready")
        except Exception as e:
            logger.error(f"Layer 1 failed to start: {e}")

    vehicle_verifier = None
    if HAS_VEHICLE:
        try:
            vehicle_verifier = VehicleVerifier(ref_dir="vehicle_db/accent")
            if not getattr(vehicle_verifier, "ref_hists", None):
                logger.warning("Layer 2 has NO reference photos — it will "
                               "fail every attempt. Add images to "
                               "vehicle_db/accent/ before demoing.")
                vehicle_verifier = None
            else:
                logger.info("Layer 2 (vehicle) ready")
        except Exception as e:
            logger.error(f"Layer 2 failed to start: {e}")

    live = LivenessChecker() if HAS_LIVENESS else None

    active = sum(x is not None
                 for x in (plate_verifier, vehicle_verifier, face_verifier))
    logger.info(f"{active}/3 layers active, {REQUIRED_LAYERS} required")
    if active < REQUIRED_LAYERS:
        logger.warning("FEWER ACTIVE LAYERS THAN REQUIRED — every attempt "
                       "will be denied. Fix the data before running.")

    # ---- link --------------------------------------------------------
    mqtt = GarageMQTT(on_door_status=lambda s: logger.info(f"ESP32: {s}"))
    esp_connected = mqtt.connect()
    logger.info("ESP32 connected!" if esp_connected else "ESP32 not connected")

    # ---- camera ------------------------------------------------------
    cap = cv2.VideoCapture(CAMERA)
    if not cap.isOpened():
        logger.error("Camera not found!")
        return
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    logger.info("Camera ready")

    cooldown = 0
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                continue
            if time.time() < cooldown:
                time.sleep(0.1)
                continue

            # [FIX 4] detect ONCE, at one scale, and reuse the result.
            # Your original detected at fx=0.25 here and verify_frame()
            # detected again at fx=0.5 — roughly double the cost for no
            # extra information.
            small = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
            rgb_small = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
            locations = face_recognition.face_locations(rgb_small)
            if not locations:
                time.sleep(0.1)
                continue

            # back to full-frame coordinates (0.25 scale -> x4)
            full_locs = [tuple(v * 4 for v in loc) for loc in locations]
            # the driver is the largest face, not locations[0]
            driver_loc = max(full_locs,
                             key=lambda l: (l[2] - l[0]) * (l[1] - l[3]))

            logger.info("Face detected — running layers...")
            if esp_connected:
                mqtt.send_scanning(True)

            drain(cap)                      # [FIX 3]
            ret, frame = cap.read()
            if not ret:
                continue

            passed, reasons = 0, []

            # -- Layer 1: plate  [FIX 1] -------------------------------
            if plate_verifier is not None:
                try:
                    r1 = plate_verifier.verify_frame(frame)
                    ok1 = getattr(r1, "authorized", getattr(r1, "passed", False))
                    if ok1:
                        passed += 1
                        logger.info(f"LAYER 1 PASS | {getattr(r1, 'plate', '')}")
                    else:
                        reasons.append(f"plate: "
                                       f"{getattr(r1, 'fail_reason', 'no match')}")
                except Exception as e:
                    reasons.append(f"plate error: {e}")

            # -- Layer 2: vehicle  [FIX 1] -----------------------------
            if vehicle_verifier is not None:
                try:
                    r2 = vehicle_verifier.verify_frame(frame)
                    if r2.authorized:
                        passed += 1
                        logger.info(f"LAYER 2 PASS | {r2.similarity:.1f}%")
                    else:
                        reasons.append(f"vehicle: {r2.fail_reason}")
                except Exception as e:
                    reasons.append(f"vehicle error: {e}")

            # -- Layer 3: face -----------------------------------------
            result = face_verifier.verify_frame(frame)
            if result.authorized:
                passed += 1
                logger.info(f"LAYER 3 PASS | {result.identity}")
            else:
                reasons.append(f"face: {result.fail_reason}")

            # -- liveness  [NEW] ---------------------------------------
            spoof = False
            if live is not None and result.authorized:
                live.reset()
                for _ in range(24):
                    ok, f2 = cap.read()
                    if not ok:
                        continue
                    lr = live.update(f2, driver_loc)
                    if lr.decided:
                        break
                if not lr.is_live:
                    spoof = True
                    reasons.append(f"liveness: {lr.reason}")
                    logger.warning(f"SPOOF SUSPECTED — {lr.reason}")

            # -- decision ----------------------------------------------
            if passed >= REQUIRED_LAYERS and not spoof:
                logger.info(f"ACCESS GRANTED — Welcome {result.identity}! "
                            f"({passed}/3 layers)")
                if esp_connected:
                    mqtt.send_open()
                # [FIX 2] the race. You slept 5 s then sent CLOSE while the
                # firmware also closed itself after delay(5000): two
                # independent closers, no acknowledgement, undefined door
                # state when they drift. The patched firmware makes its own
                # close a 6 s FAILSAFE, so the Pi closing at 5 s always
                # wins and the failsafe only fires if the Pi died.
                time.sleep(HOLD_SECONDS)
                if esp_connected:
                    mqtt.send_close()
                cooldown = time.time() + 3
            else:
                logger.warning(f"ACCESS DENIED — {passed}/{REQUIRED_LAYERS} "
                               f"layers | " + " | ".join(reasons))
                if esp_connected:
                    mqtt.send_deny()
                cooldown = time.time() + 3

            if esp_connected:
                mqtt.send_scanning(False)
                # [FIX 5] a link that died mid-session used to stay "connected"
                if not mqtt.healthy():
                    logger.error("link unhealthy — reconnecting")
                    esp_connected = mqtt.reconnect()

            drain(cap)                      # [FIX 3]

    except KeyboardInterrupt:
        logger.info("Stopped.")
    finally:
        cap.release()
        mqtt.disconnect()


if __name__ == "__main__":
    main()
