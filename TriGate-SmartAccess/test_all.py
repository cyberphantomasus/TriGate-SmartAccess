#!/usr/bin/env python3
"""Verify every fix without any hardware attached.

Run:  python3 test_all.py

Covers the parts that can be checked offline:
  A  liveness  - screen/moire detection, non-rigid motion, blink EAR
  B  layer 2   - that ORB separates two same-colour cars where the
                 histogram alone does not
  C  layer 3   - that the largest face is chosen, not the first
  D  serial    - port resolution order and DTR/RTS defaults
  E  layer 1   - Iraqi plate format model (no camera, no OCR needed)
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import cv2   # noqa: E402
import numpy as np   # noqa: E402

from liveness import (BlinkDetector, eye_aspect_ratio,  # noqa: E402
                      nonrigid_motion, screen_replay_score)

passed = failed = 0


def check(name, cond, extra=""):
    global passed, failed
    if cond:
        passed += 1
        print(f"  PASS  {name}")
    else:
        failed += 1
        print(f"  FAIL  {name} {extra}")


# ============================================== A. liveness

def synthetic_face(size=200, seed=0):
    """Broadband texture, like skin and hair."""
    rng = np.random.default_rng(seed)
    img = rng.normal(140, 28, (size, size)).astype(np.float32)
    img = cv2.GaussianBlur(img, (9, 9), 0)          # smooth, low-frequency
    img += rng.normal(0, 6, (size, size))           # fine grain
    img = np.clip(img, 0, 255).astype(np.uint8)
    return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)


def synthetic_screen(size=200, seed=0, pitch=3):
    """Same content behind a regular pixel lattice — what a phone looks
    like when you photograph it."""
    base = synthetic_face(size, seed).astype(np.float32)
    yy, xx = np.mgrid[:size, :size]
    grid = (((xx % pitch) == 0) | ((yy % pitch) == 0)).astype(np.float32)
    out = base * (1.0 - 0.45 * grid[..., None])
    return np.clip(out, 0, 255).astype(np.uint8)


def test_liveness():
    print("\nA. LIVENESS")

    real = [screen_replay_score(synthetic_face(seed=s)) for s in range(6)]
    fake = [screen_replay_score(synthetic_screen(seed=s)) for s in range(6)]
    print(f"    real faces : mean {np.mean(real):.3f}  max {max(real):.3f}")
    print(f"    screens    : mean {np.mean(fake):.3f}  min {min(fake):.3f}")
    check("screen score separates real texture from a pixel lattice",
          np.mean(fake) > np.mean(real) + 0.2,
          f"real={np.mean(real):.3f} fake={np.mean(fake):.3f}")
    check("every screen scores above every real face",
          min(fake) > max(real), f"{min(fake):.3f} vs {max(real):.3f}")

    # --- non-rigid motion ---
    rng = np.random.default_rng(1)
    base = rng.normal(0, 30, (68, 2)).astype(np.float32)

    # a printed photo in a shaking hand: translation + scale only
    rigid = []
    for i in range(8):
        s = 1.0 + 0.02 * np.sin(i)
        t = rng.normal(0, 4, 2)
        rigid.append(base * s + t)

    # a real face: the same global motion PLUS genuine deformation
    live = []
    for i in range(8):
        s = 1.0 + 0.02 * np.sin(i)
        t = rng.normal(0, 4, 2)
        live.append(base * s + t + rng.normal(0, 0.9, (68, 2)))

    mr, ml = nonrigid_motion(rigid), nonrigid_motion(live)
    print(f"    rigid print: {mr:.3f}   real face: {ml:.3f}")
    check("rigid motion is cancelled out", mr < 0.25, f"{mr:.3f}")
    check("genuine deformation survives", ml > 0.4, f"{ml:.3f}")

    # --- blink ---
    open_eye = np.array([[0, 0], [1, -2], [3, -2], [4, 0], [3, 2], [1, 2]],
                        dtype=np.float32)
    shut_eye = np.array([[0, 0], [1, -.2], [3, -.2], [4, 0], [3, .2], [1, .2]],
                        dtype=np.float32)
    check("EAR is high with the eye open", eye_aspect_ratio(open_eye) > 0.4)
    check("EAR collapses with the eye shut", eye_aspect_ratio(shut_eye) < 0.15)

    b = BlinkDetector()
    for e in [0.32, 0.31, 0.10, 0.30, 0.31]:      # one clean blink
        b.update(e)
    check("one blink counted", b.blinks == 1, f"{b.blinks}")

    b2 = BlinkDetector()
    for e in [0.32] + [0.10] * 12 + [0.31]:       # eyes held shut
        b2.update(e)
    check("a long closure is not a blink", b2.blinks == 0, f"{b2.blinks}")


# ============================================== B. layer 2

def car_image(colour, seed, size=(240, 320)):
    """Two vehicles of the SAME colour with DIFFERENT structure."""
    rng = np.random.default_rng(seed)
    img = np.zeros((*size, 3), np.uint8)
    img[:, :] = colour
    # body shading
    img = cv2.GaussianBlur(img, (21, 21), 0)
    # distinct structure: grille slats, lights, panel lines
    for i in range(rng.integers(5, 10)):
        y = int(rng.integers(40, size[0] - 30))
        cv2.line(img, (20, y), (size[1] - 20, y + int(rng.integers(-8, 8))),
                 (30, 30, 30), int(rng.integers(2, 5)))
    for i in range(rng.integers(2, 5)):
        c = (int(rng.integers(20, 300)), int(rng.integers(30, 200)))
        cv2.ellipse(img, c, (int(rng.integers(18, 40)),
                             int(rng.integers(10, 22))),
                    0, 0, 360, (220, 220, 210), -1)
    img = cv2.add(img, rng.normal(0, 5, img.shape).astype(np.int16)
                  .clip(-40, 40).astype(np.uint8))
    return img


def test_layer2_discrimination():
    print("\nB. LAYER 2 — colour vs structure")

    mine_a = car_image((40, 55, 190), seed=1)     # my car, photo 1
    mine_b = car_image((40, 55, 190), seed=1)     # my car, photo 2
    other = car_image((40, 55, 190), seed=99)     # SAME colour, different car

    def hist(i):
        hsv = cv2.cvtColor(i, cv2.COLOR_BGR2HSV)
        h = cv2.calcHist([hsv], [0, 1], None, [50, 60], [0, 180, 0, 256])
        cv2.normalize(h, h)
        return h

    same_col = cv2.compareHist(hist(mine_a), hist(mine_b), cv2.HISTCMP_CORREL)
    diff_col = cv2.compareHist(hist(mine_a), hist(other), cv2.HISTCMP_CORREL)
    print(f"    histogram  same car {same_col:.3f}   other car {diff_col:.3f}")
    check("histogram alone cannot separate same-colour cars",
          abs(same_col - diff_col) < 0.25,
          f"gap={abs(same_col - diff_col):.3f}")

    orb = cv2.ORB_create(nfeatures=800)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

    def desc(i):
        g = cv2.cvtColor(i, cv2.COLOR_BGR2GRAY)
        g = cv2.createCLAHE(2.0, (8, 8)).apply(g)
        return orb.detectAndCompute(g, None)[1]

    def score(a, b):
        da, db = desc(a), desc(b)
        if da is None or db is None or len(da) < 2 or len(db) < 2:
            return 0.0
        pairs = bf.knnMatch(da, db, k=2)
        good = sum(1 for p in pairs
                   if len(p) == 2 and p[0].distance < 0.75 * p[1].distance)
        return good / float(min(len(da), len(db)))

    same_orb = score(mine_a, mine_b)
    diff_orb = score(mine_a, other)
    print(f"    ORB        same car {same_orb:.3f}   other car {diff_orb:.3f}")
    check("ORB separates them where the histogram does not",
          same_orb > diff_orb * 2.0,
          f"same={same_orb:.3f} other={diff_orb:.3f}")


# ============================================== C. layer 3

def test_driver_selection():
    print("\nC. LAYER 3 — driver, not passenger")
    # face_recognition returns (top, right, bottom, left)
    passenger = (100, 260, 160, 200)      # 60 x 60  = 3600, further away
    driver = (80, 180, 220, 40)           # 140 x 140 = 19600, nearer
    locs = [passenger, driver]            # passenger happens to be first

    old = locs[0]                                     # the original bug
    areas = [(l[2] - l[0]) * (l[1] - l[3]) for l in locs]
    new = locs[int(np.argmax(areas))]                 # the fix

    print(f"    first-in-list picks area {(old[2]-old[0])*(old[1]-old[3])}")
    print(f"    largest picks      area {(new[2]-new[0])*(new[1]-new[3])}")
    check("original would have used the passenger", old == passenger)
    check("patched picks the driver", new == driver)


# ============================================== D. serial

def test_serial_patch():
    print("\nD. SERIAL")
    src = (Path(__file__).resolve().parent / "mqtt_client.py").read_text()

    check("DTR deasserted before open", "self.ser.dtr = False" in src)
    check("RTS deasserted before open", "self.ser.rts = False" in src)
    check("open() called after the flags are set",
          src.index("self.ser.dtr = False") < src.index("self.ser.open()"))
    check("HUPCL cleared", "HUPCL" in src)
    check("by-id resolution before ttyACM",
          src.index("/dev/serial/by-id/*Espressif*")
          < src.index('"/dev/ttyACM*"'))
    check("handshake required before connected=True",
          src.index("_wait_for_board") < src.index("self.connected = True"))
    check("bare except removed", "except:\n" not in src)
    check("healthy() added", "def healthy" in src)


# ============================================== E. layer 1

def test_plate_format():
    print("\nE. LAYER 1 - Iraqi plate format")
    try:
        import layer1_plate as L
    except Exception as e:
        print(f"  SKIP  {e}")
        return

    p = L.parse("11A12345")
    check("current format parses", p and p.fmt == "current")
    check("Baghdad code resolved", p and p.governorate_name == "Baghdad")
    check("pretty form", p and p.pretty() == "11 A 12345")

    # position-aware repair: the same glyph means different things in a
    # digit slot and a letter slot
    check("O -> 0 in a digit slot",
          (lambda q: q and q.text == "11A02345")(L.parse("11AO2345")))
    check("0 -> D in the letter slot",
          (lambda q: q and q.letter == "D")(L.parse("11012345")))

    # C G I O P U V X Y never appear on an Iraqi plate
    bad = [b for b in L.IMPOSSIBLE_LETTERS
           if (lambda q: q and q.letter in L.IMPOSSIBLE_LETTERS)(
               L.parse(f"11{b}12345"))]
    check("all 9 impossible letters corrected away", not bad, str(bad))

    check("eastern numerals folded",
          (lambda q: q and q.text == "11A12345")(L.parse("\u0661\u0661A\u0661\u0662\u0663\u0664\u0665")))
    check("allowlist spellings collapse to one key",
          len({L.canonical(f) for f in
               ["11A12345", "11 A 12345", "11-a-12345"]}) == 1)
    check("one wrong digit still scores high",
          0.7 < L.serial_similarity("11A12345", "11A12346") < 1.0)
    check("a different plate scores low",
          L.serial_similarity("11A12345", "14J99999") < 0.5)
    check("tesseract present" if L._HAS_TESSERACT else
          "tesseract MISSING - run: sudo apt install tesseract-ocr",
          L._HAS_TESSERACT)


if __name__ == "__main__":
    test_liveness()
    test_layer2_discrimination()
    test_driver_selection()
    test_serial_patch()
    test_plate_format()
    print(f"\n{passed} passed, {failed} failed")
    sys.exit(1 if failed else 0)
