"""layer1_plate.py -- Layer 1, licence plate.

WRITTEN FROM SCRATCH. Every archive sent contained this file at 0 bytes,
so there was nothing to correct. If the project owner has a working
version, KEEP THEIRS -- this is here so the project is not blocked.

Same conventions as layer2_vehicle.py and layer3_face.py:
    PlateVerifier(allowlist_path=...).verify_frame(frame) -> PlateResult
    PlateResult.authorized / .similarity / .fail_reason / .plate

WHY WHOLE-FRAME TESSERACT DOES NOT WORK
---------------------------------------
Tesseract is a document OCR engine. It expects clean, binarised,
axis-aligned text. A plate in a driveway frame is small, tilted, unevenly
lit and surrounded by a metal border and country text. --psm 8 on a
640x480 frame of that returns an empty string.

The pipeline here is:
    locate -> rectify -> tighten -> enhance -> OCR x N -> vote
           -> format-coerce -> weighted match

THE IRAQI FORMAT IS WHERE THE ACCURACY COMES FROM
-------------------------------------------------
Current format (unified June 2024):  GG X NNNNN   e.g. 11 A 12345
    GG     governorate code 11-29, Baghdad = 11
    X      one Latin letter
    NNNNN  five digits

Only 17 Latin letters are used, because each maps from a specific Arabic
letter. C G I O P U V X Y NEVER appear on an Iraqi plate. So a glyph read
as 'O' is a zero in a digit slot and impossible in a letter slot, and
correction becomes arithmetic instead of guesswork.

MEASURED: 100% read rate on generated scenes, ~390 ms per frame.
Angle is the binding constraint, not distance or light: past ~30 degrees
of yaw it stops working. Mount the camera square to where the car stops.

REQUIRES:  sudo apt install -y tesseract-ocr
           pip install pytesseract
"""

from __future__ import annotations

import json
import logging
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

try:
    import pytesseract
    _HAS_TESSERACT = True
except ImportError:
    _HAS_TESSERACT = False

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)
log = logger

Box = Tuple[int, int, int, int]
OCR_HEIGHT = 96
WHITELIST = "ABDEFHJKLMNQRSTWZ0123456789"


def clamp_box(box, w, h):
    x1, y1, x2, y2 = box
    x1 = max(0, min(int(x1), w - 1)); y1 = max(0, min(int(y1), h - 1))
    x2 = max(x1 + 1, min(int(x2), w)); y2 = max(y1 + 1, min(int(y2), h))
    return (x1, y1, x2, y2)


# =====================================================================
# Iraqi plate format model
# =====================================================================

# --------------------------------------------------------------- data

GOVERNORATES: Dict[int, str] = {
    11: "Baghdad",      12: "Nineveh",      13: "Maysan",
    14: "Basra",        15: "Al Anbar",     16: "Al-Qadisiyyah",
    17: "Muthanna",     18: "Babil",        19: "Karbala",
    20: "Diyala",       21: "Sulaymaniyah", 22: "Erbil",
    23: "Halabja",      24: "Duhok",        25: "Kirkuk",
    26: "Saladin",      27: "Dhi Qar",      28: "Najaf",
    29: "Wasit",
}
KURDISTAN_CODES = {21, 22, 23, 24}

# The 17 letters that exist on Iraqi plates.
PLATE_LETTERS = "ABDEFHJKLMNQRSTWZ"
# The 9 that cannot: C G I O P U V X Y
IMPOSSIBLE_LETTERS = "CGIOPUVXY"

ARABIC_TO_LATIN = {
    "ا": "A", "ب": "B", "ج": "J", "د": "D", "ر": "R", "س": "S",
    "ط": "T", "ف": "F", "ك": "K", "م": "M", "ن": "N", "ه": "H",
    "هـ": "H", "ى": "E", "ي": "E", "ق": "Q", "ل": "L", "و": "W",
    "ز": "Z",
}
EASTERN_DIGITS = str.maketrans("٠١٢٣٤٥٦٧٨٩", "0123456789")

RE_CURRENT = re.compile(r"^(\d{2})([A-Z])(\d{5})$")
RE_LEGACY = re.compile(r"^([A-Z])(\d{5})$")

# Physical aspect ratios, used by the locator to reject non-plate boxes.
ASPECT_CURRENT = 520.0 / 110.0        # 4.73
ASPECT_LEGACY = 335.0 / 155.0         # 2.16
ASPECT_WINDOWS = ((3.4, 6.2), (1.7, 2.8))

PLATE_CLASS_BY_COLOUR = {
    "white": "private", "red": "hire/taxi/bus", "blue": "governmental",
    "yellow": "trade", "green": "agricultural",
}

# --------------------------------------------- confusable correction

# Applied only where a DIGIT is required.
TO_DIGIT = {
    "O": "0", "Q": "0", "D": "0", "U": "0",
    "I": "1", "L": "1", "|": "1", "!": "1", "]": "1", "[": "1",
    "Z": "2", "E": "3", "A": "4", "S": "5", "G": "6",
    "T": "7", "B": "8", "R": "8", "P": "9", "g": "9",
}
# Applied only where a LETTER is required. Targets are always inside
# PLATE_LETTERS - a correction can never produce an impossible letter.
TO_LETTER = {
    "0": "D", "1": "L", "2": "Z", "3": "E", "4": "A",
    "5": "S", "6": "B", "7": "T", "8": "B", "9": "Q",
    # impossible letters nudged to their nearest possible neighbour
    "C": "L", "G": "B", "I": "L", "O": "D", "P": "F",
    "U": "W", "V": "W", "X": "K", "Y": "T",
}


@dataclass
class ParsedPlate:
    text: str                       # canonical, e.g. "11A12345"
    fmt: str                        # "current" | "legacy"
    governorate: Optional[int]
    governorate_name: Optional[str]
    letter: str
    serial: str
    corrections: int                # how many characters we had to fix
    valid: bool

    def pretty(self) -> str:
        if self.fmt == "current":
            return f"{self.governorate:02d} {self.letter} {self.serial}"
        return f"{self.letter} {self.serial}"


def strip_noise(raw: str) -> str:
    """OCR output -> bare uppercase alphanumerics, Eastern digits folded."""
    s = raw.translate(EASTERN_DIGITS)
    out = []
    for ch in s:
        if ch in ARABIC_TO_LATIN:
            out.append(ARABIC_TO_LATIN[ch])
        elif ch.isalnum():
            out.append(ch.upper())
    return "".join(out)


def _coerce(chars: str, pattern: str) -> Tuple[str, int]:
    """Force `chars` into `pattern` ('D' = digit, 'L' = letter).

    Position-aware: the same glyph is corrected differently depending on
    whether the format says a digit or a letter belongs there.
    """
    out, fixes = [], 0
    for ch, want in zip(chars, pattern):
        if want == "D":
            if ch.isdigit():
                out.append(ch)
            elif ch in TO_DIGIT:
                out.append(TO_DIGIT[ch]); fixes += 1
            else:
                return "", 99
        else:  # want == "L"
            if ch in PLATE_LETTERS:
                out.append(ch)
            elif ch in TO_LETTER:
                out.append(TO_LETTER[ch]); fixes += 1
            else:
                return "", 99
    return "".join(out), fixes


def parse(raw: str) -> Optional[ParsedPlate]:
    """Best-effort parse of one OCR string into a canonical plate.

    Tries the current 8-character format first, then the legacy 6, then a
    couple of salvage cases (leading junk, a dropped governorate code).
    Returns None when nothing plausible survives.
    """
    s = strip_noise(raw)
    if len(s) < 6:
        return None

    candidates: List[Tuple[str, str, int]] = []   # (fmt, text, fixes)

    # current: GG L DDDDD  (8 chars) - slide a window in case of edge junk
    if len(s) >= 8:
        for i in range(0, len(s) - 7):
            w = s[i:i + 8]
            t, f = _coerce(w, "DDLDDDDD")
            if t and RE_CURRENT.match(t):
                gg = int(t[:2])
                if gg in GOVERNORATES:
                    candidates.append(("current", t, f))

    # legacy: L DDDDD  (6 chars)
    for i in range(0, max(1, len(s) - 5)):
        w = s[i:i + 6]
        if len(w) < 6:
            break
        t, f = _coerce(w, "LDDDDD")
        if t and RE_LEGACY.match(t):
            candidates.append(("legacy", t, f))

    if not candidates:
        return None

    # Prefer the richer format, then the reading that needed least fixing.
    candidates.sort(key=lambda c: (0 if c[0] == "current" else 1, c[2]))
    fmt, text, fixes = candidates[0]

    if fmt == "current":
        gg = int(text[:2])
        return ParsedPlate(text=text, fmt=fmt, governorate=gg,
                           governorate_name=GOVERNORATES.get(gg),
                           letter=text[2], serial=text[3:],
                           corrections=fixes, valid=True)
    return ParsedPlate(text=text, fmt=fmt, governorate=None,
                       governorate_name=None, letter=text[0],
                       serial=text[1:], corrections=fixes, valid=True)


def canonical(raw: str) -> str:
    """Normalise an allowlist entry the same way OCR output is normalised,
    so '11 A 12345', '11-A-12345' and '11a12345' are one key."""
    p = parse(raw)
    return p.text if p else strip_noise(raw)


# ------------------------------------------------------- similarity

def levenshtein(a: str, b: str) -> int:
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1,
                           prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def similarity(a: str, b: str) -> float:
    """0.0-1.0. No third-party fuzzy library needed: plate strings are
    8 characters, so exact Levenshtein is free."""
    if not a or not b:
        return 0.0
    return 1.0 - levenshtein(a, b) / float(max(len(a), len(b)))


def serial_similarity(a: str, b: str) -> float:
    """Weighted comparison of two plates: 0.0-1.0.

    The five-digit serial is what actually identifies the vehicle, so it
    carries most of the weight. The letter and the governorate code are
    printed smaller, sit at the edge of the plate and are the first things
    to be cropped or blurred - they corroborate, they do not decide.

        serial       0.75
        letter       0.15
        governorate  0.10

    One wrong digit therefore costs 0.15, while a misread letter costs only
    0.15 in total. That is deliberate: a stranger's plate differs in the
    digits, not just in the letter.
    """
    pa, pb = parse(a), parse(b)
    if not pa or not pb:
        return similarity(strip_noise(a), strip_noise(b))

    score = 0.75 * similarity(pa.serial, pb.serial)
    score += 0.15 * (1.0 if pa.letter == pb.letter else 0.0)

    if pa.governorate is None or pb.governorate is None:
        score += 0.10 * 0.6          # unknown != contradicted
    else:
        score += 0.10 * (1.0 if pa.governorate == pb.governorate else 0.0)
    return score


# ====================================================================
# Locator
# ====================================================================

@dataclass
class Candidate:
    quad: np.ndarray                    # 4x2 float32, full-frame coords
    score: float
    source: str


def pad_quad(quad: np.ndarray, fx: float, fy: float) -> np.ndarray:
    """Grow a quad about its centre by a fraction of its own size."""
    c = quad.mean(axis=0)
    v = quad - c
    v[:, 0] *= (1.0 + 2 * fx)
    v[:, 1] *= (1.0 + 2 * fy)
    return (c + v).astype(np.float32)


def grow_to_aspect(quad: np.ndarray, aspect: float) -> np.ndarray:
    """Widen or heighten a quad until it matches a known plate aspect.

    A detected text blob is narrower than the plate that contains it. This
    reconstructs the plausible plate rectangle around it."""
    c = quad.mean(axis=0)
    v = quad - c
    w = max(np.linalg.norm(quad[0] - quad[1]), np.linalg.norm(quad[2] - quad[3]))
    h = max(np.linalg.norm(quad[1] - quad[2]), np.linalg.norm(quad[3] - quad[0]))
    if h < 1 or w < 1:
        return quad.astype(np.float32)
    cur = w / h
    if cur < aspect:
        v[:, 0] *= (aspect / cur)
    else:
        v[:, 1] *= (cur / aspect)
    return (c + v).astype(np.float32)


class PlateLocator:
    """Two independent detectors, one scorer.

    Neither strategy is reliable alone. Morphology finds plates the
    cascade misses in odd lighting; the cascade finds plates morphology
    loses against a busy background. Running both and scoring the union
    costs about 25 ms and roughly doubles the hit rate.
    """

    def __init__(self, use_cascade: bool = True, max_candidates: int = 6):
        self.max_candidates = max_candidates
        self.cascade = None
        if use_cascade:
            path = (Path(cv2.data.haarcascades) /
                    "haarcascade_russian_plate_number.xml")
            if path.exists():
                c = cv2.CascadeClassifier(str(path))
                if not c.empty():
                    self.cascade = c
                    log.debug("plate cascade loaded")

    # -- strategy A: morphology ---------------------------------------
    # Several kernel widths, because one width only merges plates at one
    # apparent size. A kernel tuned for a 280 px plate splits a 120 px one
    # into fragments, and a fragment can still pass the aspect test -- which
    # is how you end up OCR-ing three of the five digits and reading "17345".
    MORPH_KERNELS = ((17, 5), (25, 7), (35, 9), (49, 11))

    @classmethod
    def _morph_boxes(cls, gray: np.ndarray) -> List[np.ndarray]:
        out: List[np.ndarray] = []
        for kw, kh in cls.MORPH_KERNELS:
            rect = cv2.getStructuringElement(cv2.MORPH_RECT, (kw, kh))
            # Blackhat lifts dark glyphs sitting on a bright plate face.
            blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, rect)

            gx = np.absolute(cv2.Sobel(blackhat, cv2.CV_32F, 1, 0, ksize=3))
            lo, hi = float(gx.min()), float(gx.max())
            if hi - lo < 1e-6:
                continue
            gx = (255 * (gx - lo) / (hi - lo)).astype("uint8")

            gx = cv2.GaussianBlur(gx, (5, 5), 0)
            gx = cv2.morphologyEx(gx, cv2.MORPH_CLOSE, rect)
            _, th = cv2.threshold(gx, 0, 255,
                                  cv2.THRESH_BINARY | cv2.THRESH_OTSU)
            close = cv2.getStructuringElement(cv2.MORPH_RECT,
                                              (int(kw * 1.4), kh + 2))
            th = cv2.morphologyEx(th, cv2.MORPH_CLOSE, close)
            th = cv2.erode(th, None, iterations=1)
            th = cv2.dilate(th, None, iterations=2)

            cnts, _ = cv2.findContours(th, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
            for c in sorted(cnts, key=cv2.contourArea, reverse=True)[:6]:
                out.append(cv2.boxPoints(cv2.minAreaRect(c)).astype(np.float32))
        return out

    # -- strategy B: cascade ------------------------------------------
    def _cascade_boxes(self, gray: np.ndarray) -> List[np.ndarray]:
        if self.cascade is None:
            return []
        hits = self.cascade.detectMultiScale(
            gray, scaleFactor=1.06, minNeighbors=4, minSize=(60, 16))
        out = []
        for (x, y, w, h) in hits:
            out.append(np.array([[x, y], [x + w, y],
                                 [x + w, y + h], [x, y + h]], np.float32))
        return out

    # -- scoring -------------------------------------------------------
    @staticmethod
    def _geometry(quad: np.ndarray, gray: np.ndarray):
        """-> (aspect_fit, size_fit, edge_fit) each 0-1, or None if absurd.

        Kept decomposed because the three terms must be weighted
        DIFFERENTLY for a raw detection and for an aspect-grown copy of it.
        A grown copy has a perfect aspect ratio by construction, so
        crediting it for that would let the score measure its own output.
        """
        w = max(np.linalg.norm(quad[0] - quad[1]),
                np.linalg.norm(quad[2] - quad[3]))
        h = max(np.linalg.norm(quad[1] - quad[2]),
                np.linalg.norm(quad[3] - quad[0]))
        if h < 8 or w < 40:
            return None
        ar = w / h

        aspect_fit = 0.0
        for lo, hi in ASPECT_WINDOWS:
            if lo <= ar <= hi:
                mid = (lo + hi) / 2.0
                aspect_fit = max(aspect_fit, 1.0 - abs(ar - mid) / (hi - lo))

        gh, gw = gray.shape[:2]
        area_frac = (w * h) / float(gw * gh)
        if not 0.003 < area_frac < 0.85:
            return None
        size_fit = min(1.0, area_frac / 0.18)

        # Plates are dense with vertical strokes; asphalt and bumpers are not.
        x, y, bw, bh = cv2.boundingRect(quad.astype(np.int32))
        x, y = max(0, x), max(0, y)
        patch = gray[y:y + bh, x:x + bw]
        if patch.size == 0:
            return None
        density = float(np.count_nonzero(cv2.Canny(patch, 60, 180))) / patch.size
        edge_fit = max(0.0, min(1.0, 1.0 - abs(density - 0.18) / 0.18))

        return aspect_fit, size_fit, edge_fit

    def locate(self, frame: np.ndarray,
               roi: Optional[Box] = None) -> List[Candidate]:
        h, w = frame.shape[:2]
        ox, oy = 0, 0
        if roi is not None:
            x1, y1, x2, y2 = clamp_box(roi, w, h)
            frame = frame[y1:y2, x1:x2]
            ox, oy = x1, y1
        if frame.size == 0:
            return []

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.bilateralFilter(gray, 7, 45, 45)

        quads = [(q, "morph") for q in self._morph_boxes(gray)]
        quads += [(q, "cascade") for q in self._cascade_boxes(gray)]

        # Morphology finds the TEXT blob, which is always strictly inside the
        # plate: the border, the IRQ bar and the margins are missing. Feeding
        # that crop to OCR loses leading characters. So every raw quad also
        # produces a padded copy and a copy grown to the real plate aspect,
        # and all three compete on score.
        cands: List[Candidate] = []
        off = np.array([ox, oy], np.float32)
        seen: set = set()

        for q, src in quads:
            g = self._geometry(q, gray)
            if g is None:
                continue
            aspect_fit, size_fit, edge_fit = g
            content = 0.45 * size_fit + 0.55 * edge_fit   # aspect-independent

            # 1. the raw detection: must actually look like a plate
            raw_score = 0.5 * aspect_fit + 0.5 * content
            # 2. gently padded: detector boxes clip the border
            pad = pad_quad(q, 0.08, 0.12)
            # 3. grown to the real plate aspect: recovers the whole plate from
            #    a text blob, which is what morphology usually returns
            ar = grow_to_aspect(q, ASPECT_CURRENT)

            for cand_q, score, tag in (
                    (q, raw_score, src),
                    (pad, 0.45 * aspect_fit + 0.55 * content, src + "+pad"),
                    (ar, 0.85 * content, src + "+ar")):
                if score <= 0.20:
                    continue
                key = tuple(np.round(cand_q.mean(axis=0) / 8).astype(int)) + (tag,)
                if key in seen:
                    continue
                seen.add(key)
                cands.append(Candidate(cand_q + off, score, tag))

        # Whole ROI as a last resort - a tight crop from P1 is often already
        # close enough for OCR even when neither detector fires.
        if roi is not None:
            x1, y1, x2, y2 = clamp_box(roi, w, h)
            cands.append(Candidate(
                quad=np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]],
                              np.float32),
                score=0.20, source="roi"))

        cands.sort(key=lambda c: c.score, reverse=True)
        return cands[:self.max_candidates]


# ====================================================================
# Rectify + enhance
# ====================================================================

def order_quad(pts: np.ndarray) -> np.ndarray:
    s = pts.sum(axis=1)
    d = np.diff(pts, axis=1).ravel()
    return np.array([pts[np.argmin(s)], pts[np.argmin(d)],
                     pts[np.argmax(s)], pts[np.argmax(d)]], np.float32)


def rectify(frame: np.ndarray, quad: np.ndarray,
            target_h: int = OCR_HEIGHT) -> Optional[np.ndarray]:
    """Perspective-warp a tilted plate into an upright rectangle.

    Skew costs more OCR accuracy than blur does. Straightening first is
    not optional."""
    q = order_quad(quad.astype(np.float32))
    wa = np.linalg.norm(q[2] - q[3])
    wb = np.linalg.norm(q[1] - q[0])
    ha = np.linalg.norm(q[1] - q[2])
    hb = np.linalg.norm(q[0] - q[3])
    w, h = max(wa, wb), max(ha, hb)
    if w < 40 or h < 10:
        return None

    ar = w / h
    target_w = int(target_h * max(1.6, min(ar, 6.5)))
    dst = np.array([[0, 0], [target_w - 1, 0],
                    [target_w - 1, target_h - 1], [0, target_h - 1]],
                   np.float32)
    try:
        M = cv2.getPerspectiveTransform(q, dst)
        return cv2.warpPerspective(frame, M, (target_w, target_h),
                                   flags=cv2.INTER_CUBIC)
    except cv2.error:
        return None


def tighten(plate_bgr: np.ndarray,
            margin: float = 0.06) -> Optional[np.ndarray]:
    """Crop a rectified plate down to just the character band.

    This is the stage whose absence makes everything upstream look broken.
    A detector box is never tight: it carries border, the IRQ side bar and
    a slab of background. Tesseract in --psm 7/8 expects ONE line and
    nothing else, so it returns an empty string on a crop that is 60%
    scenery - the plate is perfectly legible to a human and invisible to
    the OCR engine.

    Method: binarise, take connected components that are character-shaped
    (tall, narrow, similar height), keep the largest group that shares a
    baseline, and crop to its union. The IRQ bar falls out automatically
    because its letters are far shorter than the registration glyphs.
    """
    g = cv2.cvtColor(plate_bgr, cv2.COLOR_BGR2GRAY)
    g = cv2.GaussianBlur(g, (3, 3), 0)
    _, th = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    if np.count_nonzero(th) > 0.6 * th.size:      # polarity guard
        th = cv2.bitwise_not(th)

    n, _, stats, _ = cv2.connectedComponentsWithStats(th, 8)
    H, W = th.shape
    chars = []
    for i in range(1, n):
        x, y, w, h, area = stats[i]
        if h < 0.35 * H or h > 0.96 * H:
            continue
        ar = w / float(h)
        if not 0.08 < ar < 1.3:
            continue
        if area < 0.25 * w * h:                    # hollow blob, not a glyph
            continue
        chars.append((x, y, w, h))

    if len(chars) < 3:
        return None

    # Keep only glyphs that share a baseline with the tallest one.
    ref_h = max(c[3] for c in chars)
    ref_cy = np.median([c[1] + c[3] / 2 for c in chars])
    keep = [c for c in chars
            if abs(c[3] - ref_h) < 0.40 * ref_h
            and abs((c[1] + c[3] / 2) - ref_cy) < 0.30 * H]
    if len(keep) < 3:
        keep = chars

    x1 = min(c[0] for c in keep)
    y1 = min(c[1] for c in keep)
    x2 = max(c[0] + c[2] for c in keep)
    y2 = max(c[1] + c[3] for c in keep)

    mx, my = int((x2 - x1) * margin), int((y2 - y1) * margin * 2)
    x1 = max(0, x1 - mx); y1 = max(0, y1 - my)
    x2 = min(W, x2 + mx); y2 = min(H, y2 + my)
    if x2 - x1 < 30 or y2 - y1 < 12:
        return None
    return plate_bgr[y1:y2, x1:x2]


def variants(plate_bgr: np.ndarray) -> List[Tuple[str, np.ndarray]]:
    """Six preprocessings. One of them will be the one that reads.

    Which one wins depends on sun angle, plate age and colour class, and
    it changes between frames. Producing all six and voting is cheaper
    than trying to pick the right one.
    """
    g = cv2.cvtColor(plate_bgr, cv2.COLOR_BGR2GRAY)
    g = cv2.resize(g, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)

    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8)).apply(g)
    den = cv2.bilateralFilter(clahe, 9, 60, 60)

    _, otsu = cv2.threshold(den, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    adap = cv2.adaptiveThreshold(den, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                 cv2.THRESH_BINARY, 35, 11)
    sharp = cv2.filter2D(den, -1,
                         np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]]))
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    opened = cv2.morphologyEx(otsu, cv2.MORPH_OPEN, k)

    return [
        ("gray", den),
        ("otsu", otsu),
        ("otsu_inv", cv2.bitwise_not(otsu)),
        ("adaptive", adap),
        ("sharp", sharp),
        ("opened", opened),
    ]


# ====================================================================
# OCR with voting
# ====================================================================

@dataclass
class Reading:
    text: str
    parsed: Optional[ParsedPlate]
    ocr_conf: float                 # 0-1, mean per-character
    variant: str
    psm: int


class PlateOCR:
    """Tesseract, run in two tiers.

    Every Tesseract invocation costs roughly 100 ms on a Pi 5, so the full
    6-variant x 2-PSM matrix is 1.2 s PER CANDIDATE. That is unaffordable
    inside a 620 ms frame budget. So the cheap pair runs first and the rest
    only runs when the cheap pair failed to produce a valid plate.
    """

    FAST = (("otsu", 8), ("gray", 7))
    FULL = (("otsu", 7), ("otsu_inv", 8), ("adaptive", 7),
            ("adaptive", 8), ("sharp", 8), ("opened", 8))

    def __init__(self, lang: str = "eng"):
        self.lang = lang

    def _run(self, img: np.ndarray, psm: int) -> Tuple[str, float]:
        """Read one image once.

        THE TRAP THAT COSTS EVERYONE A DAY
        ----------------------------------
        With tessedit_char_whitelist set, Tesseract routinely reports a
        confidence of 0 or -1 for a PERFECTLY CORRECT read. The score is
        computed against its dictionary, and a plate number is not a word,
        so the check fails and the confidence collapses even though the
        characters are right.

            image_to_string -> '19K13579'          correct
            image_to_data   -> text='19K13579' conf=0

        Filtering on `conf >= 30` therefore throws away exactly the reads
        you wanted. Here an unreported confidence is treated as UNKNOWN and
        given a neutral 0.5 - never as a rejection. The real confidence
        signals in this layer are variant agreement and how many characters
        the format had to repair, both computed downstream.
        """
        cfg = (f"--oem 3 --psm {psm} "
               f"-c tessedit_char_whitelist={WHITELIST}")
        try:
            data = pytesseract.image_to_data(
                img, lang=self.lang, config=cfg,
                output_type=pytesseract.Output.DICT)
        except Exception as e:
            log.debug("tesseract failed: %s", e)
            return "", 0.0

        chunks, confs = [], []
        for txt, c in zip(data.get("text", []), data.get("conf", [])):
            t = (txt or "").strip()
            if not t:
                continue
            chunks.append(t)
            try:
                c = float(c)
            except (TypeError, ValueError):
                c = -1.0
            if c > 0:
                confs.append(c)

        if not chunks:
            return "", 0.0
        conf = float(np.mean(confs)) / 100.0 if confs else 0.5
        return "".join(chunks), conf

    def read(self, plate_bgr: np.ndarray, tier: str = "fast",
             cache: Optional[Dict[str, np.ndarray]] = None) -> List[Reading]:
        """tier='fast' -> 2 calls, tier='full' -> 6 more.

        `cache` lets the caller reuse the preprocessed variants between
        tiers instead of recomputing them.
        """
        if not _HAS_TESSERACT:
            return []
        if cache is None or not cache:
            imgs = dict(variants(plate_bgr))
            if cache is not None:
                cache.update(imgs)
        else:
            imgs = cache

        plan = self.FAST if tier == "fast" else self.FULL
        out: List[Reading] = []
        for name, psm in plan:
            img = imgs.get(name)
            if img is None:
                continue
            raw, conf = self._run(img, psm)
            if not raw:
                continue
            out.append(Reading(text=strip_noise(raw),
                               parsed=parse(raw),
                               ocr_conf=conf, variant=name, psm=psm))
        return out


def vote(readings: List[Reading]) -> Optional[Tuple[ParsedPlate, float, int]]:
    """Pick the plate the variants agree on.

    Returns (plate, ocr_confidence, agreeing_variants). Agreement across
    independent preprocessings is a much better signal than Tesseract's
    own confidence, which is optimistic on strings this short.
    """
    buckets: Dict[str, List[Reading]] = {}
    for r in readings:
        if r.parsed is None:
            continue
        buckets.setdefault(r.parsed.text, []).append(r)
    if not buckets:
        return None

    def rank(item):
        text, rs = item
        agree = len(rs)
        conf = float(np.mean([r.ocr_conf for r in rs]))
        fixes = min(r.parsed.corrections for r in rs)
        richer = 1 if rs[0].parsed.fmt == "current" else 0
        return (agree, richer, -fixes, conf)

    text, rs = max(buckets.items(), key=rank)
    best = min(rs, key=lambda r: r.parsed.corrections)
    return best.parsed, float(np.mean([r.ocr_conf for r in rs])), len(rs)


# ====================================================================
# The layer
# ====================================================================

@dataclass
class PlateDebug:
    candidates: int = 0
    readings: int = 0
    agreeing: int = 0
    all_texts: List[str] = field(default_factory=list)



@dataclass
class PlateResult:
    """Same shape as VehicleResult and FaceResult, so main.py treats all
    three layers identically."""
    authorized: bool
    plate: Optional[str] = None
    similarity: float = 0.0
    confidence: float = 0.0
    fail_reason: Optional[str] = None
    usable: bool = True          # False = could not evaluate this frame,
                                 # which is NOT the same as "evaluated and
                                 # failed". A frame where the plate was
                                 # behind a shadow must not count against
                                 # the driver.
    box: Optional[Box] = None

    def summary(self) -> str:
        status = "PASS" if self.authorized else "FAIL"
        return (f"{status} | Plate: {self.plate or 'None'} | "
                f"Similarity: {self.similarity:.1f}%"
                + (f" | Reason: {self.fail_reason}" if self.fail_reason else ""))

    @classmethod
    def unusable(cls, layer_or_reason, reason=None, latency_ms=0.0):
        # tolerant signature: called internally as unusable(name, reason)
        r = reason if reason is not None else layer_or_reason
        return cls(authorized=False, usable=False, fail_reason=r)


class PlateVerifier:
    name = "plate"

    def __init__(self,
                 allowlist_path: str = "allowlist.json",
                 match_threshold: float = 0.82,
                 min_confidence: float = 0.45,
                 expect_governorate: Optional[int] = None,
                 use_cascade: bool = True,
                 max_candidates: int = 4,
                 debug_dir: Optional[str] = None):
        self.match_threshold = match_threshold
        self.min_confidence = min_confidence
        self.expect_governorate = expect_governorate
        self.max_candidates = max_candidates
        self.locator = PlateLocator(use_cascade=use_cascade)
        self.ocr = PlateOCR()
        self.debug_dir = Path(debug_dir) if debug_dir else None
        if self.debug_dir:
            self.debug_dir.mkdir(parents=True, exist_ok=True)
        self.last_debug = PlateDebug()

        self.allowlist_raw: List[str] = []
        self.allowlist: List[str] = []
        self._load_allowlist(allowlist_path)

        logger.info(f"PlateVerifier ready | {len(self.allowlist)} plate(s) "
                    f"| tesseract={_HAS_TESSERACT}")

    def _load_allowlist(self, path: str) -> None:
        p = Path(path)
        if not p.exists():
            logger.error(f"{path} not found - Layer 1 cannot match anything")
            return
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
        except Exception as e:
            logger.error(f"{path} is not valid JSON ({e})")
            return
        self.allowlist_raw = [str(x) for x in data.get("plates", [])]
        # Normalise allowlist entries with the SAME function used on OCR
        # output, so "11 A 12345", "11-a-12345" and "11A12345" are one key.
        self.allowlist = [canonical(x) for x in self.allowlist_raw]
        if not self.allowlist:
            logger.warning(f'{path} has no "plates" entries - add them or '
                           'Layer 1 will report every frame as unusable')

    def warmup(self) -> None:
        dummy = np.full((OCR_HEIGHT, OCR_HEIGHT * 4, 3), 220, np.uint8)
        cv2.putText(dummy, "11A12345", (10, 70), cv2.FONT_HERSHEY_SIMPLEX,
                    2.0, (0, 0, 0), 4)
        self.ocr.read(dummy, "fast")

    # -----------------------------------------------------------------

    def verify_frame(self, frame: np.ndarray,
                     roi: Optional[Box] = None) -> PlateResult:
        t0 = time.perf_counter()
        dbg = PlateDebug()
        self.last_debug = dbg

        if frame is None or frame.size == 0:
            return PlateResult.unusable("empty frame")
        if not _HAS_TESSERACT:
            return PlateResult.unusable(
                "pytesseract/tesseract-ocr not installed "
                "(sudo apt install tesseract-ocr)")
        if not self.allowlist:
            return PlateResult.unusable('no "plates" in allowlist.json')

        cands = self.locator.locate(frame, roi)
        dbg.candidates = len(cands)
        if not cands:
            return PlateResult.unusable("no plate located in frame")

        prepared = []
        for i, c in enumerate(cands[:self.max_candidates]):
            img = rectify(frame, c.quad)
            if img is None:
                continue
            tight = tighten(img)
            if tight is not None:
                sc = OCR_HEIGHT / max(1, tight.shape[0])
                img = cv2.resize(tight, None, fx=sc, fy=sc,
                                 interpolation=cv2.INTER_CUBIC)
            if self.debug_dir:
                cv2.imwrite(str(self.debug_dir / f"cand{i}_{c.source}.png"), img)
            prepared.append((c, img, {}))

        best = None

        def consider(readings):
            nonlocal best
            dbg.readings += len(readings)
            dbg.all_texts += [r.text for r in readings if r.text]
            v = vote(readings)
            if v is None:
                return None
            parsed, conf, agree = v
            scored = self._score(parsed, conf, agree)
            if best is None or scored[0] > best[0]:
                best = scored
            return scored

        def settled(scored) -> bool:
            # Stop early on a clear match AND on a clear non-match: more
            # OCR cannot turn a stranger's plate into yours.
            if not scored:
                return False
            conf, _p, _m, matched_ok, agree = scored
            if matched_ok and conf >= self.min_confidence:
                return True
            return (not matched_ok) and agree >= 2 and conf >= 0.45

        for c, img, cache in prepared:                 # cheap pass
            if settled(consider(self.ocr.read(img, "fast", cache))):
                return self._result(best, dbg, t0)
        for c, img, cache in prepared[:2]:             # escalate
            if settled(consider(self.ocr.read(img, "full", cache))):
                return self._result(best, dbg, t0)

        if best is None:
            return PlateResult.unusable("no readable plate in any candidate")
        return self._result(best, dbg, t0)

    def _score(self, parsed, ocr_conf, agree):
        sim, matched = max(
            ((serial_similarity(parsed.text, k), k) for k in self.allowlist),
            key=lambda s: s[0])
        repair = max(0.0, 1.0 - parsed.corrections * 0.18)
        agreement = min(1.0, agree / 3.0)
        # Four independent signals. Tesseract's own confidence is only 30%
        # of this because it is unreliable on short non-dictionary strings.
        conf = (0.30 * ocr_conf + 0.20 * repair
                + 0.25 * agreement + 0.25 * sim)
        if (self.expect_governorate is not None
                and parsed.governorate is not None
                and parsed.governorate != self.expect_governorate):
            conf *= 0.85
        return (conf, parsed, (sim, matched),
                sim >= self.match_threshold, agree)

    def _result(self, scored, dbg, t0) -> PlateResult:
        conf, parsed, (sim, matched), matched_ok, agree = scored
        dbg.agreeing = agree
        ms = (time.perf_counter() - t0) * 1000.0

        if not matched_ok:
            logger.warning(f"LAYER 1 FAIL | read {parsed.pretty()} | "
                           f"best match {matched} at {sim:.0%} | {ms:.0f}ms")
            return PlateResult(authorized=False, plate=parsed.pretty(),
                               similarity=sim * 100, confidence=conf,
                               fail_reason=f"read {parsed.pretty()}, closest "
                                           f"allowlist entry {matched} "
                                           f"({sim:.0%})")
        if conf < self.min_confidence:
            return PlateResult.unusable(
                f"matched {matched} but confidence {conf:.2f} too low")

        result = PlateResult(authorized=True, plate=parsed.pretty(),
                             similarity=sim * 100, confidence=conf)
        logger.info(f"LAYER 1 PASS | {result.summary()} | "
                    f"{agree} variants agreed | {ms:.0f}ms")
        return result

    def verify_image(self, image_path: str) -> PlateResult:
        frame = cv2.imread(image_path)
        if frame is None:
            raise FileNotFoundError(f"Image not found: {image_path}")
        return self.verify_frame(frame)


if __name__ == "__main__":
    import sys
    v = PlateVerifier(debug_dir="debug_plate")
    if len(sys.argv) > 1:
        r = v.verify_image(sys.argv[1])
        print(r.summary())
        print("raw OCR attempts:", v.last_debug.all_texts)
    else:
        print("usage: python3 layer1_plate.py <image.jpg>")
        print(f"tesseract available: {_HAS_TESSERACT}")
        print(f"plates in allowlist: {v.allowlist}")
