"""
layer1_plate.py - Layer 1: licence plate check
Corrected in the TriGate code review - Document 6 of 6.
Full explanation, tests and installation steps: TriGate_Doc6_layer1_plate.pdf
Same class names, functions and arguments as the original. Every change is
marked next to the lines it touches:
FIX P1  find the plate in the camera frame first, then run the OCR on it
        (the whole frame read no plate at all in testing)
FIX P2  only the 17 letters used on Iraqi plates are allowed
FIX P3  threshold 85 instead of 75 - 75 accepted a different plate with
        the same five digits
FIX P4  a missing allowlist.json gives a clear message
FIX P5  allowlist entries are compared as letters and digits only
P6 addition: python3 layer1_plate.py photo.jpg  checks a photo
With document 1's main.py, set USE_LAYER1_PLATE = True to turn this layer
on. The original main.py never calls this file.
Known limit, not changed: this reads the current Iraqi format (Latin
letters, Western digits). Older plates with Arabic numerals are not read.
"""
import cv2
import pytesseract
import numpy as np
import logging
import json
from pathlib import Path
from dataclasses import dataclass
from typing import Optional
from thefuzz import fuzz

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


@dataclass
class PlateResult:
    authorized: bool
    plate_text: Optional[str] = None
    match_score: float = 0.0
    fail_reason: Optional[str] = None

    def summary(self) -> str:
        status = "PASS" if self.authorized else "FAIL"
        return (
            f"{status} | Plate: {self.plate_text or 'None'} | "
            f"Match: {self.match_score:.1f}%"
            + (f" | Reason: {self.fail_reason}" if self.fail_reason else "")
        )


class PlateVerifier:
    # FIX P3: was 75. On an 8-character plate, 75 accepted a DIFFERENT plate
    # with the same five digits (14J12345 for 11A12345 scores exactly 75),
    # and a read of the five digits alone (77). 85 allows one wrong
    # character - enough to absorb a typical OCR mistake.
    def __init__(self, allowlist_path: str = "allowlist.json", match_threshold: int = 85):
        self.match_threshold = match_threshold
        self.allowlist = self._load_allowlist(allowlist_path)
        logger.info(
            f"PlateVerifier ready | {len(self.allowlist)} plate(s) in allowlist"
        )

    def _load_allowlist(self, path: str):
        # FIX P4: clear message instead of a bare traceback
        try:
            with open(path) as f:
                data = json.load(f)
        except FileNotFoundError:
            raise FileNotFoundError(f"{path} not found - add it with a \"plates\" list")

        # FIX P5: was .replace(" ", "") only. Hyphens or dots left in an
        # entry such as "11-A-12345" lower every score: with the stricter
        # threshold (P3) one OCR mistake then scores 78 instead of 88 and
        # the owner is refused. Keep letters and digits only - exactly what
        # the OCR output keeps.
        return ["".join(ch for ch in p.upper() if ch.isalnum()) for p in data.get("plates", [])]

    def _preprocess(self, frame: np.ndarray):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.bilateralFilter(gray, 11, 17, 17)
        thresh = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 15
        )
        return thresh

    def _plate_crops(self, frame: np.ndarray):
        """FIX P1: likely plate regions, tightest first, whole frame last.

        The OCR settings read a cropped plate well - but they were given the
        WHOLE camera frame with --psm 8 ("treat the image as one word"). On
        full camera frames that read no plate at all (0 of 18 test scenes).
        This finds the band of dark characters on a light plate, straightens
        and enlarges it, and hands the OCR just that."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT,
                                    cv2.getStructuringElement(cv2.MORPH_RECT, (25, 9)))
        _, mask = cv2.threshold(blackhat, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE,
                                cv2.getStructuringElement(cv2.MORPH_RECT, (31, 9)))
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        crops = []
        for c in sorted(contours, key=cv2.contourArea, reverse=True)[:10]:
            (cx, cy), (w, h), angle = cv2.minAreaRect(c)
            if w < h:
                w, h, angle = h, w, angle + 90
            if h < 12 or not 2.5 <= w / h <= 8.0:
                continue

            # straighten a tilted plate, then cut it out with a small margin
            M = cv2.getRotationMatrix2D((cx, cy), angle, 1.0)
            upright = cv2.warpAffine(frame, M, (frame.shape[1], frame.shape[0]))
            x1, y1 = int(cx - w / 2 - 0.04 * w), int(cy - h / 2 - 0.15 * h)
            x2, y2 = int(cx + w / 2 + 0.04 * w), int(cy + h / 2 + 0.15 * h)
            crop = upright[max(0, y1):y2, max(0, x1):x2]
            if crop.size:
                # Tesseract wants characters ~30 px tall or more
                scale = 100.0 / crop.shape[0]
                crops.append(cv2.resize(crop, None, fx=scale, fy=scale,
                                        interpolation=cv2.INTER_CUBIC))
            if len(crops) == 4:
                break

        return crops + [frame]

    def verify_frame(self, frame: np.ndarray) -> PlateResult:
        if frame is None:
            return PlateResult(authorized=False, fail_reason="Empty frame")

        if not self.allowlist:
            return PlateResult(
                authorized=False,
                fail_reason="No plates in allowlist.json — add a 'plates' list",
            )

        # FIX P1: run your OCR on each likely plate region, keep the best read
        plate_text, best_score = "", 0
        for region in self._plate_crops(frame):
            processed = self._preprocess(region)
            raw_text = pytesseract.image_to_string(
                processed,
                # FIX P2: Iraqi plates use only these 17 letters - C G I O P U V
                # X Y never appear. Allowing them invited O/0 and I/1 mix-ups.
                config="--psm 8 -c tessedit_char_whitelist=ABDEFHJKLMNQRSTWZ0123456789",
            )
            text = "".join(ch for ch in raw_text.upper() if ch.isalnum())
            if not text:
                continue
            score = max(fuzz.ratio(text, known) for known in self.allowlist)
            if score > best_score:
                plate_text, best_score = text, score
            if best_score >= self.match_threshold:
                break

        if not plate_text:
            return PlateResult(authorized=False, fail_reason="No plate text detected")

        if best_score < self.match_threshold:
            logger.warning(f"LAYER 1 FAIL | read='{plate_text}' best={best_score}%")
            return PlateResult(
                authorized=False,
                plate_text=plate_text,
                match_score=best_score,
                fail_reason=f"No allowlist match (best {best_score}%)",
            )

        result = PlateResult(authorized=True, plate_text=plate_text, match_score=best_score)
        logger.info(f"LAYER 1 PASS | {result.summary()}")
        return result

    def verify_image(self, image_path: str) -> PlateResult:
        frame = cv2.imread(image_path)
        if frame is None:
            raise FileNotFoundError(f"Image not found: {image_path}")
        return self.verify_frame(frame)


if __name__ == "__main__":
    # P6 (addition): check photos from the command line, without main.py.
    #   python3 layer1_plate.py                    show the allowlist
    #   python3 layer1_plate.py car1.jpg car2.jpg   read and check each photo
    import sys

    verifier = PlateVerifier()
    print(f"Plates in allowlist: {verifier.allowlist}")
    for path in sys.argv[1:]:
        print(f"{path}: {verifier.verify_image(path).summary()}")
