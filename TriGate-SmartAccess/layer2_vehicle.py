"""layer2_vehicle.py -- PATCHED

Your file. Same class, same VehicleResult, same `verify_frame(frame)`
signature. Four defects fixed and one capability added.

  FIX 1  line 76   `int(b.cls[0]) == 2` accepts COCO class 2 (car) only.
                   YOLO routinely classifies SUVs, pickups and vans as
                   class 7 (truck) and larger vehicles as class 5 (bus).
                   Your own car will sometimes be detected as a truck and
                   silently produce "No car detected in frame".
  FIX 2  line 87   the histogram is computed over the FULL bounding box,
                   which always contains background in its corners —
                   asphalt, sky, the garage wall. That background moves
                   the signature around depending on where the car stops.
  FIX 3  line 45   an H-S histogram identifies COLOUR, not your vehicle.
                   Any car of the same colour passes; your own car under
                   different light fails. This is the single most serious
                   weakness in the layer.
  FIX 4  line 75   YOLO runs at its default 640 px on every frame. At
                   imgsz=320 it is roughly 3x faster with no practical
                   loss at garage distances.

FIX 3 IS THE ONE THAT MATTERS, AND IT IS ADDITIVE
-------------------------------------------------
The histogram is kept, because it is fast and it is a real signal. What
is added alongside it is ORB keypoint matching: structural features —
the shape of the grille, badge placement, light clusters, panel lines —
rather than colour distribution. Two cars of the same colour have very
similar histograms and very different keypoints.

The final score combines them. Colour alone can no longer authorise a
vehicle, which is what you want, and neither can a single bad-light
frame reject your own car.

If you are short on time, you can set `use_orb=False` and get exactly
your original behaviour with FIX 1, 2 and 4 applied. Nothing breaks.
"""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np
from ultralytics import YOLO

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# [FIX 1] COCO vehicle classes, not just car
VEHICLE_CLASSES = {2: "car", 5: "bus", 7: "truck", 3: "motorcycle"}


@dataclass
class VehicleResult:
    authorized: bool
    similarity: float = 0.0
    fail_reason: Optional[str] = None
    colour_score: float = 0.0        # NEW, for debugging
    shape_score: float = 0.0         # NEW
    box: Optional[Tuple[int, int, int, int]] = None   # NEW, for Layer 1 ROI

    def summary(self) -> str:
        status = "PASS" if self.authorized else "FAIL"
        return (f"{status} | Similarity: {self.similarity:.1f}% "
                f"(colour {self.colour_score:.0f}% shape {self.shape_score:.0f}%)"
                + (f" | Reason: {self.fail_reason}" if self.fail_reason else ""))


class VehicleVerifier:
    def __init__(self,
                 ref_dir: str = "vehicle_db/accent",
                 model_path: str = "yolov8n.pt",
                 similarity_threshold: float = 0.55,
                 use_orb: bool = True,
                 colour_weight: float = 0.40,
                 imgsz: int = 320):
        logger.info("Loading YOLOv8 model (first run downloads weights)...")
        self.model = YOLO(model_path)
        self.ref_dir = Path(ref_dir)
        self.similarity_threshold = similarity_threshold
        self.use_orb = use_orb
        self.colour_weight = colour_weight if use_orb else 1.0
        self.imgsz = imgsz                        # [FIX 4]

        self.orb = cv2.ORB_create(nfeatures=800) if use_orb else None
        self.matcher = (cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
                        if use_orb else None)

        self.ref_hists: List[np.ndarray] = []
        self.ref_descs: List[np.ndarray] = []
        self._load_references()

        logger.info(f"VehicleVerifier ready | {len(self.ref_hists)} "
                    f"reference photo(s) | orb={use_orb}")

    # ------------------------------------------------------------------

    @staticmethod
    def _inner(img: np.ndarray, frac: float = 0.72) -> np.ndarray:
        """[FIX 2] Central region of the crop.

        A bounding box is a rectangle around a curved object, so its
        corners are always background. Taking the inner ~72% keeps the
        body panels and drops the asphalt and sky that would otherwise
        change the signature every time the car parks slightly differently.
        """
        h, w = img.shape[:2]
        mh, mw = int(h * (1 - frac) / 2), int(w * (1 - frac) / 2)
        out = img[mh:h - mh, mw:w - mw]
        return out if out.size else img

    def _compute_hist(self, img: np.ndarray) -> np.ndarray:
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist([hsv], [0, 1], None, [50, 60], [0, 180, 0, 256])
        cv2.normalize(hist, hist)
        return hist

    def _compute_desc(self, img: np.ndarray) -> Optional[np.ndarray]:
        """[FIX 3] Structural signature. Grille shape, badge position,
        light clusters, panel lines — things a same-colour car does not
        share."""
        if self.orb is None:
            return None
        g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        g = cv2.resize(g, (320, 240), interpolation=cv2.INTER_AREA)
        g = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(g)
        _, desc = self.orb.detectAndCompute(g, None)
        return desc

    def _load_references(self):
        if not self.ref_dir.exists():
            logger.warning(
                f"{self.ref_dir} not found — add reference photos of your "
                "car (same rough distance/angle as the real camera position)")
            return
        valid_ext = {".jpg", ".jpeg", ".png"}
        for img_file in sorted(self.ref_dir.glob("*")):
            if img_file.suffix.lower() not in valid_ext:
                continue
            img = cv2.imread(str(img_file))
            if img is None:
                continue
            inner = self._inner(img)
            self.ref_hists.append(self._compute_hist(inner))
            d = self._compute_desc(inner)
            if d is not None and len(d):
                self.ref_descs.append(d)

        if self.ref_hists and self.use_orb and not self.ref_descs:
            logger.warning("no ORB features in any reference photo — "
                           "images may be too small or too blurred")

    # ------------------------------------------------------------------

    def _orb_score(self, desc: Optional[np.ndarray]) -> float:
        """0.0-1.0. Lowe ratio test against every reference."""
        if desc is None or not len(desc) or not self.ref_descs:
            return 0.0
        best = 0.0
        for ref in self.ref_descs:
            if ref is None or len(ref) < 2:
                continue
            try:
                pairs = self.matcher.knnMatch(desc, ref, k=2)
            except cv2.error:
                continue
            good = sum(1 for p in pairs
                       if len(p) == 2 and p[0].distance < 0.75 * p[1].distance)
            best = max(best, good / float(min(len(desc), len(ref))))
        # ~0.25 of features matching is already a strong same-object signal
        return float(np.clip(best / 0.25, 0.0, 1.0))

    def verify_frame(self, frame: np.ndarray) -> VehicleResult:
        if frame is None:
            return VehicleResult(authorized=False, fail_reason="Empty frame")
        if not self.ref_hists:
            return VehicleResult(
                authorized=False,
                fail_reason="No reference photos loaded — add to "
                            "vehicle_db/accent/")

        results = self.model(frame, verbose=False, imgsz=self.imgsz)[0]

        # [FIX 1] accept every vehicle class, not only class 2
        boxes = [b for b in results.boxes
                 if int(b.cls[0]) in VEHICLE_CLASSES]
        if not boxes:
            return VehicleResult(authorized=False,
                                 fail_reason="No vehicle detected in frame")

        def area(b):
            x1, y1, x2, y2 = b.xyxy[0]
            return float((x2 - x1) * (y2 - y1))

        best_box = max(boxes, key=area)
        cls_name = VEHICLE_CLASSES.get(int(best_box.cls[0]), "?")
        x1, y1, x2, y2 = map(int, best_box.xyxy[0])
        h, w = frame.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            return VehicleResult(authorized=False,
                                 fail_reason="Invalid crop region")

        inner = self._inner(crop)                       # [FIX 2]

        colour = max(cv2.compareHist(self._compute_hist(inner), ref,
                                     cv2.HISTCMP_CORREL)
                     for ref in self.ref_hists)
        colour = float(max(0.0, colour))

        shape = self._orb_score(self._compute_desc(inner)) if self.use_orb else 0.0

        score = (self.colour_weight * colour
                 + (1.0 - self.colour_weight) * shape)
        similarity = score * 100

        box = (x1, y1, x2, y2)
        if score < self.similarity_threshold:
            logger.warning(f"LAYER 2 FAIL | {cls_name} | "
                           f"combined={similarity:.1f}% "
                           f"(colour {colour*100:.0f}% shape {shape*100:.0f}%)")
            return VehicleResult(
                authorized=False, similarity=similarity,
                colour_score=colour * 100, shape_score=shape * 100, box=box,
                fail_reason=f"Similarity {similarity:.1f}% below threshold")

        result = VehicleResult(authorized=True, similarity=similarity,
                               colour_score=colour * 100,
                               shape_score=shape * 100, box=box)
        logger.info(f"LAYER 2 PASS | {cls_name} | {result.summary()}")
        return result

    def verify_image(self, image_path: str) -> VehicleResult:
        frame = cv2.imread(image_path)
        if frame is None:
            raise FileNotFoundError(f"Image not found: {image_path}")
        return self.verify_frame(frame)


if __name__ == "__main__":
    v = VehicleVerifier()
    print(f"Reference photos loaded: {len(v.ref_hists)}")
    print(f"ORB descriptor sets:     {len(v.ref_descs)}")
    if not v.ref_hists:
        print("\nAdd 8-15 photos of your car to vehicle_db/accent/ from the "
              "REAL camera position, at 3 distances and in 3 lighting "
              "conditions. Without them this layer fails every attempt.")
