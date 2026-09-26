"""
layer2_vehicle.py - Layer 2: vehicle check
Corrected in the TriGate code review - Document 3 of 6.
Full explanation, tests and installation steps: TriGate_Doc3_layer2_vehicle.pdf
Same class names, functions and arguments as the original. Every change is
marked next to the lines it touches:
FIX V1  reference photos are cropped to the car exactly like the live
        frame (before: whole photo vs car-only crop); photos with no
        detectable car are skipped
FIX V2  SUVs, pickups and vans (YOLO "truck"/"bus") are checked too
V3 addition: python3 layer2_vehicle.py photo.jpg  checks a photo
With document 1's main.py, set USE_LAYER2_VEHICLE = True to turn this layer
on. The original main.py never calls this file.
Known limit, not changed: this layer compares COLOUR only. A different car
of the same colour can pass. See the document, chapter 7.
"""
import cv2
import numpy as np
import logging
from pathlib import Path
from dataclasses import dataclass
from typing import Optional
from ultralytics import YOLO

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


@dataclass
class VehicleResult:
    authorized: bool
    similarity: float = 0.0
    fail_reason: Optional[str] = None

    def summary(self) -> str:
        status = "✅ PASS" if self.authorized else "❌ FAIL"
        return (
            f"{status} | Similarity: {self.similarity:.1f}%"
            + (f" | Reason: {self.fail_reason}" if self.fail_reason else "")
        )


class VehicleVerifier:
    def __init__(
        self,
        ref_dir: str = "vehicle_db/accent",
        model_path: str = "yolov8n.pt",
        similarity_threshold: float = 0.55,
    ):
        logger.info("Loading YOLOv8 model (first run downloads weights)...")
        self.model = YOLO(model_path)
        self.ref_dir = Path(ref_dir)
        self.similarity_threshold = similarity_threshold
        self.ref_hists = self._load_reference_histograms()
        logger.info(
            f"✅ VehicleVerifier ready | {len(self.ref_hists)} reference photo(s) loaded"
        )

    # FIX V2: COCO class 2 is "car". YOLO often labels SUVs, pickups and vans
    # as 7 ("truck"), and large vehicles as 5 ("bus"). Accept all three.
    VEHICLE_CLASSES = (2, 5, 7)

    def _car_crop(self, img: np.ndarray):
        """FIX V1: the largest vehicle YOLO finds, or None.

        Used for the reference photos AND the live frame, so both sides of
        the comparison are built from the same thing."""
        results = self.model(img, verbose=False)[0]
        boxes = [b for b in results.boxes if int(b.cls[0]) in self.VEHICLE_CLASSES]
        if not boxes:
            return None

        def box_area(b):
            x1, y1, x2, y2 = b.xyxy[0]
            return float((x2 - x1) * (y2 - y1))

        x1, y1, x2, y2 = map(int, max(boxes, key=box_area).xyxy[0])
        crop = img[max(0, y1):y2, max(0, x1):x2]
        return crop if crop.size else None

    def _compute_hist(self, img: np.ndarray):
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist([hsv], [0, 1], None, [50, 60], [0, 180, 0, 256])
        cv2.normalize(hist, hist)
        return hist

    def _load_reference_histograms(self):
        hists = []
        skipped = 0  # FIX V1
        if not self.ref_dir.exists():
            logger.warning(
                f"⚠️  {self.ref_dir} not found — add reference photos of your "
                "car (same rough distance/angle as the real camera position)"
            )
            return hists

        valid_ext = {".jpg", ".jpeg", ".png"}
        for img_file in sorted(self.ref_dir.glob("*")):
            if img_file.suffix.lower() not in valid_ext:
                continue
            img = cv2.imread(str(img_file))
            if img is not None:
                # FIX V1: was _compute_hist(img) on the WHOLE photo - car plus
                # wall, floor and sky - while the live frame used only the car
                # crop. Crop the reference the same way. A photo with no car
                # YOLO can find is skipped: using it whole would bring the
                # background back, and a car YOLO cannot see in a photo it
                # cannot see live either.
                crop = self._car_crop(img)
                if crop is None:
                    logger.warning(f"No car found in {img_file.name} - photo skipped")
                    skipped += 1
                    continue
                hists.append(self._compute_hist(crop))

        if skipped:
            logger.warning(f"{skipped} reference photo(s) skipped - retake them "
                           "with the whole car in view")
        return hists

    def verify_frame(self, frame: np.ndarray) -> VehicleResult:
        if frame is None:
            return VehicleResult(authorized=False, fail_reason="Empty frame")

        if not self.ref_hists:
            return VehicleResult(
                authorized=False,
                fail_reason="No reference photos loaded — add to vehicle_db/accent/",
            )

        crop = self._car_crop(frame)  # FIX V1 + V2
        if crop is None:
            return VehicleResult(authorized=False, fail_reason="No car detected in frame")

        crop_hist = self._compute_hist(crop)
        best_score = max(
            cv2.compareHist(crop_hist, ref, cv2.HISTCMP_CORREL) for ref in self.ref_hists
        )
        similarity = max(0.0, best_score) * 100

        if best_score < self.similarity_threshold:
            logger.warning(f"❌ LAYER 2 FAIL | similarity={similarity:.1f}%")
            return VehicleResult(
                authorized=False,
                similarity=similarity,
                fail_reason=f"Similarity {similarity:.1f}% below threshold",
            )

        result = VehicleResult(authorized=True, similarity=similarity)
        logger.info(f"✅ LAYER 2 PASS | {result.summary()}")
        return result

    def verify_image(self, image_path: str) -> VehicleResult:
        frame = cv2.imread(image_path)
        if frame is None:
            raise FileNotFoundError(f"Image not found: {image_path}")
        return self.verify_frame(frame)


if __name__ == "__main__":
    # V3 (addition): check photos from the command line, without main.py.
    #   python3 layer2_vehicle.py                    count the references
    #   python3 layer2_vehicle.py car1.jpg car2.jpg   check each photo
    import sys

    verifier = VehicleVerifier()
    print(f"Reference photos loaded: {len(verifier.ref_hists)}")
    for path in sys.argv[1:]:
        print(f"{path}: {verifier.verify_image(path).summary()}")
