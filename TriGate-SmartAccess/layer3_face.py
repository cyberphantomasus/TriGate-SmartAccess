"""
layer3_face.py - Layer 3: face check

Corrected in the TriGate code review - Document 4 of 6.
Full explanation, tests and installation steps: TriGate_Doc4_layer3_face.pdf

Same class names, functions and arguments as the original. Every change is
marked next to the lines it touches:

  FIX F1  when several faces are in view, check the largest one (the
          driver), not whichever the detector listed first (could be the
          passenger)
  FIX F2  a missing or broken allowlist.json gives a clear message
  FIX F3  enrol through the same camera main.py uses, not /dev/video0
  FIX F4  an enrolment photo with more than one face is skipped - the
          original could store the OTHER person under this name
  F5      addition: python3 layer3_face.py [photo.jpg ...] shows who is
          enrolled and checks photos

Unchanged on purpose: the person is identified among ALL enrolled faces
first, and only then checked against the allowlist. That is the safe order.

Known limit, not changed: there is no liveness check. A printed photo or a
phone screen showing an allowed face can pass. See the document, chapter 7.
"""
import face_recognition
import numpy as np
import logging
import json
import glob  # FIX F3
import cv2
from pathlib import Path
from dataclasses import dataclass
from typing import Optional

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


@dataclass
class FaceResult:
    authorized: bool
    identity: Optional[str] = None
    similarity: float = 0.0
    distance: float = 1.0
    fail_reason: Optional[str] = None

    def summary(self) -> str:
        status = "✅ PASS" if self.authorized else "❌ FAIL"
        return (
            f"{status} | Identity: {self.identity or 'None'} | "
            f"Similarity: {self.similarity:.1f}% | "
            f"Distance: {self.distance:.4f}"
            + (f" | Reason: {self.fail_reason}" if self.fail_reason else "")
        )


class FaceVerifier:
    def __init__(
        self,
        db_path: str = "face_db",
        allowlist_path: str = "allowlist.json",
        tolerance: float = 0.5,
    ):
        self.tolerance  = tolerance
        self.db_path    = Path(db_path)

        self.known_encodings: list = []
        self.known_names: list[str] = []

        self.allowlist = self._load_allowlist(allowlist_path)
        self._load_face_db()

        logger.info(f"✅ FaceVerifier ready | {len(self.known_encodings)} face(s) loaded")

    def _load_allowlist(self, path: str) -> list[str]:
        # FIX F2: a missing or broken allowlist.json crashed with a bare
        # traceback that did not say what to do. Same result, clear message.
        try:
            with open(path) as f:
                data = json.load(f)
        except FileNotFoundError:
            raise FileNotFoundError(
                f"{path} not found. Create it, for example:\n"
                '  {"faces": ["mahdi"], "plates": ["11 A 12345"]}')
        except json.JSONDecodeError as e:
            raise ValueError(f"{path} is not valid JSON: {e}")
        return [name.lower() for name in data.get("faces", [])]

    def _load_face_db(self):
        if not self.db_path.exists():
            raise FileNotFoundError(
                f"face_db folder not found: {self.db_path}\n"
                "Create it and add subfolders with photos."
            )

        valid_ext = {".jpg", ".jpeg", ".png"}
        loaded    = 0
        skipped   = 0                                             # FIX F4

        for person_folder in sorted(self.db_path.iterdir()):
            if not person_folder.is_dir():
                continue
            name = person_folder.name.lower()
            for img_file in person_folder.glob("*"):
                if img_file.suffix.lower() not in valid_ext:
                    continue
                img = face_recognition.load_image_file(str(img_file))
                # FIX F4: was face_encodings(img) and then encodings[0] - the
                # FIRST face in the photo. In a photo of two people, the
                # OTHER person could be stored under this name, and then let
                # in as them. A photo with more than one face is skipped.
                locations = face_recognition.face_locations(img)
                if len(locations) > 1:
                    logger.warning(f"{len(locations)} faces in {img_file.name} — skipping "
                                   f"(use photos with only {name}'s face)")
                    skipped += 1
                    continue
                encodings = face_recognition.face_encodings(img, locations)
                if not encodings:
                    logger.warning(f"No face found in {img_file.name} — skipping")
                    continue
                self.known_encodings.append(encodings[0])
                self.known_names.append(name)
                loaded += 1

        if loaded == 0:
            logger.warning("⚠️  No faces loaded from face_db — add photos to face_db/name/ folders")
        # FIX F4: never let the skip above lock someone out without saying so
        if skipped:
            logger.warning(f"{skipped} photo(s) skipped because they show more than one face")
        for allowed in self.allowlist:
            if allowed not in self.known_names:
                logger.warning(f"'{allowed}' is in the allowlist but has no usable photo "
                               f"in {self.db_path}/{allowed}/ — they cannot be recognised")

    def enroll_from_camera(self, name: str, num_photos: int = 8):
        save_dir = self.db_path / name.lower()
        save_dir.mkdir(parents=True, exist_ok=True)

        # FIX F3: was VideoCapture(0). On a Pi 5 the board's own video
        # engines also appear as /dev/videoN, so index 0 is not guaranteed to
        # be your USB camera. Enrol through the same camera main.py uses.
        cams = sorted(glob.glob("/dev/v4l/by-id/*-video-index0"))
        cap = cv2.VideoCapture(cams[0] if cams else 0)
        if not cap.isOpened():
            logger.error("Camera not available for enrollment")
            return

        print(f"\nEnrolling: {name}")
        print("Look at the camera — photos taken automatically")
        print("Press Q to stop early\n")

        count = 0
        while count < num_photos:
            ret, frame = cap.read()
            if not ret:
                break
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            locs = face_recognition.face_locations(rgb)
            if len(locs) == 1:     # FIX F4: was `if locs:` - saved frames with 2+ faces
                path = save_dir / f"img{count}.jpg"
                cv2.imwrite(str(path), frame)
                count += 1
                logger.info(f"Saved {path}")
            cv2.putText(frame, f"Captured: {count}/{num_photos}",
                        (10, 40), cv2.FONT_HERSHEY_SIMPLEX,
                        1, (0, 255, 0), 2)
            cv2.imshow("Enrollment", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()
        self.known_encodings.clear()
        self.known_names.clear()
        self._load_face_db()
        print(f"✅ Enrolled {count} photos for '{name}'")

    def verify_frame(self, frame: np.ndarray) -> FaceResult:
        if frame is None:
            return FaceResult(authorized=False, fail_reason="Empty frame")
        if not self.known_encodings:
            return FaceResult(authorized=False, fail_reason="No faces enrolled in database")

        small = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
        rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
        locations = face_recognition.face_locations(rgb)
        if not locations:
            return FaceResult(authorized=False, fail_reason="No face detected in frame")

        # FIX F1: encodings[0] used whichever face the detector listed first,
        # and that order means nothing. With a driver and a passenger in the
        # frame, the PASSENGER could be the one checked - an authorised
        # passenger could open the gate for a stranger driving. Check the
        # largest face: the person nearest the camera.
        if len(locations) > 1:
            locations = [max(locations, key=lambda l: (l[2] - l[0]) * (l[1] - l[3]))]

        encodings = face_recognition.face_encodings(rgb, locations)
        if not encodings:
            return FaceResult(authorized=False, fail_reason="Could not encode face")

        face_enc = encodings[0]
        distances = face_recognition.face_distance(self.known_encodings, face_enc)

        best_idx  = int(np.argmin(distances))
        best_dist = float(distances[best_idx])
        best_name = self.known_names[best_idx]
        similarity = max(0.0, (1 - best_dist) * 100)

        if best_dist > self.tolerance:
            logger.warning(f"❌ LAYER 3 FAIL | Best: '{best_name}' dist={best_dist:.4f}")
            return FaceResult(
                authorized=False, identity=best_name, distance=best_dist,
                similarity=similarity,
                fail_reason=f"Distance {best_dist:.3f} > tolerance {self.tolerance}"
            )

        if best_name not in self.allowlist:
            logger.warning(f"❌ Face recognized but not in allowlist: {best_name}")
            return FaceResult(
                authorized=False, identity=best_name, distance=best_dist,
                similarity=similarity, fail_reason=f"'{best_name}' not in allowlist"
            )

        result = FaceResult(authorized=True, identity=best_name, distance=best_dist, similarity=similarity)
        logger.info(f"✅ LAYER 3 PASS | {result.summary()}")
        return result

    def verify_image(self, image_path: str) -> FaceResult:
        frame = cv2.imread(image_path)
        if frame is None:
            raise FileNotFoundError(f"Image not found: {image_path}")
        return self.verify_frame(frame)


if __name__ == "__main__":
    # F5 (addition): check the face database and photos without main.py.
    #   python3 layer3_face.py                     who is enrolled
    #   python3 layer3_face.py me.jpg other.jpg    check each photo
    import sys
    from collections import Counter
    verifier = FaceVerifier()
    counts = Counter(verifier.known_names)
    print(f"{'name':16s}{'photos':>7s}  allowlist")
    for person in sorted(set(counts) | set(verifier.allowlist)):
        allowed = "allowed" if person in verifier.allowlist else "not allowed"
        print(f"{person:16s}{counts.get(person, 0):7d}  {allowed}")
    for path in sys.argv[1:]:
        print(f"{path}: {verifier.verify_image(path).summary()}")
