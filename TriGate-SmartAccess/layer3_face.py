"""layer3_face.py -- PATCHED

Your file. Same class, same FaceResult, same thresholds. Four defects
fixed. `verify_frame(frame)` still works exactly as before, so nothing
that calls it needs to change.

  FIX 1  line 139  encodings[0] took whichever face came first. With a
                   driver and a passenger in frame, a PASSENGER could
                   authorise the car. This is a security bug, not a
                   performance one.
  FIX 2  line 129  detected again at fx=0.5 after main.py already detected
                   at fx=0.25 — roughly double the cost for no new
                   information. verify_frame() now accepts the locations
                   main.py already has.
  FIX 3  line 51   open(path) with no guard: a missing or malformed
                   allowlist.json crashes at construction, before any
                   logging, with a bare traceback.
  FIX 4  line 140  distances were computed against EVERY enrolled face,
                   including people not on the allowlist. A non-allowlisted
                   face that happens to sit slightly closer produced a
                   false DENY for an authorised driver standing right there.
"""

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import face_recognition
import numpy as np

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


@dataclass
class FaceResult:
    authorized: bool
    identity: Optional[str] = None
    similarity: float = 0.0
    distance: float = 1.0
    fail_reason: Optional[str] = None
    location: Optional[Tuple[int, int, int, int]] = None   # NEW, full-frame

    def summary(self) -> str:
        status = "PASS" if self.authorized else "FAIL"
        return (
            f"{status} | Identity: {self.identity or 'None'} | "
            f"Similarity: {self.similarity:.1f}% | "
            f"Distance: {self.distance:.4f}"
            + (f" | Reason: {self.fail_reason}" if self.fail_reason else "")
        )


class FaceVerifier:
    def __init__(self, db_path: str = "face_db",
                 allowlist_path: str = "allowlist.json",
                 tolerance: float = 0.5,
                 detect_scale: float = 0.5):
        self.tolerance = tolerance
        self.db_path = Path(db_path)
        self.detect_scale = detect_scale

        self.known_encodings: list = []
        self.known_names: List[str] = []

        self.allowlist = self._load_allowlist(allowlist_path)
        self._load_face_db()

        logger.info(f"FaceVerifier ready | {len(self.known_encodings)} "
                    f"face(s) loaded | {len(self.allowlist)} allowed")

    def _load_allowlist(self, path: str) -> List[str]:
        # [FIX 3] was a bare open() — a missing file crashed the process
        # at construction with no useful message.
        p = Path(path)
        if not p.exists():
            logger.error(f"{path} not found — NOBODY will be authorised. "
                         f'Create it with {{"faces": ["yourname"], '
                         f'"plates": [], "vehicles": []}}')
            return []
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
        except Exception as e:
            logger.error(f"{path} is not valid JSON ({e}) — "
                         "NOBODY will be authorised")
            return []
        names = [str(n).lower() for n in data.get("faces", [])]
        if not names:
            logger.warning(f'{path} has no "faces" entries')
        return names

    def _load_face_db(self):
        if not self.db_path.exists():
            raise FileNotFoundError(
                f"face_db folder not found: {self.db_path}\n"
                "Create it and add subfolders with photos.")

        valid_ext = {".jpg", ".jpeg", ".png"}
        loaded = 0
        for person_folder in sorted(self.db_path.iterdir()):
            if not person_folder.is_dir():
                continue
            name = person_folder.name.lower()
            for img_file in person_folder.glob("*"):
                if img_file.suffix.lower() not in valid_ext:
                    continue
                img = face_recognition.load_image_file(str(img_file))
                encodings = face_recognition.face_encodings(img)
                if not encodings:
                    logger.warning(f"No face found in {img_file.name} — skipping")
                    continue
                self.known_encodings.append(encodings[0])
                self.known_names.append(name)
                loaded += 1

        if loaded == 0:
            logger.warning("No faces loaded from face_db — add photos to "
                           "face_db/name/ folders")

        # [FIX 4] pre-compute which enrolled faces are actually allowed, so
        # matching happens only among them.
        self._allowed_idx = [i for i, n in enumerate(self.known_names)
                             if n in self.allowlist]
        if self.known_encodings and not self._allowed_idx:
            logger.warning("None of the enrolled faces are on the allowlist "
                           "— every attempt will be denied")

    # ------------------------------------------------------------------

    def verify_frame(self, frame: np.ndarray,
                     locations=None,
                     location_scale: float = 1.0) -> FaceResult:
        """[FIX 2] `locations` lets the caller pass detections it already
        has. `location_scale` is the factor those coordinates were detected
        at (main.py detects at 0.25, so pass 0.25). Omit both and this
        behaves exactly like your original."""
        if frame is None:
            return FaceResult(authorized=False, fail_reason="Empty frame")
        if not self.known_encodings:
            return FaceResult(authorized=False,
                              fail_reason="No faces enrolled in database")

        if locations:
            inv = 1.0 / location_scale if location_scale else 1.0
            full = [tuple(int(v * inv) for v in loc) for loc in locations]
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            work_locs = full
        else:
            s = self.detect_scale
            small = cv2.resize(frame, (0, 0), fx=s, fy=s)
            rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
            work_locs = face_recognition.face_locations(rgb)
            full = [tuple(int(v / s) for v in loc) for loc in work_locs]

        if not work_locs:
            return FaceResult(authorized=False,
                              fail_reason="No face detected in frame")

        # [FIX 1] THE security fix.
        # Was: face_enc = encodings[0]
        # face_locations() does not return faces in any meaningful order,
        # so with two people in frame the passenger could be the one
        # checked. The driver is the largest face — closest to the camera.
        areas = [(loc[2] - loc[0]) * (loc[1] - loc[3]) for loc in work_locs]
        best_face = int(np.argmax(areas))
        chosen = [work_locs[best_face]]

        encodings = face_recognition.face_encodings(rgb, chosen)
        if not encodings:
            return FaceResult(authorized=False,
                              fail_reason="Could not encode face",
                              location=full[best_face])

        face_enc = encodings[0]
        loc = full[best_face]

        # [FIX 4] compare only against people on the allowlist
        pool = self._allowed_idx or list(range(len(self.known_encodings)))
        encs = [self.known_encodings[i] for i in pool]
        distances = face_recognition.face_distance(encs, face_enc)

        k = int(np.argmin(distances))
        best_dist = float(distances[k])
        best_name = self.known_names[pool[k]]
        similarity = max(0.0, (1 - best_dist) * 100)

        if best_dist > self.tolerance:
            logger.warning(f"LAYER 3 FAIL | Best: '{best_name}' "
                           f"dist={best_dist:.4f}")
            return FaceResult(authorized=False, identity=best_name,
                              distance=best_dist, similarity=similarity,
                              location=loc,
                              fail_reason=f"Distance {best_dist:.3f} > "
                                          f"tolerance {self.tolerance}")

        if best_name not in self.allowlist:
            logger.warning(f"Face recognized but not in allowlist: {best_name}")
            return FaceResult(authorized=False, identity=best_name,
                              distance=best_dist, similarity=similarity,
                              location=loc,
                              fail_reason=f"'{best_name}' not in allowlist")

        result = FaceResult(authorized=True, identity=best_name,
                            distance=best_dist, similarity=similarity,
                            location=loc)
        logger.info(f"LAYER 3 PASS | {result.summary()}")
        return result

    def verify_image(self, image_path: str) -> FaceResult:
        frame = cv2.imread(image_path)
        if frame is None:
            raise FileNotFoundError(f"Image not found: {image_path}")
        return self.verify_frame(frame)

    # ------------------------------------------------------------------

    def enroll_from_camera(self, name: str, num_photos: int = 8,
                           camera=0):
        """Unchanged except `camera`: your version hard-coded
        VideoCapture(0), which on the Pi can be a different device from the
        by-id path main.py uses. Enrolling through one lens and verifying
        through another costs accuracy for no reason."""
        save_dir = self.db_path / name.lower()
        save_dir.mkdir(parents=True, exist_ok=True)

        cap = cv2.VideoCapture(camera)
        if not cap.isOpened():
            logger.error("Camera not available for enrollment")
            return

        print(f"\nEnrolling: {name}")
        print("Look at the camera — photos taken automatically")
        print("Vary your angle and expression between shots")
        print("Press Q to stop early\n")

        count = 0
        while count < num_photos:
            ret, frame = cap.read()
            if not ret:
                break
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            locs = face_recognition.face_locations(rgb)
            if locs:
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
        print(f"Enrolled {count} photos for '{name}'")
