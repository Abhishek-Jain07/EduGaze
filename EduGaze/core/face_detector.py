from typing import List, Tuple

import cv2
import mediapipe as mp
import numpy as np

from .types import BBox


class FaceDetector:
    """
    Wrapper around MediaPipe face detection that tries both close-range (model 0)
    and long-range (model 1) detectors, automatically falling back when the first
    one returns no faces.
    """

    def __init__(self, min_confidence: float = 0.5):
        self.mp_face_detection = mp.solutions.face_detection
        self.detectors = [
            self.mp_face_detection.FaceDetection(
                model_selection=0, min_detection_confidence=min_confidence
            ),
            self.mp_face_detection.FaceDetection(
                model_selection=1, min_detection_confidence=min_confidence * 0.9
            ),
        ]

    def detect(
        self,
        frame_bgr: np.ndarray,
        restore_scale: Tuple[float, float] | None = None,
    ) -> List[Tuple[BBox, float]]:
        """
        Return list of (bbox, confidence) in pixel coords.

        :param frame_bgr: Frame passed to Mediapipe (can be downscaled)
        :param restore_scale: Optional (scale_x, scale_y) to map boxes back to the
                              original resolution. Each value is the factor to
                              multiply the detected coordinates by.
        """
        for idx, detector in enumerate(self.detectors):
            faces = self._run_detector(detector, frame_bgr, restore_scale)
            if faces:
                if idx == 1:
                    print(
                        f"[FaceDetector] Long-range model detected {len(faces)} face(s)"
                    )
                return faces
        return []

    def _run_detector(
        self,
        detector: mp.solutions.face_detection.FaceDetection,
        frame_bgr: np.ndarray,
        restore_scale: Tuple[float, float] | None = None,
    ) -> List[Tuple[BBox, float]]:
        h, w, _ = frame_bgr.shape
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        results = detector.process(frame_rgb)
        faces: List[Tuple[BBox, float]] = []
        if not results.detections:
            return faces
        for det in results.detections:
            bbox = det.location_data.relative_bounding_box
            x = int(bbox.xmin * w)
            y = int(bbox.ymin * h)
            bw = int(bbox.width * w)
            bh = int(bbox.height * h)
            conf = float(det.score[0]) if det.score else 0.0
            if restore_scale is not None:
                scale_x, scale_y = restore_scale
                x = int(x * scale_x)
                y = int(y * scale_y)
                bw = int(bw * scale_x)
                bh = int(bh * scale_y)
            faces.append(((x, y, bw, bh), conf))
        return faces





