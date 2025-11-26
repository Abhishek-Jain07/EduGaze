from typing import Dict, Tuple

import cv2
import numpy as np
from fer import FER

from .types import BBox


class EmotionRecognizer:
    """
    Lightweight emotion recognizer using the `fer` library.
    Expects face crops in BGR format.
    """

    def __init__(self):
        # mtcnn=False because we already provide faces
        self.detector = FER(mtcnn=False)

    def analyze_face(self, frame_bgr: np.ndarray, bbox: BBox) -> Tuple[str, Dict[str, float]]:
        x, y, w, h = bbox
        h_img, w_img, _ = frame_bgr.shape
        x = max(0, x)
        y = max(0, y)
        w = max(1, min(w, w_img - x))
        h = max(1, min(h, h_img - y))

        crop = frame_bgr[y : y + h, x : x + w]
        if crop.size == 0:
            return "neutral", {"neutral": 1.0}

        # FER expects RGB
        crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        # FER returns a list of results; we'll just analyze the single crop
        emotions = self.detector.detect_emotions(crop_rgb)
        if not emotions:
            return "neutral", {"neutral": 1.0}

        scores = emotions[0]["emotions"]
        # Normalize to our supported emotion labels
        mapping = {
            "happy": "happy",
            "neutral": "neutral",
            "sad": "sad",
            "angry": "angry",
            "surprise": "surprised",
        }
        mapped_scores: Dict[str, float] = {}
        for src, dest in mapping.items():
            if src in scores:
                mapped_scores[dest] = float(scores[src])
        if not mapped_scores:
            mapped_scores = {"neutral": 1.0}
        best_emotion = max(mapped_scores.items(), key=lambda kv: kv[1])[0]
        return best_emotion, mapped_scores





