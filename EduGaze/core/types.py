from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

import numpy as np


BBox = Tuple[int, int, int, int]


@dataclass
class StudentSnapshot:
    id: int
    name: str
    last_emotion: Optional[str] = None
    last_emotion_confidences: Dict[str, float] = field(default_factory=dict)
    last_engagement: Optional[float] = None
    last_bbox: Optional[BBox] = None
    ocr_confidence: Optional[float] = None
    detection_confidence: Optional[float] = None
    face_area_ratio: Optional[float] = None
    motion_level: Optional[float] = None
    brightness: Optional[float] = None
    attention_score: Optional[float] = None
    emotion_confidence: Optional[float] = None
    # Face mesh metrics
    gaze_x: Optional[float] = None
    gaze_y: Optional[float] = None
    blink_rate: Optional[float] = None
    head_yaw: Optional[float] = None
    head_pitch: Optional[float] = None
    head_roll: Optional[float] = None


