from typing import Literal

from .types import BBox


EmotionLabel = Literal["happy", "neutral", "sad", "angry", "surprised"]


EMOTION_POSITIVITY = {
    "happy": 1.0,
    "neutral": 0.65,
    "surprised": 0.55,
    "sad": 0.25,
    "angry": 0.15,
}


TARGET_FACE_RATIO = 0.08  # rough proportion of frame when student is centered/engaged


def head_orientation_score(bbox: BBox | None) -> float:
    """
    Rough heuristic: more square boxes are considered frontal,
    and very wide/flat boxes are treated as turned.
    """
    if bbox is None:
        return 0.0
    _, _, w, h = bbox
    if h == 0:
        return 0.0
    aspect = w / float(h)
    if 0.8 <= aspect <= 1.4:
        return 1.0
    if 0.5 <= aspect <= 1.8:
        return 0.6
    return 0.2


def attention_score(bbox: BBox | None, face_area_ratio: float | None) -> float:
    """
    Combines head orientation and how large the face is relative to the frame.
    Larger faces imply the student is closer and attentive.
    """
    head_score = head_orientation_score(bbox)
    if face_area_ratio is None:
        return head_score * 0.6
    size_score = min(1.0, face_area_ratio / TARGET_FACE_RATIO)
    return min(1.0, head_score * 0.6 + size_score * 0.4)


def motion_score(level: float | None) -> float:
    if level is None:
        return 0.5
    return max(0.0, min(1.0, level))


def brightness_score(value: float | None) -> float:
    if value is None:
        return 0.5
    # Ideal range roughly mid exposure
    if 0.35 <= value <= 0.75:
        return 1.0
    if 0.25 <= value <= 0.85:
        return 0.7
    if 0.15 <= value <= 0.9:
        return 0.4
    return 0.2


def gaze_focus_score(gaze_x: float | None, gaze_y: float | None) -> float:
    """Score based on gaze direction - center gaze indicates attention."""
    if gaze_x is None or gaze_y is None:
        return 0.5  # Neutral
    # Center gaze (0.4-0.6 range) gets high score
    center_distance_x = abs(gaze_x - 0.5)
    center_distance_y = abs(gaze_y - 0.5)
    score_x = max(0.0, 1.0 - center_distance_x * 2.0)
    score_y = max(0.0, 1.0 - center_distance_y * 2.0)
    return (score_x + score_y) / 2.0


def head_pose_engagement_score(head_yaw: float | None, head_pitch: float | None, head_roll: float | None) -> float:
    """Score based on head pose - frontal head indicates engagement."""
    if head_yaw is None:
        head_yaw = 0.0
    if head_pitch is None:
        head_pitch = 0.0
    if head_roll is None:
        head_roll = 0.0
    
    # Frontal head (small angles) gets high score
    yaw_score = max(0.0, 1.0 - abs(head_yaw) / 30.0)  # Penalize if > 30 degrees
    pitch_score = max(0.0, 1.0 - abs(head_pitch) / 20.0)  # Penalize if > 20 degrees
    roll_score = max(0.0, 1.0 - abs(head_roll) / 15.0)  # Penalize if > 15 degrees
    
    return (yaw_score + pitch_score + roll_score) / 3.0


def blink_engagement_score(blink_rate: float | None) -> float:
    """Score based on blink rate - normal blinking indicates alertness."""
    if blink_rate is None:
        return 0.5  # Unknown
    # Higher blink rate (eyes open) = better engagement
    # blink_rate is 0 (closed) to 1 (open) from face mesh
    return blink_rate


def engagement_score(
    face_present: bool,
    emotion: EmotionLabel,
    bbox: BBox | None,
    detection_confidence: float | None,
    face_area_ratio: float | None,
    emotion_confidence: float | None,
    motion_level: float | None,
    brightness: float | None,
    attention_override: float | None = None,
    gaze_x: float | None = None,
    gaze_y: float | None = None,
    blink_rate: float | None = None,
    head_yaw: float | None = None,
    head_pitch: float | None = None,
    head_roll: float | None = None,
) -> float:
    if not face_present or bbox is None:
        return 0.0

    eye_focus = max(0.0, min(1.0, detection_confidence or 0.0))
    emotion_pos = EMOTION_POSITIVITY.get(emotion, 0.6)
    emotion_term = emotion_pos * max(0.2, min(1.0, (emotion_confidence or 0.6)))
    attention = attention_override if attention_override is not None else attention_score(bbox, face_area_ratio)
    motion_term = motion_score(motion_level)
    brightness_term = brightness_score(brightness)
    
    # Face mesh enhanced features
    gaze_score = gaze_focus_score(gaze_x, gaze_y)
    head_pose_score = head_pose_engagement_score(head_yaw, head_pitch, head_roll)
    blink_score = blink_engagement_score(blink_rate)
    
    # Enhanced multimodal scoring with face mesh data
    if blink_rate is not None or head_yaw is not None:
        # Use face mesh data if available (more accurate)
        score = (
            0.25 * eye_focus
            + 0.20 * emotion_term
            + 0.15 * attention
            + 0.15 * gaze_score
            + 0.10 * head_pose_score
            + 0.10 * blink_score
            + 0.05 * motion_term
        )
    else:
        # Fallback to original scoring if face mesh unavailable
        score = (
            0.35 * eye_focus
            + 0.25 * emotion_term
            + 0.2 * attention
            + 0.15 * motion_term
            + 0.05 * brightness_term
        )
    
    return round(min(1.0, max(0.0, score)), 4)


