"""
Face mesh analyzer using MediaPipe for gaze estimation, blink detection, and head pose.
"""
from typing import Dict, Optional, Tuple

import cv2
import mediapipe as mp
import numpy as np

from .types import BBox


class FaceMeshAnalyzer:
    """Analyzes face mesh for gaze, blink rate, and head pose."""

    def __init__(self):
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=25,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        
        # Landmark indices for different features
        # Left eye landmarks
        self.LEFT_EYE_INDICES = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
        # Right eye landmarks
        self.RIGHT_EYE_INDICES = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
        # Eye corners for EAR (Eye Aspect Ratio)
        self.LEFT_EYE_CORNERS = [33, 133, 157, 153, 158, 144]
        self.RIGHT_EYE_CORNERS = [362, 263, 387, 373, 388, 380]
        # Nose tip and bridge for head pose
        self.NOSE_TIP = 1
        self.NOSE_BRIDGE = 6
        # Face outline for head pose estimation
        self.FACE_OUTLINE = [10, 151, 9, 175, 18, 200, 269, 270, 291, 308, 324, 318]

    def analyze(self, frame_bgr: np.ndarray, bbox: BBox) -> Dict[str, float]:
        """
        Analyze face mesh and return gaze, blink, and head pose metrics.
        Returns dict with keys: gaze_x, gaze_y, blink_rate, head_yaw, head_pitch, head_roll
        """
        x, y, w, h = bbox
        h_img, w_img, _ = frame_bgr.shape
        
        # Crop face region with padding
        pad = 10
        x1 = max(0, x - pad)
        y1 = max(0, y - pad)
        x2 = min(w_img, x + w + pad)
        y2 = min(h_img, y + h + pad)
        
        face_crop = frame_bgr[y1:y2, x1:x2]
        if face_crop.size == 0:
            return self._empty_results()
        
        frame_rgb = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(frame_rgb)
        
        if not results.multi_face_landmarks:
            return self._empty_results()
        
        # Use first detected face
        landmarks = results.multi_face_landmarks[0]
        
        # Convert landmarks to image coordinates
        h_crop, w_crop = face_crop.shape[:2]
        landmark_points = []
        for landmark in landmarks.landmark:
            px = int(landmark.x * w_crop) + x1
            py = int(landmark.y * h_crop) + y1
            landmark_points.append((px, py, landmark.z))
        
        # Calculate metrics
        blink_rate = self._calculate_blink_rate(landmark_points)
        head_yaw, head_pitch, head_roll = self._estimate_head_pose(landmark_points, w_img, h_img)
        gaze_x, gaze_y = self._estimate_gaze(landmark_points)
        
        return {
            "gaze_x": gaze_x,
            "gaze_y": gaze_y,
            "blink_rate": blink_rate,
            "head_yaw": head_yaw,
            "head_pitch": head_pitch,
            "head_roll": head_roll,
        }
    
    def _calculate_blink_rate(self, landmarks: list) -> float:
        """Calculate Eye Aspect Ratio (EAR) to detect blinks."""
        try:
            # Left eye EAR
            left_eye_points = [landmarks[i] for i in self.LEFT_EYE_CORNERS if i < len(landmarks)]
            if len(left_eye_points) >= 6:
                left_ear = self._eye_aspect_ratio(left_eye_points)
            else:
                left_ear = 0.3  # Default open eye
            
            # Right eye EAR
            right_eye_points = [landmarks[i] for i in self.RIGHT_EYE_CORNERS if i < len(landmarks)]
            if len(right_eye_points) >= 6:
                right_ear = self._eye_aspect_ratio(right_eye_points)
            else:
                right_ear = 0.3  # Default open eye
            
            # Average EAR
            ear = (left_ear + right_ear) / 2.0
            
            # Blink threshold: EAR < 0.25 typically indicates closed eye
            # Return normalized value: 0 (closed) to 1 (open)
            blink_state = 1.0 if ear > 0.25 else 0.0
            return blink_state
            
        except Exception:
            return 0.5  # Neutral/unknown
    
    @staticmethod
    def _eye_aspect_ratio(eye_points: list) -> float:
        """Calculate Eye Aspect Ratio (EAR)."""
        # Vertical distances
        v1 = np.linalg.norm(np.array(eye_points[1][:2]) - np.array(eye_points[5][:2]))
        v2 = np.linalg.norm(np.array(eye_points[2][:2]) - np.array(eye_points[4][:2]))
        # Horizontal distance
        h = np.linalg.norm(np.array(eye_points[0][:2]) - np.array(eye_points[3][:2]))
        
        if h == 0:
            return 0.3
        ear = (v1 + v2) / (2.0 * h)
        return ear
    
    def _estimate_head_pose(self, landmarks: list, img_w: int, img_h: int) -> Tuple[float, float, float]:
        """Estimate head pose (yaw, pitch, roll) in degrees."""
        try:
            if len(landmarks) < max(self.NOSE_TIP, self.NOSE_BRIDGE, *self.FACE_OUTLINE):
                return (0.0, 0.0, 0.0)
            
            # Get key points
            nose_tip = np.array(landmarks[self.NOSE_TIP][:2])
            nose_bridge = np.array(landmarks[self.NOSE_BRIDGE][:2])
            face_points = [np.array(landmarks[i][:2]) for i in self.FACE_OUTLINE[:6] if i < len(landmarks)]
            
            if len(face_points) < 3:
                return (0.0, 0.0, 0.0)
            
            # Calculate yaw (left-right rotation) using face outline asymmetry
            left_face = np.mean([p[0] for p in face_points[:3]])
            right_face = np.mean([p[0] for p in face_points[3:]])
            face_center_x = (left_face + right_face) / 2
            nose_offset_x = nose_tip[0] - face_center_x
            yaw = np.clip(nose_offset_x / img_w * 90, -45, 45)  # Rough estimate
            
            # Calculate pitch (up-down rotation) using nose bridge
            face_center_y = np.mean([p[1] for p in face_points])
            nose_offset_y = nose_tip[1] - face_center_y
            pitch = np.clip(nose_offset_y / img_h * 60, -30, 30)  # Rough estimate
            
            # Calculate roll (tilt) using face outline
            if len(face_points) >= 2:
                left_eye = np.mean([p[1] for p in face_points[:2]])
                right_eye = np.mean([p[1] for p in face_points[-2:]])
                roll = np.arctan2(right_eye - left_eye, right_face - left_face) * 180 / np.pi
                roll = np.clip(roll, -30, 30)
            else:
                roll = 0.0
            
            return (float(yaw), float(pitch), float(roll))
            
        except Exception:
            return (0.0, 0.0, 0.0)
    
    def _estimate_gaze(self, landmarks: list) -> Tuple[float, float]:
        """Estimate gaze direction (normalized coordinates)."""
        try:
            if len(landmarks) < max(self.LEFT_EYE_INDICES + self.RIGHT_EYE_INDICES):
                return (0.5, 0.5)  # Center gaze
            
            # Get eye centers
            left_eye_points = [np.array(landmarks[i][:2]) for i in self.LEFT_EYE_INDICES[:4] if i < len(landmarks)]
            right_eye_points = [np.array(landmarks[i][:2]) for i in self.RIGHT_EYE_INDICES[:4] if i < len(landmarks)]
            
            if len(left_eye_points) < 2 or len(right_eye_points) < 2:
                return (0.5, 0.5)
            
            left_eye_center = np.mean(left_eye_points, axis=0)
            right_eye_center = np.mean(right_eye_points, axis=0)
            eye_center = (left_eye_center + right_eye_center) / 2.0
            
            # Rough gaze estimation based on eye position relative to face
            # This is simplified; full gaze estimation requires iris detection
            # Normalize to 0-1 range (0=left/top, 1=right/bottom)
            gaze_x = np.clip(eye_center[0] / 1000.0, 0.0, 1.0) if eye_center[0] > 0 else 0.5
            gaze_y = np.clip(eye_center[1] / 1000.0, 0.0, 1.0) if eye_center[1] > 0 else 0.5
            
            return (float(gaze_x), float(gaze_y))
            
        except Exception:
            return (0.5, 0.5)
    
    @staticmethod
    def _empty_results() -> Dict[str, float]:
        """Return empty results when face mesh fails."""
        return {
            "gaze_x": 0.5,
            "gaze_y": 0.5,
            "blink_rate": 0.5,
            "head_yaw": 0.0,
            "head_pitch": 0.0,
            "head_roll": 0.0,
        }


