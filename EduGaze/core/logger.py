import csv
import json
import os
import tempfile
from datetime import datetime
from typing import List

import pandas as pd

from .types import StudentSnapshot


LOG_FIELDS = [
    "timestamp",
    "student_id",
    "student_name",
    "face_detected",
    "emotion",
    "engagement_score",
    "raw_emotion_confidences",
    "emotion_confidence",
    "face_bbox",
    "face_area_ratio",
    "detection_confidence",
    "attention_score",
    "motion_level",
    "brightness",
    "ocr_confidence",
    "gaze_x",
    "gaze_y",
    "blink_rate",
    "head_yaw",
    "head_pitch",
    "head_roll",
]


class EngagementLogger:
    """
    Handles CSV logging while tolerating locked files (e.g., when opened in Excel).
    Falls back to a temp queue file if the target CSV cannot be written.
    """

    def __init__(self, csv_path: str):
        self.csv_path = csv_path
        self.temp_csv_path = os.path.splitext(self.csv_path)[0] + "_queue.tmp"
        self._perm_warned = False

        os.makedirs(os.path.dirname(os.path.abspath(csv_path)), exist_ok=True)
        self._ensure_header(self.csv_path)

    def log_snapshot(self, student: StudentSnapshot):
        """Append a snapshot row; never raises to keep pipeline alive."""
        row = self._build_row(student)

        try:
            self._write_row(self.csv_path, row)
        except PermissionError:
            # Notify once so the user knows why logging is degraded
            if not self._perm_warned:
                print(
                    "[EduGaze] Warning: engagement_log.csv is locked (maybe open in Excel). "
                    "Logging to a temporary queue until the file becomes writable."
                )
                self._perm_warned = True
            self._write_row(self.temp_csv_path, row, allow_permission_retry=False)
        except Exception as exc:
            print(f"[EduGaze] Logging error: {exc}")

    def to_dataframe(self) -> pd.DataFrame:
        """Read main + temp queue CSVs with resilient parsing."""
        frames: List[pd.DataFrame] = []

        for path in [self.csv_path, self.temp_csv_path]:
            if not os.path.exists(path):
                continue
            df = self._read_csv_safe(path)
            if not df.empty:
                frames.append(df)

        if not frames:
            return pd.DataFrame(columns=LOG_FIELDS)

        combined = pd.concat(frames, ignore_index=True)

        # Ensure all expected columns exist in the right order
        for field in LOG_FIELDS:
            if field not in combined.columns:
                combined[field] = ""
        combined = combined[LOG_FIELDS]
        return combined

    # ----------------------------
    # Internal helpers
    # ----------------------------

    def _build_row(self, student: StudentSnapshot) -> dict:
        face_bbox_str = ""
        if student.last_bbox:
            bx, by, bw, bh = student.last_bbox
            face_bbox_str = f"({bx},{by},{bw},{bh})"

        return {
            "timestamp": datetime.utcnow().isoformat(),
            "student_id": student.id,
            "student_name": student.name,
            "face_detected": 1 if student.last_bbox is not None else 0,
            "emotion": student.last_emotion or "",
            "engagement_score": f"{student.last_engagement:.4f}" if student.last_engagement is not None else "",
            "raw_emotion_confidences": json.dumps(student.last_emotion_confidences) if student.last_emotion_confidences else "",
            "emotion_confidence": f"{student.emotion_confidence:.4f}" if student.emotion_confidence is not None else "",
            "face_bbox": face_bbox_str,
            "face_area_ratio": f"{student.face_area_ratio:.4f}" if student.face_area_ratio is not None else "",
            "detection_confidence": f"{student.detection_confidence:.4f}" if student.detection_confidence is not None else "",
            "attention_score": f"{student.attention_score:.4f}" if student.attention_score is not None else "",
            "motion_level": f"{student.motion_level:.4f}" if student.motion_level is not None else "",
            "brightness": f"{student.brightness:.4f}" if student.brightness is not None else "",
            "ocr_confidence": f"{student.ocr_confidence:.2f}" if student.ocr_confidence is not None else "",
            "gaze_x": f"{student.gaze_x:.4f}" if student.gaze_x is not None else "",
            "gaze_y": f"{student.gaze_y:.4f}" if student.gaze_y is not None else "",
            "blink_rate": f"{student.blink_rate:.4f}" if student.blink_rate is not None else "",
            "head_yaw": f"{student.head_yaw:.2f}" if student.head_yaw is not None else "",
            "head_pitch": f"{student.head_pitch:.2f}" if student.head_pitch is not None else "",
            "head_roll": f"{student.head_roll:.2f}" if student.head_roll is not None else "",
        }

    def _write_row(self, path: str, row: dict, allow_permission_retry: bool = True):
        try:
            need_header = not os.path.exists(path) or os.path.getsize(path) == 0
        except OSError:
            need_header = True

        try:
            with open(path, "a", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=LOG_FIELDS)
                if need_header:
                    writer.writeheader()
                writer.writerow(row)
        except PermissionError:
            if allow_permission_retry:
                raise
        except OSError:
            # If temp directory disappears or disk error occurs, write to system temp file
            fallback = os.path.join(tempfile.gettempdir(), "edugaze_fallback.csv")
            if path == fallback:
                return  # Avoid infinite recursion
            self._write_row(fallback, row, allow_permission_retry=False)

    def _ensure_header(self, path: str):
        """Ensure CSV has the expected header; migrate older schemas if needed."""
        expected = ",".join(LOG_FIELDS)
        if not os.path.exists(path):
            with open(path, "w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=LOG_FIELDS)
                writer.writeheader()
            return

        try:
            with open(path, "r", encoding="utf-8") as handle:
                first_line = handle.readline().strip()
        except Exception:
            first_line = ""

        if first_line == expected:
            return

        backup_path = path + ".bak"
        try:
            df = pd.read_csv(
                path,
                encoding="utf-8",
                engine="python",
                on_bad_lines="skip",
            )
        except Exception as exc:
            print(f"[EduGaze] Warning: could not migrate existing log ({exc}). Backup saved to {backup_path}")
            os.replace(path, backup_path)
            with open(path, "w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=LOG_FIELDS)
                writer.writeheader()
            return

        for field in LOG_FIELDS:
            if field not in df.columns:
                df[field] = ""
        df = df[LOG_FIELDS]
        os.replace(path, backup_path)
        df.to_csv(path, index=False, encoding="utf-8")
        print(f"[EduGaze] Migrated engagement log to new schema. Backup saved at {backup_path}")

    def _read_csv_safe(self, path: str) -> pd.DataFrame:
        """Read CSV while tolerating bad lines between versions."""
        if not os.path.exists(path):
            return pd.DataFrame(columns=LOG_FIELDS)

        try:
            try:
                df = pd.read_csv(path, on_bad_lines="skip", encoding="utf-8")
            except TypeError:
                # Older pandas
                df = pd.read_csv(path, error_bad_lines=False, warn_bad_lines=False, encoding="utf-8")
        except Exception as exc:
            print(f"[EduGaze] CSV read warning for {os.path.basename(path)}: {exc}")
            return pd.DataFrame(columns=LOG_FIELDS)

        return df


