import time
from typing import Dict, Optional
from concurrent.futures import ThreadPoolExecutor, Future

import cv2
import numpy as np

from .engagement import engagement_score, attention_score
from .emotion import EmotionRecognizer
from .face_detector import FaceDetector
from .face_mesh_analyzer import FaceMeshAnalyzer
from .logger import EngagementLogger
from .ocr import OCRBox, ZoomNameOCR
from .screen_capture import ScreenCapturer
from .tracker import CentroidTracker
from .types import BBox, StudentSnapshot


class ProcessingPipeline:
    """
    Ties together screen capture, face detection, OCR, tracking, emotion, engagement, and logging.
    """

    def __init__(self):
        self.capturer = ScreenCapturer()
        self.detector = FaceDetector()
        self.ocr = ZoomNameOCR()
        self.tracker = CentroidTracker()
        self.emotion = EmotionRecognizer()
        self.face_mesh = FaceMeshAnalyzer()
        self.logger: Optional[EngagementLogger] = None

        self.students: Dict[int, StudentSnapshot] = {}

        self._last_emotion_time = 0.0
        self._emotion_interval = 5.0
        self._latest_detection_conf: Dict[int, float] = {}
        self._last_ocr_boxes = []
        self._last_label_regions = []
        self._last_mic_icons = []
        self._prev_gray: Optional[np.ndarray] = None
        self._motion_map: Optional[np.ndarray] = None
        self._motion_norm = 4.0  # empirical scale factor for optical-flow magnitude
        self._frame_index = 0
        self._debug_iterations = 5  # Detailed logs for first few frames
        self._max_detection_width = 1280  # downscale large frames for faster detection
        self._last_ocr_time = 0.0
        self._ocr_interval = 1.5  # seconds between OCR runs
        self._cached_ocr_boxes: list[OCRBox] = []
        self._start_time = 0.0
        self._ocr_warmup_seconds = 3.0  # wait before first OCR so preview appears instantly
        self._ocr_executor = ThreadPoolExecutor(max_workers=1)
        self._pending_ocr_future: Future | None = None

    def start(self, log_path: str):
        self.logger = EngagementLogger(log_path)
        self._last_emotion_time = 0.0
        self._start_time = time.time()
        # Force warmup delay by pretending OCR just ran now
        self._last_ocr_time = self._start_time
        self._cached_ocr_boxes = []
        # Clear any pending OCR future
        if self._pending_ocr_future and not self._pending_ocr_future.done():
            self._pending_ocr_future.cancel()
        self._pending_ocr_future = None

    def stop(self):
        # Nothing special; logging is already flushed per write
        pass

    def finalize(self):
        # Placeholder for any cleanup/reporting triggered externally
        pass

    def _match_names(self, faces: Dict[int, BBox], ocr_boxes: list[OCRBox]):
        """
        For each face ID, find the closest OCR box below the face.
        Prioritizes names found near microphone icons (most reliable).
        """
        if not ocr_boxes:
            print(f"[Pipeline][MatchNames] No OCR boxes to match (faces={len(faces)})")
            return
        
        print(f"[Pipeline][MatchNames] Matching {len(ocr_boxes)} OCR boxes to {len(faces)} faces")
        for i, ob in enumerate(ocr_boxes[:5]):
            print(f"[Pipeline][MatchNames] OCR box {i+1}: '{ob.text}' conf={ob.confidence:.1f}% box={ob.box}")
        
        for sid, bbox in faces.items():
            x, y, w, h = bbox
            face_left = x
            face_bottom_y = y + h
            best_name = None
            best_conf = 0.0
            best_score = -1.0  # Higher is better
            best_box: OCRBox | None = None
            
            for ob in ocr_boxes:
                ox, oy, ow, oh = ob.box
                
                # Only consider boxes below the face (RELAXED for single face)
                # For single face, be very permissive with position
                tolerance = 200 if len(faces) == 1 else 10
                if oy < face_bottom_y - tolerance:
                    continue
                
                # Zoom names appear at bottom-left, so prefer left-aligned
                # Score based on horizontal alignment and distance
                horizontal_overlap = min(x + w, ox + ow) - max(x, ox)
                if horizontal_overlap < 0:
                    continue  # No horizontal overlap
                
                # Check if this OCR box is near a mic icon (big bonus)
                mic_bonus = 1.0
                for mic_x, mic_y, mic_w, mic_h in self._last_mic_icons:
                    # OCR box should be near and to the right of mic icon
                    mic_right = mic_x + mic_w
                    if (ox >= mic_right - 5 and ox <= mic_right + 100 and
                        abs(oy - mic_y) < 15):
                        mic_bonus = 2.0  # Double the score for mic-based names
                        break
                
                # Distance from face bottom-left corner
                dx = abs(ox - face_left)  # Prefer boxes starting near face left edge
                dy = max(0, oy - face_bottom_y)
                distance = (dx**2 + dy**2) ** 0.5
                
                # Prefer left-aligned, close, high-confidence boxes
                # Score = confidence_weight * horizontal_alignment_weight * mic_bonus / distance
                alignment_bonus = 1.0 / (1.0 + dx / max(w, 50))  # Favor left alignment
                conf_weight = ob.confidence / 100.0
                distance_penalty = 1.0 / (1.0 + distance / max(h, 100))
                
                # Bonus for longer names (full names like "John Doe" vs just "Doe")
                name_words = len(ob.text.split())
                name_length_bonus = 1.0 + (name_words - 1) * 0.4  # +40% per additional word
                
                score = conf_weight * alignment_bonus * distance_penalty * mic_bonus * name_length_bonus
                
                if score > best_score:
                    best_score = score
                    best_name = ob.text
                    best_conf = ob.confidence
                    best_box = ob

            # If no name found yet, try fallback strategies
            if best_name is None and ocr_boxes:
                # Strategy: Find any OCR box that overlaps with face region
                for ob in ocr_boxes:
                    ox, oy, ow, oh = ob.box
                    # Check if OCR box overlaps horizontally with face (VERY RELAXED)
                    # Allow OCR boxes anywhere near the face region
                    if (ox <= face_left + w + 300 and ox + ow >= face_left - 200 and  # Much wider horizontal tolerance
                        oy >= face_bottom_y - 150 and ob.confidence >= 20):  # Allow above face too (150px tolerance)
                        best_name = ob.text
                        best_conf = ob.confidence
                        best_box = ob
                        print(f"[Pipeline] Using overlap fallback: '{best_name}' (conf={best_conf:.1f}%)")
                        break
                
                # If still no name, and only one face, use best OCR result (VERY RELAXED)
                if best_name is None and len(faces) == 1:
                    # Accept ANY name with confidence >= 20% for single face
                    # Prioritize proper names like "Jain" (capitalized, 4+ chars)
                    candidates = [ob for ob in ocr_boxes if ob.confidence >= 20 and len(ob.text) >= 2]
                    if candidates:
                        # Sort by: proper name status (capitalized 4+ char words), word count, then confidence
                        def name_score(b):
                            text = b.text.strip()
                            words = text.split()
                            # Big bonus for proper names (single word, 4+ chars, capitalized like "Jain")
                            proper_name_bonus = 10 if (len(words) == 1 and len(text) >= 4 and text[0].isupper()) else 0
                            word_count = len(words)
                            return (proper_name_bonus, word_count, b.confidence)
                        fallback = max(candidates, key=name_score)
                        if fallback:
                            best_name = fallback.text
                            best_conf = fallback.confidence
                            best_box = fallback
                            print(f"[Pipeline] Using fallback name for single face: '{best_name}' (words={len(best_name.split())}, conf={best_conf:.1f}%)")
                elif best_name is None:
                    # For multiple faces, still try to match any high-confidence name
                    candidates = [ob for ob in ocr_boxes if ob.confidence >= 40 and len(ob.text) >= 2]
                    if candidates and len(candidates) <= len(faces):
                        # Try to match candidates to faces by proximity
                        for ob in candidates:
                            ox, oy, ow, oh = ob.box
                            # Check if this box is below any face
                            for fid, fbbox in faces.items():
                                fx, fy, fw, fh = fbbox
                                if oy >= fy + fh - 50:  # Below face
                                    best_name = ob.text
                                    best_conf = ob.confidence
                                    best_box = ob
                                    print(f"[Pipeline] Using proximity match: '{best_name}' for face {fid} (conf={best_conf:.1f}%)")
                                    break
                            if best_name:
                                break
                else:
                    # Otherwise fallback to the highest confidence box under the face even if
                    # it did not fully meet scoring threshold
                    if best_box is None:
                        overlap_candidates = [
                            ob for ob in ocr_boxes
                            if ob.box[0] <= face_left + w and (ob.box[0] + ob.box[2]) >= face_left and ob.confidence >= 30
                        ]
                        if overlap_candidates:
                            # Prefer multi-word names even with slightly lower confidence
                            fallback = max(overlap_candidates, key=lambda b: (len(b.text.split()), b.confidence))
                            if fallback:
                                best_name = fallback.text
                                best_conf = fallback.confidence
                                best_box = fallback
                                print(f"[Pipeline] Using overlap fallback: '{best_name}' (words={len(best_name.split())}, conf={best_conf:.1f}%)")

            # Update or create student entry
            if sid not in self.students:
                # Initialize student (shouldn't happen often since we create students earlier, but fallback)
                name = best_name if (best_name and len(best_name) >= 2 and best_conf >= 35) else f"Unknown-{sid}"
                self.students[sid] = StudentSnapshot(
                    id=sid,
                    name=name,
                    ocr_confidence=best_conf if best_name else None,
                    last_bbox=bbox,
                    detection_confidence=self._latest_detection_conf.get(sid),
                )
                if best_name:
                    print(f"[Pipeline][MatchNames] Created student {sid} with name: '{name}' (conf={best_conf:.1f}%)")
                else:
                    print(f"[Pipeline][MatchNames] Created student {sid} as '{name}' (no name found)")
            else:
                s = self.students[sid]
                # Update name if we get a better match
                # Always update if current name is "Unknown", otherwise require better confidence
                if best_name and len(best_name) >= 2 and best_conf >= 25:  # Lower threshold to 25% to catch more names
                    # More aggressive updating: prefer longer, more complete names
                    # Count words to prefer full names ("John Doe" > "Doe")
                    current_words = len(s.name.split()) if s.name else 0
                    best_words = len(best_name.split()) if best_name else 0
                    
                    # ALWAYS update Unknown names if we have ANY valid name
                    if s.name.startswith("Unknown"):
                        is_better = True
                    else:
                        is_better = (
                            best_words > current_words or  # Prefer more words (full names vs last names)
                            (best_words == current_words and len(best_name) > len(s.name)) or  # Prefer longer names
                            (best_conf >= (s.ocr_confidence or 0) + 10)  # Or significantly better confidence
                        )
                    if is_better:
                        old_name = s.name
                        s.name = best_name
                        s.ocr_confidence = best_conf
                        print(f"[Pipeline] ✓ Updated student {sid} name: '{old_name}' -> '{best_name}' (conf={best_conf:.1f}%, words={best_words})")
                elif best_name and len(best_name) >= 2:
                    # Even if confidence is slightly low, always update Unknown names (VERY RELAXED)
                    if s.name.startswith("Unknown") and best_conf >= 20:  # Lower threshold to 20%
                        old_name = s.name
                        s.name = best_name
                        s.ocr_confidence = best_conf
                        print(f"[Pipeline] ✓ Updated Unknown student {sid} to '{best_name}' (conf={best_conf:.1f}%)")
                    else:
                        print(f"[Pipeline] Keeping '{s.name}' over '{best_name}' (words: {len(s.name.split()) if s.name else 0} vs {len(best_name.split())}, conf: {s.ocr_confidence:.1f if s.ocr_confidence else 0} vs {best_conf:.1f})")
                elif best_name:
                    print(f"[Pipeline] Rejected '{best_name}' (too short: {len(best_name)} chars, conf={best_conf:.1f}%)")
                else:
                    print(f"[Pipeline] No name found for student {sid} from {len(ocr_boxes)} OCR boxes")
                s.last_bbox = bbox

    def process_frame(self):
        self._frame_index += 1
        try:
            frame = self.capturer.grab()
            if frame is None:
                print("[Pipeline] grab() returned None")
                return None, self.students
            if frame.size == 0:
                print("[Pipeline] Warning: Received empty frame from capturer")
                return None, self.students
            # Log first successful frame
            if not hasattr(self, '_first_frame_logged'):
                print(f"[Pipeline] First frame captured: shape={frame.shape}, size={frame.size}")
                self._first_frame_logged = True
        except Exception as e:
            print(f"[Pipeline] Error in process_frame (capture): {e}")
            import traceback
            traceback.print_exc()
            return None, self.students
        
        # Downscale frame for detection if needed to keep Mediapipe fast
        detection_frame = frame
        restore_scale = None
        if frame.shape[1] > self._max_detection_width:
            scale = self._max_detection_width / frame.shape[1]
            new_w = max(320, int(frame.shape[1] * scale))
            new_h = max(240, int(frame.shape[0] * scale))
            detection_frame = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)
            restore_scale = (frame.shape[1] / new_w, frame.shape[0] / new_h)

        debug = self._frame_index <= self._debug_iterations

        # ALWAYS check for completed OCR results FIRST (before any processing)
        # This ensures we get OCR results even if they completed between frames
        if self._pending_ocr_future is not None:
            if self._pending_ocr_future.done():
                try:
                    result = self._pending_ocr_future.result(timeout=0.1)  # Get result (should be instant since done())
                    if result:
                        ocr_boxes_new, mic_icons, duration = result
                        self._cached_ocr_boxes = ocr_boxes_new
                        self._last_mic_icons = mic_icons
                        self._last_ocr_time = time.time()
                        print(
                            f"[Pipeline][Frame {self._frame_index}] OCR result completed "
                            f"(names={len(ocr_boxes_new)}, took {duration:.3f}s)"
                        )
                        if len(ocr_boxes_new) > 0:
                            name_list = [ob.text for ob in ocr_boxes_new[:5]]
                            print(f"[Pipeline] OCR detected names: {name_list}")
                            # Log OCR box details for debugging
                            for i, ob in enumerate(ocr_boxes_new[:3]):
                                print(f"[Pipeline]   OCR box {i+1}: '{ob.text}' conf={ob.confidence:.1f}% box={ob.box}")
                            # Immediately try to match names with tracked faces
                            # (tracked should be available from previous frame processing)
                            if hasattr(self, '_last_tracked') and len(self._last_tracked) > 0:
                                print(f"[Pipeline] Matching {len(ocr_boxes_new)} OCR names to {len(self._last_tracked)} tracked faces")
                                self._match_names(self._last_tracked, ocr_boxes_new)
                            else:
                                print(f"[Pipeline] No tracked faces available for matching (hasattr={hasattr(self, '_last_tracked')})")
                        else:
                            print(f"[Pipeline] OCR found 0 names")
                except Exception as ocr_exc:
                    print(f"[Pipeline] Error retrieving OCR result: {ocr_exc}")
                    import traceback
                    traceback.print_exc()
                finally:
                    self._pending_ocr_future = None

        # Process the frame - if processing fails, still return the raw frame
        try:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            self._update_motion_map(gray)
            if debug:
                print(f"[Pipeline][Frame {self._frame_index}] Running face detection "
                      f"(scale restore={restore_scale})")
            detect_start = time.time()
            detections = self.detector.detect(detection_frame, restore_scale)
            detect_duration = time.time() - detect_start
            if debug or detect_duration > 0.4:
                print(
                    f"[Pipeline][Frame {self._frame_index}] Face detection took "
                    f"{detect_duration:.3f}s, faces={len(detections)}"
                )

            # If we downscaled and found nothing, retry on full resolution
            if not detections and restore_scale is not None:
                if debug:
                    print(
                        f"[Pipeline][Frame {self._frame_index}] No faces detected at scaled "
                        "resolution. Retrying on full resolution."
                    )
                detect_start = time.time()
                detections = self.detector.detect(frame)
                detect_duration = time.time() - detect_start
                if debug or detect_duration > 0.4:
                    print(
                        f"[Pipeline][Frame {self._frame_index}] Full-res detection took "
                        f"{detect_duration:.3f}s, faces={len(detections)}"
                    )

            rects = [bbox for bbox, _ in detections]
            tracked = self.tracker.update(rects)
            self._assign_detection_confidences(tracked, detections)

            # Ensure all tracked faces have student entries (create immediately, not waiting for OCR)
            for sid, bbox in tracked.items():
                if sid not in self.students:
                    self.students[sid] = StudentSnapshot(
                        id=sid,
                        name=f"Unknown-{sid}",  # Will be updated by _match_names if OCR finds a name
                        last_bbox=bbox,
                        detection_confidence=self._latest_detection_conf.get(sid),
                    )
                    if debug:
                        print(f"[Pipeline][Frame {self._frame_index}] Created new student ID {sid}")

            label_regions = self._build_label_regions(tracked, frame.shape)
            face_boxes = list(tracked.values())

            now = time.time()
            warmup_active = (now - self._start_time) < self._ocr_warmup_seconds if self._start_time else False
            run_ocr = (
                not warmup_active
                and ((now - self._last_ocr_time) >= self._ocr_interval or not self._cached_ocr_boxes)
            )
            # Re-read cached boxes in case OCR results were just updated above
            ocr_boxes = list(self._cached_ocr_boxes)  # Use latest cached boxes

            if run_ocr and debug:
                print(f"[Pipeline][Frame {self._frame_index}] Running OCR "
                      f"(regions={len(label_regions)}, faces={len(face_boxes)})")
            if warmup_active and debug:
                remaining = self._ocr_warmup_seconds - (now - self._start_time)
                print(f"[Pipeline][Frame {self._frame_index}] Skipping OCR during warmup "
                      f"({remaining:.2f}s remaining)")

            # Launch new OCR job if needed
            if run_ocr:
                can_submit = (
                    self._pending_ocr_future is None or self._pending_ocr_future.done()
                )
                if can_submit:
                    # Copy data to avoid race with main thread
                    frame_copy = frame.copy()
                    label_regions_copy = list(label_regions)
                    face_boxes_copy = list(face_boxes)
                    self._pending_ocr_future = self._ocr_executor.submit(
                        self._run_ocr_job,
                        frame_copy,
                        label_regions_copy,
                        face_boxes_copy,
                        debug,
                    )
                    print(f"[Pipeline][Frame {self._frame_index}] OCR job submitted "
                          f"(regions={len(label_regions)}, faces={len(face_boxes)})")
                else:
                    if debug:
                        print(f"[Pipeline][Frame {self._frame_index}] OCR already running; "
                              f"using cached {len(ocr_boxes)} boxes")

            # Store tracked faces for potential immediate matching when OCR completes
            self._last_tracked = tracked.copy()
            
            # Match names to faces
            if ocr_boxes:
                print(f"[Pipeline][Frame {self._frame_index}] Calling _match_names with {len(ocr_boxes)} boxes, {len(tracked)} faces")
                for i, ob in enumerate(ocr_boxes[:3]):
                    print(f"[Pipeline][Frame {self._frame_index}] OCR box {i+1} for matching: '{ob.text}' conf={ob.confidence:.1f}%")
            self._match_names(tracked, ocr_boxes)
        except Exception as e:
            print(f"[Pipeline] Error in process_frame (processing): {e}")
            import traceback
            traceback.print_exc()
            # Still return the frame even if processing failed, so user can see something
            # Draw a simple message on the frame to indicate processing error
            try:
                cv2.putText(frame, "Processing Error - Displaying Raw Frame", (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            except:
                pass
            return frame, self.students
        
        # Store for visualization
        self._last_ocr_boxes = ocr_boxes
        self._last_label_regions = label_regions
        if not run_ocr:
            # ensure mic icons exist even if OCR skipped (reuse previous)
            self._last_mic_icons = getattr(self, "_last_mic_icons", [])

        now = time.time()
        run_emotion = (now - self._last_emotion_time) >= self._emotion_interval

        if run_emotion:
            self._last_emotion_time = now

        # Update motion and brightness for all tracked students (not just when emotion runs)
        # Wrap processing in try-except to prevent crashes
        try:
            for sid, bbox in tracked.items():
                student = self.students.get(sid)
                if not student:
                    continue

                # Always update motion and brightness (these are fast calculations)
                # Always update fast metrics - not just when emotion runs
                try:
                    frame_area = frame.shape[0] * frame.shape[1]
                    student.face_area_ratio = (bbox[2] * bbox[3]) / frame_area if frame_area else 0.0
                    student.motion_level = self._normalized_motion_level(bbox)
                    student.brightness = self._region_mean(gray, bbox) / 255.0 if bbox and gray is not None else None
                    student.attention_score = attention_score(bbox, student.face_area_ratio)
                except Exception as metric_e:
                    print(f"[Pipeline] Error calculating metrics for student {sid}: {metric_e}")

                if run_emotion:
                    try:
                        emotion, scores = self.emotion.analyze_face(frame, bbox)
                        student.last_emotion = emotion
                        student.last_emotion_confidences = scores
                        emotion_conf = scores.get(emotion, max(scores.values())) if scores else 1.0
                        student.emotion_confidence = emotion_conf
                        
                        # Run face mesh analysis for gaze, blink, head pose
                        try:
                            mesh_results = self.face_mesh.analyze(frame, bbox)
                            student.gaze_x = mesh_results.get("gaze_x")
                            student.gaze_y = mesh_results.get("gaze_y")
                            student.blink_rate = mesh_results.get("blink_rate")
                            student.head_yaw = mesh_results.get("head_yaw")
                            student.head_pitch = mesh_results.get("head_pitch")
                            student.head_roll = mesh_results.get("head_roll")
                        except Exception as mesh_e:
                            print(f"[Pipeline] Face mesh analysis error for student {sid}: {mesh_e}")
                            # Continue with default values
                        
                        student.last_engagement = engagement_score(
                            True,
                            emotion,
                            bbox,
                            student.detection_confidence,
                            student.face_area_ratio,
                            emotion_conf,
                            student.motion_level,
                            student.brightness,
                            student.attention_score,
                            student.gaze_x,
                            student.gaze_y,
                            student.blink_rate,
                            student.head_yaw,
                            student.head_pitch,
                            student.head_roll,
                        )
                        if self.logger is not None:
                            try:
                                self.logger.log_snapshot(student)
                            except Exception as log_exc:
                                print(f"[EduGaze] Logger error: {log_exc}")
                    except Exception as emotion_e:
                        print(f"[Pipeline] Emotion processing error for student {sid}: {emotion_e}")
                        # Continue processing other students
                        continue
        except Exception as e:
            print(f"[Pipeline] Error in emotion processing loop: {e}")
            import traceback
            traceback.print_exc()
            # Continue to drawing - don't let emotion errors block display

        # Always return a frame - even if drawing fails, return the raw frame
        # This ensures the UI always gets something to display
        try:
            preview = frame.copy()

            # Draw detected microphone icons (red circles)
            for mic_x, mic_y, mic_w, mic_h in self._last_mic_icons:
                center = (mic_x + mic_w // 2, mic_y + mic_h // 2)
                radius = max(mic_w, mic_h) // 2 + 2
                cv2.circle(preview, center, radius, (0, 0, 255), 2)  # Red circle for mic icon
                cv2.putText(preview, "MIC", (mic_x, max(10, mic_y - 5)), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, 255), 1)
            
            # Draw OCR label regions (semi-transparent blue boxes)
            for x1, y1, x2, y2 in self._last_label_regions:
                cv2.rectangle(preview, (x1, y1), (x2, y2), (255, 165, 0), 1)  # Orange for OCR search regions
            
            # Draw detected OCR text boxes (yellow boxes)
            for ob in self._last_ocr_boxes:
                ox, oy, ow, oh = ob.box
                cv2.rectangle(preview, (ox, oy), (ox + ow, oy + oh), (0, 255, 255), 2)  # Yellow for detected text
                # Show detected text
                cv2.putText(
                    preview,
                    f"{ob.text} ({ob.confidence:.0f}%)",
                    (ox, max(15, oy - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.4,
                    (0, 255, 255),
                    1,
                    cv2.LINE_AA,
                )
            
            # Draw face bounding boxes and student info
            for sid, bbox in tracked.items():
                x, y, w, h = bbox
                cv2.rectangle(preview, (x, y), (x + w, y + h), (0, 255, 0), 2)
                name = self.students.get(sid).name if sid in self.students else f"ID-{sid}"
                student = self.students.get(sid)
                lines = [f"{sid}: {name}"]
                if student:
                    eng = f"{(student.last_engagement or 0.0) * 100:.0f}%"
                    motion = f"M:{(student.motion_level or 0.0):.2f}"
                    bright = f"B:{(student.brightness or 0.0):.2f}"
                    emotion = student.last_emotion or "-"
                    lines.append(f"E:{emotion} S:{eng}")
                    lines.append(f"{motion} A:{(student.attention_score or 0.0):.2f} {bright}")
                for idx, text in enumerate(lines):
                    cv2.putText(
                        preview,
                        text,
                        (x, max(15, y - 10 - idx * 15)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.45,
                        (0, 0, 0),
                        3,
                        cv2.LINE_AA,
                    )
                    cv2.putText(
                        preview,
                        text,
                        (x, max(15, y - 10 - idx * 15)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.45,
                        (0, 255, 0),
                        1,
                        cv2.LINE_AA,
                    )

            print("[Pipeline] Drawing complete, returning preview")
            return preview, self.students
        except Exception as e:
            print(f"[Pipeline] Error in process_frame (drawing): {e}")
            import traceback
            traceback.print_exc()
            # Return frame without overlays if drawing fails
            print("[Pipeline] Returning raw frame after drawing error")
            return frame, self.students

    def _run_ocr_job(
        self,
        frame_bgr: np.ndarray,
        label_regions: list[tuple[int, int, int, int]],
        face_boxes: list[BBox],
        debug: bool = False,
    ):
        """Background job that performs OCR without blocking the UI thread."""
        start = time.time()
        print(f"[Pipeline][OCR Job] Starting OCR job (regions={len(label_regions)}, faces={len(face_boxes)})")
        boxes = []
        mic_icons = []
        try:
            # Detect mic icons first (fast)
            print(f"[Pipeline][OCR Job] Detecting mic icons...")
            mic_start = time.time()
            mic_icons = self.ocr.detect_mic_icons(frame_bgr)
            mic_duration = time.time() - mic_start
            print(f"[Pipeline][OCR Job] Mic icons detected: {len(mic_icons)}, took {mic_duration:.3f}s")
            
            # Then detect names
            print(f"[Pipeline][OCR Job] Detecting names (regions={len(label_regions)}, faces={len(face_boxes)})...")
            names_start = time.time()
            boxes = self.ocr.detect_names(frame_bgr, label_regions=label_regions, face_boxes=face_boxes)
            names_duration = time.time() - names_start
            print(f"[Pipeline][OCR Job] Name detection completed: {len(boxes)} names, took {names_duration:.3f}s")
            
            duration = time.time() - start
            print(f"[Pipeline][OCR Job] OCR completed: {len(boxes)} names, {len(mic_icons)} mic icons, total took {duration:.3f}s")
            if len(boxes) > 0:
                print(f"[Pipeline][OCR Job] Names found: {[ob.text for ob in boxes[:5]]}")
            else:
                print(f"[Pipeline][OCR Job] No names detected")
            return boxes, mic_icons, duration
        except Exception as e:
            duration = time.time() - start
            print(f"[Pipeline][OCR Job] OCR job error after {duration:.3f}s: {e}")
            import traceback
            traceback.print_exc()
            return [], getattr(self, "_last_mic_icons", []), duration

    def _build_label_regions(self, tracked: Dict[int, BBox], frame_shape) -> list[tuple[int, int, int, int]]:
        """
        Build precise label regions focused on bottom-left corners of video tiles.
        Zoom names appear as small dark overlays at bottom-left of each participant video.
        """
        h, w = frame_shape[:2]
        regions = []
        
        for bbox in tracked.values():
            x, y, bw, bh = bbox
            # Zoom names appear BELOW the face box, at the bottom of the video tile
            face_bottom = y + bh
            screen_bottom = h
            
            # Calculate space below face
            space_below = screen_bottom - face_bottom
            
            # Zoom names are at bottom of video container (can be far below face)
            # Scan a wide region below the face
            label_height = min(100, max(50, space_below))  # 50-100px below face
            label_width = max(400, int(bw * 1.2))  # Wide enough for full names
            
            # Start from bottom-left, extending BELOW the face
            label_x1 = max(0, x - 30)  # Slight left offset
            label_y1 = max(face_bottom + 5, face_bottom)  # Start BELOW face
            label_x2 = min(w, label_x1 + label_width)
            label_y2 = min(h, label_y1 + label_height)
            
            # Also try a slightly wider region to catch names that might be offset
            wide_x1 = max(0, x - int(0.1 * bw))
            wide_x2 = min(w, x + label_width + int(0.2 * bw))
            
            # Add primary region (tight)
            regions.append((label_x1, label_y1, label_x2, label_y2))
            # Add wider region as fallback
            if wide_x2 > wide_x1 and label_y2 > label_y1:
                regions.append((wide_x1, label_y1, wide_x2, label_y2))
        
        # Global fallback: bottom-left quadrant of screen (for single-participant speaker view)
        if not regions:
            regions.append((0, int(h * 0.75), int(w * 0.5), h))
        else:
            # Also add a general bottom-left region as additional fallback
            fallback_h = int(h * 0.2)
            regions.append((0, h - fallback_h, int(w * 0.5), h))
        
        return regions

    def _assign_detection_confidences(self, tracked: Dict[int, BBox], detections: list[tuple[BBox, float]]):
        self._latest_detection_conf = {}
        for sid, bbox in tracked.items():
            best_conf = 0.0
            for det_bbox, conf in detections:
                iou = self._bbox_iou(bbox, det_bbox)
                if iou > 0.3 and conf > best_conf:
                    best_conf = conf
            self._latest_detection_conf[sid] = best_conf
            if sid in self.students:
                self.students[sid].detection_confidence = best_conf

    @staticmethod
    def _bbox_iou(boxA: BBox, boxB: BBox) -> float:
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[0] + boxA[2], boxB[0] + boxB[2])
        yB = min(boxA[1] + boxA[3], boxB[1] + boxB[3])

        inter_w = max(0, xB - xA)
        inter_h = max(0, yB - yA)
        inter_area = inter_w * inter_h
        if inter_area == 0:
            return 0.0
        boxA_area = boxA[2] * boxA[3]
        boxB_area = boxB[2] * boxB[3]
        return inter_area / float(boxA_area + boxB_area - inter_area)

    def _update_motion_map(self, gray: np.ndarray):
        if self._prev_gray is None:
            self._motion_map = np.zeros_like(gray, dtype=np.float32)
            self._prev_gray = gray
            return
        flow = cv2.calcOpticalFlowFarneback(
            self._prev_gray,
            gray,
            None,
            0.5,
            3,
            15,
            3,
            5,
            1.2,
            0,
        )
        mag, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1], angleInDegrees=False)
        self._motion_map = mag
        self._prev_gray = gray

    def _normalized_motion_level(self, bbox: BBox | None) -> float | None:
        if self._motion_map is None or bbox is None:
            return None
        raw = self._region_mean(self._motion_map, bbox)
        if raw is None:
            return None
        return min(1.0, raw / self._motion_norm)

    @staticmethod
    def _region_mean(matrix: np.ndarray | None, bbox: BBox | None) -> float | None:
        if matrix is None or bbox is None:
            return None
        x, y, w, h = bbox
        h_mat, w_mat = matrix.shape[:2]
        x1 = max(0, x)
        y1 = max(0, y)
        x2 = min(w_mat, x + w)
        y2 = min(h_mat, y + h)
        if x2 <= x1 or y2 <= y1:
            return None
        region = matrix[y1:y2, x1:x2]
        if region.size == 0:
            return None
        return float(region.mean())


