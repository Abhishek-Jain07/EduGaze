from dataclasses import dataclass
from typing import List, Sequence, Tuple, Optional
import re

import cv2
import numpy as np
import os
import shutil
import pytesseract


@dataclass
class OCRBox:
    text: str
    box: Tuple[int, int, int, int]
    confidence: float


class ZoomNameOCR:
    """
    OCR helper tuned for Zoom-style name labels.
    Uses microphone icon detection as a landmark to locate name labels precisely.
    """

    def __init__(self, tesseract_cmd: str | None = None, timeout: float = 3.0):
        cmd = self._resolve_tesseract_cmd(tesseract_cmd)
        if cmd:
            pytesseract.pytesseract.tesseract_cmd = cmd

        self._char_whitelist = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz .-'_"
        self._allowed_chars = set(self._char_whitelist)
        self._ocr_timeout = timeout
        self._blocked_tokens = {
            "mute",
            "unmute",
            "audio",
            "video",
            "stop",
            "start",
            "share",
            "security",
            "participants",
            "chat",
            "record",
            "apps",
            "view",
            "react",
            "reactions",
            "search",
            "broadcast",
            "host",
            "more",
            "leave",
            "end",
            "zoom",
            "unknown",
            "info",
            "command",
            "selected",
            "appears",
            "corrupt",
        }
    
    def detect_mic_icons(self, frame_bgr: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """
        Detect Zoom microphone icons using multiple methods for robustness.
        Returns list of (x, y, w, h) bounding boxes for detected mic icons.
        """
        mic_boxes = []
        
        # Method 1: HSV color detection (red muted mic)
        hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
        
        # More lenient red ranges to catch variations
        lower_red1 = np.array([0, 50, 50])
        upper_red1 = np.array([15, 255, 255])
        lower_red2 = np.array([165, 50, 50])
        upper_red2 = np.array([180, 255, 255])
        
        mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
        mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
        red_mask = cv2.bitwise_or(mask1, mask2)
        
        # Also check BGR space for bright red pixels
        b, g, r = cv2.split(frame_bgr)
        red_bgr_mask = (r > 150) & (g < 100) & (b < 100)
        red_bgr_mask = red_bgr_mask.astype(np.uint8) * 255
        red_mask = cv2.bitwise_or(red_mask, red_bgr_mask)
        
        # Morphological operations to clean up
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_CLOSE, kernel)
        red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_OPEN, kernel)
        
        # Find contours
        contours, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            area = cv2.contourArea(contour)
            # Mic icons can vary in size (20-1000 pixels)
            if 20 < area < 1000:
                x, y, w, h = cv2.boundingRect(contour)
                # Mic icons are roughly square or slightly wider
                aspect_ratio = w / float(h) if h > 0 else 0
                if 0.5 <= aspect_ratio <= 2.0:
                    mic_boxes.append((x, y, w, h))
        
        # Deduplicate overlapping detections
        return self._deduplicate_mic_boxes(mic_boxes)
    
    @staticmethod
    def _deduplicate_mic_boxes(boxes: List[Tuple[int, int, int, int]]) -> List[Tuple[int, int, int, int]]:
        """Remove overlapping mic icon detections."""
        if not boxes:
            return []
        
        # Sort by area (largest first)
        boxes_with_area = [(b, b[2] * b[3]) for b in boxes]
        boxes_with_area.sort(key=lambda x: x[1], reverse=True)
        
        result = []
        for box, _ in boxes_with_area:
            overlaps = False
            for existing in result:
                # Check overlap
                x1, y1, w1, h1 = box
                x2, y2, w2, h2 = existing
                
                overlap_x = max(0, min(x1 + w1, x2 + w2) - max(x1, x2))
                overlap_y = max(0, min(y1 + h1, y2 + h2) - max(y1, y2))
                overlap_area = overlap_x * overlap_y
                
                box_area = w1 * h1
                if overlap_area > 0.5 * box_area:  # More than 50% overlap
                    overlaps = True
                    break
            
            if not overlaps:
                result.append(box)
        
        return result

    @staticmethod
    def _resolve_tesseract_cmd(user_cmd: str | None) -> str | None:
        """Try multiple locations/env vars to find tesseract.exe on Windows."""
        candidates = []
        if user_cmd:
            candidates.append(user_cmd)
        env_cmd = os.environ.get("TESSERACT_CMD")
        if env_cmd:
            candidates.append(env_cmd)
        # Common Windows installs
        candidates.extend(
            [
                r"C:\Program Files\Tesseract-OCR\tesseract.exe",
                r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
            ]
        )
        which_cmd = shutil.which("tesseract")
        if which_cmd:
            candidates.append(which_cmd)

        for path in candidates:
            if path and os.path.exists(path):
                return path
        return None

    def detect_names(
        self,
        frame_bgr,
        label_regions: Sequence[Tuple[int, int, int, int]] | None = None,
        face_boxes: Sequence[Tuple[int, int, int, int]] | None = None,
    ) -> List[OCRBox]:
        """
        Detect names using comprehensive scanning: bottom screen strip, label regions, face-based, and mic icons.
        """
        all_boxes: List[OCRBox] = []
        h, w = frame_bgr.shape[:2]
        
        # Strategy 0: SCAN ENTIRE BOTTOM PORTION OF SCREEN FIRST (most comprehensive)
        # Zoom names are always at the bottom of video tiles
        print(f"[OCR] Strategy 0: Scanning entire bottom portion of screen...")
        bottom_y_start = max(0, int(h * 0.70))  # Bottom 30% of screen
        bottom_strip = frame_bgr[bottom_y_start:h, 0:w]
        if bottom_strip.size > 0:
            print(f"[OCR] Bottom strip: y={bottom_y_start}-{h}, width={w}, size={bottom_strip.shape}")
            boxes_bottom = self._run_ocr_on_roi(bottom_strip, offset=(0, bottom_y_start), debug_idx=0)
            print(f"[OCR] Found {len(boxes_bottom)} boxes in bottom strip")
            if boxes_bottom:
                for b in boxes_bottom[:5]:
                    print(f"[OCR] Bottom strip text: '{b.text}' conf={b.confidence:.1f}%")
            all_boxes.extend(boxes_bottom)
        
        # Strategy 1: Use pre-calculated label_regions (these are face-based regions)
        if label_regions:
            print(f"[OCR] Strategy 1: Trying {len(label_regions)} label regions...")
            for idx, (x1, y1, x2, y2) in enumerate(label_regions):
                x1 = max(0, x1)
                y1 = max(0, y1)
                x2 = min(w, x2)
                y2 = min(h, y2)
                if x2 <= x1 or y2 <= y1:
                    continue
                print(f"[OCR] Scanning label region {idx+1}: ({x1},{y1}) to ({x2},{y2}) size={(x2-x1)}x{(y2-y1)}")
                roi = frame_bgr[y1:y2, x1:x2]
                if roi.size == 0:
                    continue
                boxes = self._run_ocr_on_roi(roi, offset=(x1, y1), debug_idx=idx+10)
                print(f"[OCR] Found {len(boxes)} boxes in label region {idx+1}")
                if boxes:
                    for b in boxes[:3]:
                        print(f"[OCR] Label region text: '{b.text}' conf={b.confidence:.1f}%")
                all_boxes.extend(boxes)
        
        # Early return if label regions found something
        if all_boxes:
            deduplicated = self._deduplicate_boxes(all_boxes)
            final = [b for b in deduplicated if len(b.text) >= 2]
            if final:
                print(f"[OCR] Returning {len(final)} boxes from label regions")
                return final
        
        # Strategy 1: Face-based scanning (fallback if label regions found nothing)
        if face_boxes:
            for face_x, face_y, face_w, face_h in face_boxes:
                # Zoom names appear BELOW the face, at the bottom-left of the video tile
                # Calculate bottom edge of face box
                face_bottom = face_y + face_h
                
                # Focus on region BELOW face - names are at bottom of video tile
                # Zoom names appear at the very bottom of the video container, not just below face
                # Try scanning from face bottom all the way to screen bottom
                face_center_x = face_x + face_w // 2
                screen_bottom = h
                
                # Calculate how much space is below the face to screen bottom
                space_below = screen_bottom - face_bottom
                
                regions_to_try = [
                    # Region 1: Directly below face, extends almost to screen bottom
                    (max(0, face_x - 30), face_bottom, min(450, w - max(0, face_x - 30)), min(100, space_below)),
                    # Region 2: Centered below face
                    (max(0, face_center_x - 225), face_bottom, min(450, w - max(0, face_center_x - 225)), min(100, space_below)),
                    # Region 3: Bottom-left corner (absolute position)
                    (max(0, face_x - 50), max(face_bottom, screen_bottom - 80), min(500, w - max(0, face_x - 50)), min(80, screen_bottom - max(face_bottom, screen_bottom - 80))),
                    # Region 4: Wider scan from face left edge
                    (max(0, face_x - 40), face_bottom + 10, min(500, w - max(0, face_x - 40)), min(90, space_below - 10)),
                ]
                
                for r_x, r_y, r_w, r_h in regions_to_try:
                    r_x = max(0, r_x)
                    r_y = max(0, min(r_y, h - 30))  # Ensure we don't go beyond screen
                    r_w = min(w - r_x, r_w)
                    r_h = min(h - r_y, r_h)
                    
                    if r_w > 80 and r_h > 25:  # Ensure region is large enough
                        print(f"[OCR] Scanning region below face: ({r_x},{r_y}) size={r_w}x{r_h}")
                        roi = frame_bgr[r_y : r_y + r_h, r_x : r_x + r_w]
                        if roi.size == 0:
                            print(f"[OCR] Warning: Empty ROI")
                            continue
                        boxes = self._run_ocr_on_roi(roi, offset=(r_x, r_y), debug_idx=len(all_boxes))
                        print(f"[OCR] Found {len(boxes)} boxes in region (before filtering)")
                        # Less strict filtering - collect all boxes first, filter later
                        all_boxes.extend(boxes)
                        
                        # Early exit if we found a good name for this face
                        valid_boxes = [b for b in boxes if len(b.text) >= 3]
                        if valid_boxes:
                            print(f"[OCR] Found {len(valid_boxes)} valid boxes, breaking early")
                            break
        
        # Strategy 2: Use mic icons as anchors (secondary - only if face-based failed)
        if not all_boxes or len(all_boxes) < len(face_boxes) if face_boxes else True:
            mic_icons = self.detect_mic_icons(frame_bgr)
            # Limit to top 5 mic icons (fewer to prevent false positives)
            if len(mic_icons) > 5:
                mic_icons = sorted(mic_icons, key=lambda m: m[2] * m[3], reverse=True)[:5]
            
            for mic_x, mic_y, mic_w, mic_h in mic_icons:
                # Zoom name labels appear to the right of the mic icon
                name_region_x = mic_x + mic_w + 2
                name_region_y = mic_y - 2
                name_region_w = min(280, w - name_region_x)
                name_region_h = mic_h + 4
                
                if name_region_w > 60 and name_region_h > 15:
                    name_region_x = max(0, name_region_x)
                    name_region_y = max(0, name_region_y)
                    roi = frame_bgr[name_region_y : name_region_y + name_region_h, 
                                   name_region_x : name_region_x + name_region_w]
                    boxes = self._run_ocr_on_roi(roi, offset=(name_region_x, name_region_y))
                    valid_boxes = [b for b in boxes if len(b.text) >= 4]
                    all_boxes.extend(valid_boxes)
        
        # Strategy 3: Fallback to label regions (these are pre-calculated bottom-left regions)
        if len(all_boxes) < len(face_boxes) if face_boxes else not all_boxes:
            if label_regions:
                print(f"[OCR] Trying {len(label_regions)} label regions as fallback...")
                h, w = frame_bgr.shape[:2]
                for idx, (x1, y1, x2, y2) in enumerate(label_regions):
                    x1 = max(0, x1)
                    y1 = max(0, y1)
                    x2 = min(w, x2)
                    y2 = min(h, y2)
                    if x2 <= x1 or y2 <= y1:
                        continue
                    print(f"[OCR] Scanning label region {idx+1}: ({x1},{y1}) to ({x2},{y2}) size={(x2-x1)}x{(y2-y1)}")
                    roi = frame_bgr[y1:y2, x1:x2]
                    boxes = self._run_ocr_on_roi(roi, offset=(x1, y1))
                    print(f"[OCR] Found {len(boxes)} boxes in label region {idx+1}")
                    valid_boxes = [b for b in boxes if len(b.text) >= 3]  # Reduced from 4 to 3
                    all_boxes.extend(valid_boxes)
                    if valid_boxes:
                        print(f"[OCR] Found valid boxes in label region, breaking")
                        break
        
        # Deduplicate and filter out invalid names
        deduplicated = self._deduplicate_boxes(all_boxes)
        
        # Debug: log all found boxes before filtering
        if deduplicated:
            print(f"[OCR] Total boxes before filtering: {len(deduplicated)}")
            for i, b in enumerate(deduplicated[:5]):
                print(f"[OCR] Box {i+1}: '{b.text}' conf={b.confidence:.1f}%")
        
        # Final filtering: extract proper names from mixed text and filter garbage
        final_boxes = []
        all_words = []  # Collect all words from all boxes for intelligent combination
        
        for box in deduplicated:
            text = box.text.strip()
            if not text:
                continue
            
            # REJECT if text contains blocked UI words (case-insensitive)
            text_lower = text.lower()
            contains_blocked = any(blocked in text_lower for blocked in self._blocked_tokens)
            if contains_blocked:
                print(f"[OCR] ✗ Rejected text containing UI words: '{text}'")
                continue
            
            # Try to extract proper names from mixed text (e.g., "t . o- Jain A ee z" -> "Jain")
            extracted_names = self._extract_proper_names(text)
            
            # If we extracted clean names, create new boxes for them
            if extracted_names:
                for name, conf in extracted_names:
                    if self._is_valid_name(name):
                        # Create a new box with the extracted name
                        new_box = OCRBox(
                            text=name,
                            box=box.box,
                            confidence=max(conf, box.confidence)  # Use higher confidence
                        )
                        final_boxes.append(new_box)
                        all_words.append((name.split(), box.box, box.confidence))
                        print(f"[OCR] ✓ Extracted name from garbage: '{text}' -> '{name}'")
            else:
                # Original text might be clean - validate it strictly
                is_valid = self._is_valid_name(text)
                if is_valid:
                    # Also check it doesn't contain blocked words
                    if not any(blocked in text_lower for blocked in self._blocked_tokens):
                        final_boxes.append(box)
                        all_words.append((text.split(), box.box, box.confidence))
                        print(f"[OCR] ✓ Accepted clean name: '{text}' (conf={box.confidence:.1f}%)")
                    else:
                        print(f"[OCR] ✗ Rejected text containing UI words: '{text}'")
                else:
                    print(f"[OCR] ✗ Rejected garbage text: '{text}' (failed validation)")
        
        # Try intelligent name fragment combination across all detected words
        # Look for patterns like "John" + "Doe" that might form "John Doe"
        if all_words:
            reconstructed_names = self._reconstruct_names_from_fragments(all_words)
            for name, box, conf in reconstructed_names:
                if name not in [b.text for b in final_boxes]:  # Avoid duplicates
                    final_boxes.append(OCRBox(text=name, box=box, confidence=conf))
                    print(f"[OCR] ✓ Reconstructed name from fragments: '{name}'")
        
        # Try to combine nearby boxes that might form full names
        # This is a final pass to merge names detected separately (e.g., "John" and "Doe")
        combined_final = self._combine_nearby_words(final_boxes) if final_boxes else []
        
        # Prefer combined names over individual words - prioritize multi-word names
        if combined_final:
            # Sort by word count (prefer full names), then confidence
            combined_final.sort(key=lambda b: (len(b.text.split()), b.confidence), reverse=True)
            print(f"[OCR] Combined {len(final_boxes)} boxes into {len(combined_final)} names")
            for b in combined_final[:3]:
                print(f"[OCR]   Combined: '{b.text}' conf={b.confidence:.1f}% (words={len(b.text.split())})")
        
        final_result = combined_final if combined_final else final_boxes
        
        # Sort final results by word count (prefer full names) then confidence
        final_result.sort(key=lambda b: (len(b.text.split()), b.confidence), reverse=True)
        
        if final_result:
            print(f"[OCR] Returning {len(final_result)} final boxes (sorted by word count)")
            for b in final_result[:5]:
                print(f"[OCR] Final: '{b.text}' conf={b.confidence:.1f}% (words={len(b.text.split())})")
        else:
            print(f"[OCR] WARNING: No final boxes returned after filtering! Had {len(deduplicated)} deduplicated boxes")
            # Emergency fallback: try to extract valid names from ALL detected text using sophisticated extraction
            emergency_boxes = []
            seen_names = set()
            
            # Use the sophisticated _extract_proper_names method on ALL deduplicated boxes
            for box in deduplicated:
                text = box.text.strip()
                if not text:
                    continue
                
                # Try to extract proper names using the sophisticated extraction method
                extracted_names = self._extract_proper_names(text)
                for name, conf in extracted_names:
                    if name and name not in seen_names and len(name) >= 3:
                        seen_names.add(name)
                        # Create a new box for this extracted name with boosted confidence
                        emergency_boxes.append(OCRBox(
                            text=name,
                            box=box.box,
                            confidence=max(box.confidence, conf * 100, 60.0)  # Use highest confidence
                        ))
                        print(f"[OCR] Emergency fallback: extracted '{name}' from '{text}' (conf={max(box.confidence, conf * 100, 60.0):.1f}%)")
            
            if emergency_boxes:
                print(f"[OCR] Emergency fallback returning {len(emergency_boxes)} names: {[b.text for b in emergency_boxes]}")
                return emergency_boxes
        
        return final_result

    def _run_ocr_on_roi(self, roi_bgr, offset: Tuple[int, int], debug_idx: int = 0) -> List[OCRBox]:
        """
        Enhanced OCR for Zoom name labels with multiple preprocessing strategies.
        Tries aggressive preprocessing to handle white-on-dark text.
        """
        all_boxes = []
        
        if roi_bgr.size == 0 or roi_bgr.shape[0] < 10 or roi_bgr.shape[1] < 10:
            return []
        
        # Save first ROI for debugging
        if debug_idx == 0:
            try:
                debug_path = f"debug_roi_original_{offset[0]}_{offset[1]}.png"
                cv2.imwrite(debug_path, roi_bgr)
                print(f"[OCR] Saved original ROI to {debug_path}")
            except Exception:
                pass
        
        # Convert to grayscale
        gray = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)
        
        # Strategy: Try multiple preprocessing approaches for white-on-dark text
        
        # 1. Aggressive brightening + CLAHE
        brightened1 = cv2.convertScaleAbs(gray, alpha=2.0, beta=50)
        clahe1 = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        enhanced1 = clahe1.apply(brightened1)
        
        # 2. Very aggressive brightening
        brightened2 = cv2.convertScaleAbs(gray, alpha=3.0, beta=80)
        clahe2 = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8, 8))
        enhanced2 = clahe2.apply(brightened2)
        
        # 3. Inverted + enhanced (for dark text on light background)
        inverted = cv2.bitwise_not(gray)
        inverted_enhanced = cv2.convertScaleAbs(inverted, alpha=1.5, beta=30)
        
        # Create thresholded versions
        _, thresh1 = cv2.threshold(enhanced1, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        _, thresh2 = cv2.threshold(enhanced2, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        _, thresh3 = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        _, thresh_inv = cv2.threshold(inverted_enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Manual threshold for white text (pixels > 180)
        _, thresh_manual = cv2.threshold(enhanced2, 180, 255, cv2.THRESH_BINARY)
        
        # Adaptive thresholds
        adapt1 = cv2.adaptiveThreshold(enhanced1, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 15, 10)
        adapt2 = cv2.adaptiveThreshold(enhanced2, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 15, 10)
        
        # Try RAW GRAYSCALE FIRST (simplest - sometimes works best)
        # Then try multiple preprocessing variants with different scales and PSM modes
        variants = [
            (gray, 3.0, "7"),         # Raw grayscale, 3x scale (TRY FIRST - no preprocessing)
            (gray, 4.0, "7"),         # Raw grayscale, 4x scale
            (enhanced2, 3.0, "7"),    # Enhanced without threshold, 3x
            (thresh_manual, 4.0, "7"), # Manual threshold (catches white text)
            (thresh2, 4.0, "7"),      # Very enhanced + Otsu
            (enhanced2, 4.0, "7"),    # Enhanced without threshold, 4x
            (adapt2, 4.0, "7"),       # Adaptive threshold
            (thresh1, 4.0, "7"),      # Enhanced + Otsu
            (gray, 3.0, "8"),         # Raw, 3x, PSM 8 (single word)
            (thresh_inv, 4.0, "7"),   # Inverted (fallback)
            (thresh_manual, 5.0, "7"), # Manual threshold, 5x scale
        ]
        
        for processed_img, scale, psm in variants:
            try:
                # Scale up for better OCR
                if scale > 1.0:
                    scaled = cv2.resize(processed_img, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
                else:
                    scaled = processed_img
                
                # Save scaled image for debugging (first variant only)
                if debug_idx == 0 and variants.index((processed_img, scale, psm)) == 0:
                    try:
                        debug_path = f"debug_roi_scaled_{offset[0]}_{offset[1]}.png"
                        cv2.imwrite(debug_path, scaled)
                        print(f"[OCR] Saved scaled ROI to {debug_path}")
                    except Exception:
                        pass
                
                # Remove whitelist constraint - let Tesseract detect any text
                config_str = f"--oem 3 --psm {psm}"
                
                # Try image_to_data first
                try:
                    data = pytesseract.image_to_data(
                        scaled, output_type=pytesseract.Output.DICT, 
                        config=config_str, timeout=self._ocr_timeout
                    )
                    
                    # Log what Tesseract returned (for debugging)
                    word_count = sum(1 for t in data.get("text", []) if t.strip())
                    if word_count > 0:
                        print(f"[OCR] Tesseract returned {word_count} word(s) for scale={scale} psm={psm}")
                        for i, txt in enumerate(data.get("text", [])):
                            if txt.strip() and i < 5:  # Log first 5
                                try:
                                    conf = float(data.get("conf", [0])[i])
                                    print(f"[OCR]   Word {i+1}: '{txt}' conf={conf:.1f}%")
                                except:
                                    print(f"[OCR]   Word {i+1}: '{txt}'")
                    
                    boxes = self._extract_boxes_from_data(data, scaled, roi_bgr, offset, scale)
                    if boxes:
                        print(f"[OCR] Extracted {len(boxes)} boxes after filtering")
                        all_boxes.extend(boxes)
                        if any(b.confidence > 40 and len(b.text) >= 2 for b in boxes):
                            return self._deduplicate_boxes(all_boxes)
                except Exception as e:
                    print(f"[OCR] Error in image_to_data: {e}")
                    pass
                
                # Try image_to_string as fallback
                try:
                    text = pytesseract.image_to_string(scaled, config=config_str, timeout=self._ocr_timeout)
                    text = text.strip()
                    if text and len(text) >= 2 and any(c.isalpha() for c in text):
                        cleaned = self._clean_text(text)
                        if cleaned and len(cleaned) >= 2:
                            box = OCRBox(
                                text=cleaned,
                                box=(offset[0], offset[1], roi_bgr.shape[1], roi_bgr.shape[0]),
                                confidence=50.0
                            )
                            all_boxes.append(box)
                            print(f"[OCR] Found via image_to_string: '{cleaned}'")
                            if len(cleaned) >= 3:
                                return self._deduplicate_boxes(all_boxes)
                except Exception:
                    pass
                    
            except Exception:
                continue
        
        return self._deduplicate_boxes(all_boxes)
    
    def _extract_boxes_from_data(self, data, scaled_img, roi_bgr, offset, scale_factor) -> List[OCRBox]:
        boxes: List[OCRBox] = []
        n_boxes = len(data["text"])
        sx = (roi_bgr.shape[1] / scaled_img.shape[1]) if scaled_img.shape[1] else 1.0
        sy = (roi_bgr.shape[0] / scaled_img.shape[0]) if scaled_img.shape[0] else 1.0
        
        for i in range(n_boxes):
            raw_text = data["text"][i].strip()
            if not raw_text:
                continue
            
            # Log raw text for debugging (first few)
            if len(boxes) < 3:
                print(f"[OCR] Raw text from Tesseract: '{raw_text}'")
            
            text = self._clean_text(raw_text)
            
            # Very relaxed filtering - accept any text with 2+ characters
            if not text or len(text) < 2:
                continue
            
            # REJECT blocked UI words early (case-insensitive)
            text_lower = text.lower()
            if text_lower in self._blocked_tokens:
                continue  # Skip UI words like "react", "info", "more"
            
            # Must have at least one letter
            if not any(c.isalpha() for c in text):
                continue
            
            try:
                conf = float(data["conf"][i])
            except Exception:
                conf = 0.0
            
            # Accept any confidence above 5 (very low threshold)
            if conf < 5:
                continue
            
            # Log accepted text
            print(f"[OCR] Accepting text: '{text}' conf={conf:.1f}% (raw: '{raw_text}')")
            
            x = int(data["left"][i] * sx) + offset[0]
            y = int(data["top"][i] * sy) + offset[1]
            w = int(data["width"][i] * sx)
            h = int(data["height"][i] * sy)
            
            boxes.append(OCRBox(text=text, box=(x, y, w, h), confidence=conf))
        
        # Try to combine nearby words that might be part of the same name
        combined_boxes = self._combine_nearby_words(boxes)
        return combined_boxes

    def _scan_bottom_strips(
        self,
        frame_bgr: np.ndarray,
        face_boxes: Sequence[Tuple[int, int, int, int]] | None,
        segments: int | None = None,
    ) -> List[OCRBox]:
        """
        Scan broad strips along the bottom of the screen to catch Zoom name tags
        even when face bounding boxes are misaligned (e.g., due to screen recursion).
        """
        h, w = frame_bgr.shape[:2]
        strip_height = max(90, int(h * 0.18))
        y1 = max(0, h - strip_height)
        y2 = h

        if face_boxes:
            anchors = [min(w - 1, max(0, x + bw // 2)) for x, _, bw, _ in face_boxes]
            anchors = sorted(anchors)
        else:
            segs = segments or 6
            anchors = [int((i + 0.5) * w / segs) for i in range(segs)]

        window_width = max(220, int(w * 0.12))
        results: List[OCRBox] = []
        for anchor in anchors:
            rx = max(0, anchor - window_width // 2 - 40)
            rx2 = min(w, rx + window_width + 80)
            if rx2 <= rx:
                continue
            roi = frame_bgr[y1:y2, rx:rx2]
            boxes = self._run_ocr_on_roi(roi, offset=(rx, y1))
            results.extend(boxes)
        return results
    
    def _combine_nearby_words(self, boxes: List[OCRBox]) -> List[OCRBox]:
        """
        Combine nearby OCR boxes that might be parts of the same name.
        Handles both horizontal (same line) and vertical (stacked) layouts.
        Zoom names can be on one line (e.g., "John Doe") or stacked vertically.
        """
        if not boxes:
            return []
        
        # Sort by y-coordinate first (top to bottom), then x (left to right)
        sorted_boxes = sorted(boxes, key=lambda b: (b.box[1], b.box[0]))
        
        combined = []
        used = set()
        i = 0
        while i < len(sorted_boxes):
            current = sorted_boxes[i]
            combined_text = [current.text]
            combined_boxes_list = [current]
            
            # Look ahead for nearby boxes on the same line
            j = i + 1
            while j < len(sorted_boxes):
                next_box = sorted_boxes[j]
                cx, cy, cw, ch = current.box
                nx, ny, nw, nh = next_box.box
                
                # Check if boxes are on similar y-level (same line)
                y_diff = abs((cy + ch/2) - (ny + nh/2))
                x_gap = nx - (cx + cw)
                
                # More aggressive combination: allow up to 150px gap for multi-word names
                # This helps catch multi-word names even if they're separated
                max_gap = max(150, max(cw, nw) * 3)  # Allow larger gaps for full names
                y_overlap_threshold = max(ch, nh) * 0.7  # 70% overlap tolerance - same line
                
                # Also check for vertical stacking (names on two lines)
                vertical_gap = ny - (cy + ch)
                vertical_overlap_threshold = max(ch, nh) * 0.3  # Names can be stacked
                
                # Combine if: same line with horizontal gap OR vertically stacked
                is_horizontal_match = (y_diff < y_overlap_threshold and -10 < x_gap < max_gap)
                is_vertical_match = (vertical_gap >= 0 and vertical_gap < 50 and abs(cx - nx) < max(cw, nw) * 0.7)
                
                # Also check for vertically stacked names (common in Zoom - first name above last name)
                # Names can be on separate lines but vertically aligned
                vertical_align_tolerance = max(cw, nw) * 0.8
                is_stacked_name = (
                    vertical_gap >= 0 and vertical_gap < 100 and  # Allow up to 100px vertical gap for stacked names
                    abs((cx + cw/2) - (nx + nw/2)) < vertical_align_tolerance
                )
                
                # Prioritize combining words that look like names (capitalized, proper words)
                looks_like_name_part = (
                    (current.text[0].isupper() and next_box.text[0].isupper()) or  # Both capitalized
                    len(current.text) >= 4 or len(next_box.text) >= 4  # Or at least 4 chars
                )
                
                # Combine if they're nearby - be more aggressive for name combination
                # Don't require looks_like_name_part for vertical stacking (combine anyway if aligned)
                should_combine = (
                    (is_horizontal_match and looks_like_name_part) or
                    is_vertical_match or
                    (is_stacked_name and (looks_like_name_part or len(current.text) >= 3 or len(next_box.text) >= 3))
                )
                
                if should_combine:
                    combined_text.append(next_box.text)
                    combined_boxes_list.append(next_box)
                    # Update current box to encompass all combined boxes
                    combined_x = min(cx, nx)
                    combined_y = min(cy, ny)
                    combined_w = max(cx + cw, nx + nw) - combined_x
                    combined_h = max(cy + ch, ny + nh) - combined_y
                    # Use average confidence (or max, both work)
                    avg_conf = sum(b.confidence for b in combined_boxes_list) / len(combined_boxes_list)
                    current = OCRBox(
                        text=" ".join(combined_text),
                        box=(combined_x, combined_y, combined_w, combined_h),
                        confidence=avg_conf
                    )
                    j += 1
                else:
                    break
            
            combined.append(current)
            i = j
        
        # Log combined results for debugging
        if combined and len(combined) < len(boxes):
            print(f"[OCR] Combined {len(boxes)} boxes into {len(combined)} names")
            for cb in combined:
                print(f"[OCR]   Combined name: '{cb.text}' conf={cb.confidence:.1f}%")
        
        return combined
    
    def _deduplicate_boxes(self, boxes: List[OCRBox]) -> List[OCRBox]:
        """Keep only the highest confidence version of overlapping boxes."""
        if not boxes:
            return []
        
        # Sort by confidence descending
        boxes.sort(key=lambda b: b.confidence, reverse=True)
        
        result = []
        for box in boxes:
            # Check if this box overlaps significantly with any already added
            overlaps = False
            for existing in result:
                if self._boxes_overlap(box, existing, threshold=0.5):
                    overlaps = True
                    break
            if not overlaps:
                result.append(box)
        
        return result
    
    @staticmethod
    def _boxes_overlap(box1: OCRBox, box2: OCRBox, threshold: float = 0.5) -> bool:
        """Check if two boxes overlap by more than threshold."""
        x1, y1, w1, h1 = box1.box
        x2, y2, w2, h2 = box2.box
        
        # Calculate intersection
        xi1 = max(x1, x2)
        yi1 = max(y1, y2)
        xi2 = min(x1 + w1, x2 + w2)
        yi2 = min(y1 + h1, y2 + h2)
        
        if xi2 <= xi1 or yi2 <= yi1:
            return False
        
        inter_area = (xi2 - xi1) * (yi2 - yi1)
        box1_area = w1 * h1
        box2_area = w2 * h2
        
        # Check if intersection is significant relative to either box
        overlap1 = inter_area / box1_area if box1_area > 0 else 0
        overlap2 = inter_area / box2_area if box2_area > 0 else 0
        
        return overlap1 > threshold or overlap2 > threshold

    def _clean_text(self, text: str) -> str:
        """Clean and validate text as a potential name."""
        cleaned = "".join(ch for ch in text if ch in self._allowed_chars)
        cleaned = cleaned.strip(" .-_")
        
        # Remove common OCR artifacts
        cleaned = cleaned.replace("|", "").replace("_", "").replace("~", "").replace("'", "")
        
        # Filter out very short fragments (at least 3 characters for names)
        if len(cleaned) < 3:
            return ""
        
        # Must contain at least one letter
        if not any(c.isalpha() for c in cleaned):
            return ""
        
        lowered = cleaned.lower()
        if lowered in self._blocked_tokens:
            return ""
        
        # Reject very short single words that are likely fragments
        words = cleaned.split()
        if len(words) == 1 and len(cleaned) < 4:
            # Single word must be at least 4 chars (like "John", "Mary")
            return ""
        
        # Filter out common UI fragments
        ui_fragments = {"ig", "es", "vt", "pa", "mp", "ly", "ch", "re", "ad", "on", "of", "to", 
                       "in", "is", "at", "as", "an", "am", "we", "or", "it", "if", "up", "do",
                       "pm", "am", "q", "a", "i", "q search", "search"}
        if lowered in ui_fragments:
            return ""
        
        # Names should have proper capitalization (first letter uppercase) or be all caps
        # Accept if it has uppercase letter(s) at start or is all uppercase
        # BUT: be more lenient - accept if it's a known proper name pattern
        has_capital = cleaned[0].isupper() or cleaned.isupper() or any(c.isupper() for c in cleaned)
        if not has_capital and len(cleaned) < 5:
            # If all lowercase and short, likely not a name label
            return ""
        
        # Excessively long strings are likely UI labels, not names
        if len(cleaned) > 30:
            return ""
        
        return cleaned
    
    def _extract_proper_names(self, text: str) -> List[Tuple[str, float]]:
        """
        Extract proper names from mixed/contaminated OCR text.
        Example: "t . o- Jain        A ee  z" -> [("Jain", 0.9)]
        Looks for capitalized words that look like proper names.
        """
        if not text:
            return []
        
        # Clean the text first - remove excessive spaces and symbols
        cleaned = " ".join(text.split())  # Normalize whitespace
        # Remove excessive punctuation
        cleaned = re.sub(r'[^\w\s-]', ' ', cleaned)
        cleaned = " ".join(cleaned.split())
        
        # Split into words and look for proper name patterns
        words = cleaned.split()
        extracted = []
        
        # Look for capitalized words that are proper names
        for word in words:
            word = word.strip(" .-_,;:=")
            if not word:
                continue
            
            # Skip single letters, very short words, and common words
            if len(word) < 3:
                continue
            
            # Check if it looks like a proper name (capitalized, reasonable length)
            if word[0].isupper() and len(word) >= 3:
                # Check if it's mostly letters (not symbols/numbers)
                alpha_ratio = sum(c.isalpha() for c in word) / len(word) if len(word) > 0 else 0
                if alpha_ratio >= 0.8:  # At least 80% letters
                    # Skip common UI words
                    lower_word = word.lower()
                    ui_words = {"or", "on", "of", "to", "in", "is", "it", "if", "at", "as", 
                               "an", "am", "we", "up", "do", "pm", "am", "ee", "ae", "oe",
                               "pp", "hy", "bs", "ft", "vt", "mp", "ly", "ch", "re", "ad", 
                               "ig", "es", "pa", "a", "i", "o", "e", "z", "x", "y"}
                    if lower_word not in ui_words and len(word) >= 4:
                        extracted.append((word, 0.85))  # High confidence for extracted names
        
        # Also try to combine consecutive capitalized words (e.g., "Abhishek Jain")
        if len(extracted) > 1:
            # Look for pairs of capitalized words that might be full names
            combined = []
            i = 0
            while i < len(extracted) - 1:
                name1, conf1 = extracted[i]
                name2, conf2 = extracted[i + 1]
                # Combine if both are proper names (capitalized, 4+ chars)
                if len(name1) >= 4 and len(name2) >= 4:
                    combined_name = f"{name1} {name2}"
                    combined_conf = (conf1 + conf2) / 2
                    combined.append((combined_name, combined_conf))
                    i += 2  # Skip next since we combined it
                else:
                    combined.append((name1, conf1))
                    i += 1
            
            # Add remaining
            if i < len(extracted):
                combined.append(extracted[i])
            
            return combined if combined else extracted
        
        return extracted
    
    def _reconstruct_names_from_fragments(self, all_words: List[Tuple[List[str], Tuple[int, int, int, int], float]]) -> List[Tuple[str, Tuple[int, int, int, int], float]]:
        """
        Reconstruct full names from fragments.
        Example: "John" + "Doe" -> "John Doe"
        Looks for common name patterns and combines fragments intelligently.
        """
        if not all_words:
            return []
        
        reconstructed = []
        
        # Common name endings that might be split from the main name
        name_endings = ["ek", "hek", "ish", "esh", "kumar", "singh", "patel"]
        
        # Look for patterns like fragment + proper name (e.g., "hek" + "Jain")
        for i, (words1, box1, conf1) in enumerate(all_words):
            for j, (words2, box2, conf2) in enumerate(all_words[i+1:], start=i+1):
                if not words1 or not words2:
                    continue
                
                # Check if last word of first is a name ending and first word of second is a proper name
                last_word1 = words1[-1].lower()
                first_word2 = words2[0] if words2 else ""
                
                # Pattern: name fragment ending + proper name (e.g., "John" + "Doe")
                if (last_word1 in name_endings or 
                    (len(last_word1) >= 3 and last_word1.endswith(('ek', 'ish', 'esh', 'k', 'h')))):
                    if (first_word2 and 
                        len(first_word2) >= 4 and 
                        first_word2[0].isupper() and
                        first_word2.isalpha()):
                        # Try to reconstruct: "John" + "Doe" -> "John Doe" (can be improved later)
                        combined = f"{words1[-1]} {first_word2}"
                        if len(combined) >= 6:  # Ensure it's long enough
                            # Capitalize first letter of combined name
                            combined = combined[0].upper() + combined[1:] if len(combined) > 1 else combined
                            reconstructed.append((combined, box1, (conf1 + conf2) / 2))
                            print(f"[OCR] Reconstructing: '{words1[-1]}' + '{first_word2}' -> '{combined}'")
        
        return reconstructed
    
    def _is_valid_name(self, text: str) -> bool:
        """
        Validate if text looks like a real name.
        Rejects garbage OCR results, single letters, and UI elements.
        """
        if not text or len(text) < 2:
            return False
        
        # Clean the text
        cleaned = self._clean_text(text)
        if not cleaned or len(cleaned) < 3:
            return False
        
        words = cleaned.split()
        
        # Reject text with too many single-letter or very short words (garbage OCR)
        # Pattern like "o or o i. a" has many short words
        short_words = sum(1 for w in words if len(w) <= 2)
        if len(words) > 0:
            # Reject if more than 40% are short words OR if there are 3+ short words in a row
            if short_words > len(words) * 0.4:  # More than 40% short words = garbage
                return False
            # Also reject patterns like "o or o i a" (multiple short words)
            if short_words >= 3:
                return False
        
        # Reject if it has too many symbols mixed with letters (like "o or o i. a")
        non_alpha_chars = sum(1 for c in cleaned if not c.isalpha() and c != ' ')
        if len(cleaned) > 0 and non_alpha_chars / len(cleaned) > 0.3:  # More than 30% symbols = likely garbage
            return False
        
        # Must have at least one word that's 4+ characters and capitalized
        has_proper_word = any(
            len(w) >= 4 and (w[0].isupper() or w.isupper())
            for w in words
        )
        
        # Or if it's a single word, must be 4+ chars and capitalized
        if len(words) == 1:
            word = words[0]
            if len(word) < 4:
                return False
            if not (word[0].isupper() or word.isupper()):
                return False
        
        # Reject common UI fragments
        text_lower = cleaned.lower()
        ui_fragments = {
            "or", "on", "of", "to", "in", "is", "it", "if", "at", "as", "an", "am", "we", "up", "do",
            "pm", "am", "q", "a", "i", "o", "e", "z", "x", "y", "search", "react", "chat", "mute",
            "view", "share", "settings", "button", "click", "press", "menu", "ee", "ae", "oe",
            "pp", "oo", "wi", "co", "bs", "ft", "vn", "hy", "ig", "es", "vt", "pa", "mp", "ly",
            "q search"
        }
        if text_lower in ui_fragments:
            return False
        
        # Reject patterns that are clearly garbage (like "o or o i. a")
        # Pattern: multiple single-letter words or very short words
        if re.search(r'^[a-z](\s+[a-z]){3,}', text_lower):  # Pattern: "o or o i a"
            return False
        
        # Reject if it has multiple single letters separated by spaces (like "o or o i a")
        single_letter_words = sum(1 for w in words if len(w) == 1)
        if single_letter_words >= 2:  # Two or more single-letter words = likely garbage
            return False
        
        # For multi-word text, must have at least one proper word (4+ chars, capitalized)
        # For single-word text, must be 4+ chars and capitalized
        if not has_proper_word:
            return False
        
        return True

