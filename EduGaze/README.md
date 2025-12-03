## EduGaze – Multi-Student Engagement Analyzer for Zoom (Windows Desktop App)

EduGaze is a proof-of-concept, **fully local** Windows desktop application that:

- **Captures** your screen (e.g. Zoom gallery view)
- **Detects multiple student faces** in real time (MediaPipe)
- **Reads Zoom name labels** (Tesseract OCR)
- **Tracks each student** across frames
- **Runs emotion recognition** every 5 seconds
- **Computes engagement scores** per student
- **Logs metrics** to `engagement_log.csv`
- **Exports a final engagement report** to `final_report.xlsx`

No video is recorded and **no data is sent to the cloud**; all processing happens on-device.

---

## 1. Setup

### 1.1. Prerequisites

- **OS**: Windows 10 or later (64-bit)
- **Python**: 3.10 or 3.11 (64-bit recommended)
- **Hardware**: A GPU is not required, but a reasonably fast CPU is recommended.

### 1.2. Install system dependencies

#### Install Tesseract OCR (Windows)

1. Download the Tesseract Windows installer from the official repo or UB Mannheim builds  
   (for example: `https://github.com/UB-Mannheim/tesseract/wiki`).
2. Run the installer and remember the install path, e.g.  
   `C:\Program Files\Tesseract-OCR\tesseract.exe`
3. Add that folder to your **PATH** (recommended) or set the `TESSERACT_CMD` environment variable:

   ```powershell
   setx TESSERACT_CMD "C:\Program Files\Tesseract-OCR\tesseract.exe"
   ```

   The app auto-detects the common `C:\Program Files\Tesseract-OCR\` and `C:\Program Files (x86)\Tesseract-OCR\` locations, so if you installed there no extra steps are usually required.

### 1.3. Create a virtual environment and install Python dependencies

From the project root:

```bash
python -m venv .venv
.venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
```

> If detection still fails, set `TESSERACT_CMD` or edit `core/ocr.py` and pass your explicit path to
> `ZoomNameOCR(tesseract_cmd="C:\\Program Files\\Tesseract-OCR\\tesseract.exe")`.

---

## 2. Running EduGaze

1. Start your Zoom meeting and arrange the **gallery view** on screen.
2. Activate your Python virtual environment (if not already):

   ```bash
   .venv\Scripts\activate
   ```

3. From the project root, run:

   ```bash
   python main.py
   ```

4. In the EduGaze window:
   - **Log path**: choose where to save `engagement_log.csv`
   - Click **Start** to begin analysis
   - A live preview of the captured screen will appear
   - The right panel lists **student IDs, names, emotion, and engagement**
   - Click **Stop** to end the session and finalize logs

> For this PoC, screen capture defaults to the **primary monitor**. You should place
> the Zoom window (gallery view) on that monitor.

---

## 3. Architecture Overview

The project is organized into modular components:

- **`main.py`**
  - Application entry point. Creates and shows the PyQt5 `MainWindow`.

- **`ui/main_window.py`**
  - Modern PyQt5 GUI with a dark theme:
    - Live preview card that now always mirrors the full primary monitor
    - Session status pill + Start/Stop controls, log-path selector
    - Student leaderboard tab (ranked by engagement) and sparkline-style graphs tab
  - Spawns a background thread that continuously calls the processing pipeline.

- **`core/screen_capture.py`**
  - Uses `mss` to capture the primary monitor at high FPS.
  - Always records the full primary monitor to keep name/face detection stable (no region picker).

- **`core/face_detector.py`**
  - Uses `mediapipe.face_detection` for fast face detection.
  - Outputs pixel-space bounding boxes and detection confidences.

- **`core/ocr.py`**
  - `ZoomNameOCR` auto-detects `tesseract.exe` on Windows, tries multiple preprocessing strategies for dark Zoom labels.
  - **Multi-strategy OCR**: Tries multiple preprocessing variants (brightening, Otsu thresholding, adaptive thresholding, inverted versions) to handle Zoom's dark label overlays with white text.
  - **Focused ROI detection**: Extracts precise bottom-left regions where Zoom names appear (bottom 15% of each face tile).
  - **Text filtering**: Filters out noise, requires mostly alphabetic text, deduplicates overlapping detections.
  - **Visual feedback**: Preview shows orange boxes (OCR search regions) and yellow boxes (detected text) in real-time.
  - Returns name candidates as `OCRBox(text, box, confidence)`.

- **`core/tracker.py`**
  - Simple **CentroidTracker** providing persistent IDs for faces over time.
  - Matches current detections to existing tracks based on nearest centroid distance.

- **`core/emotion.py`**
  - Wraps the **`fer`** library for CNN-based emotion recognition.
  - Accepts a face bounding box and:
    - Crops and converts to RGB
    - Runs the model
    - Returns a dominant emotion plus per-emotion confidence scores
  - Supported emotions (after mapping): `happy`, `neutral`, `sad`, `angry`, `surprised`.

- **`core/engagement.py`**
  - Multimodal scoring pipeline:
    - **Eye focus** = MediaPipe detection confidence.
    - **Emotion** = FER label × FER confidence.
    - **Attention** = head orientation + face-area ratio (is the student centered/close?).
    - **Motion energy** = optical-flow magnitude inside the face region (is the student actively reacting?).
    - **Brightness** = exposure heuristics to penalize frames that are too dark/bright.
  - Final score blends all cues: `0.35*EyeFocus + 0.25*Emotion + 0.20*Attention + 0.15*Motion + 0.05*Brightness`.

- **`core/logger.py`**
  - `EngagementLogger` appends a multimodal snapshot for each student every 5 seconds:
    - identity + engagement (`timestamp`, `student_id`, `student_name`, `engagement_score`)
    - perception metrics (`emotion`, `emotion_confidence`, `raw_emotion_confidences`)
    - attention cues (`face_bbox`, `face_area_ratio`, `detection_confidence`, `attention_score`, `motion_level`, `brightness`, `ocr_confidence`)
    - face-mesh metrics (`gaze_x`, `gaze_y`, `blink_rate`, `head_yaw`, `head_pitch`, `head_roll`)
  - Writes to `engagement_log.csv` and transparently spills to a temp queue if the CSV is locked (e.g., opened in Excel).

- **`core/report.py`**
  - Loads the CSV into a `pandas` DataFrame.
  - Produces:
    - Per-student **average engagement**
    - **Highest** and **lowest** engaged students
  - Exports `final_report.xlsx`:
    - `Raw Log` sheet
    - `Per-Student Averages` sheet
    - `Summary` sheet (highest/lowest engagement).

- **`core/pipeline.py`**
  - Central orchestration:
    - Captures frame (`ScreenCapturer`)
    - Detects faces (`FaceDetector`)
    - Tracks them (`CentroidTracker`)
    - OCR for name labels (`ZoomNameOCR`)
    - Face–name association (closest OCR box below each face)
    - Emotion recognition (`EmotionRecognizer`) every **5 seconds**
    - Computes optical-flow motion maps + brightness to feed multimodal engagement
    - Engagement computation (`engagement_score`)
    - Logging via `EngagementLogger`
    - Overlays bounding boxes and labels for preview

- **`core/types.py`**
  - Shared data structures like `StudentSnapshot` (ID, name, last emotion, engagement, bbox, OCR confidence).

---

## 4. Data Logging & Final Report

- **Sampling interval**
  - Face detection runs continuously in the capture loop.
  - Emotion recognition and engagement computation run **every 5 seconds**.
  - At each 5-second tick, all currently tracked students are logged.

- **Log file**
  - Path is chosen in the UI (default: `engagement_log.csv` in project root).
  - Stores **metrics only**—no screenshots or raw images are saved.
  - Columns now include detection confidence, face area ratio, attention, motion level, brightness, and the emotion confidence used for scoring—useful for auditing the multimodal ranking.

- **Final report**
  - You can post-process the CSV into a final Excel report.
  - Easiest: use the helper script:

    ```bash
    python generate_report.py --csv engagement_log.csv --out final_report.xlsx
    ```

    (run this from the project root with your venv activated)

  - This produces:
    - Per-student average engagement
    - Highest and lowest engagement students
    - Full time-series log in the Excel file.

---

## 5. Limitations & Notes

- **Screen capture**
  - EduGaze always captures the **primary monitor**. Keep the Zoom meeting on that display.
  - A future iteration could reintroduce a region picker once the full-screen pipeline is fully vetted.

- **OCR robustness**
  - OCR accuracy depends heavily on:
    - Zoom UI scale
    - Screen resolution
    - Name label visibility
  - The pipeline uses simple preprocessing; additional fine-tuning (e.g. custom ROI for bottom of screen) will improve results.

- **Emotion model**
  - The `fer` library ships a small CNN model trained on FER2013.
  - Accuracy can vary depending on lighting, occlusion, and camera quality.

- **Head orientation & blink rate**
  - Head orientation is approximated using bounding box aspect ratio.
  - Blink rate and detailed head direction are placeholders and **not implemented** in this PoC.

- **Privacy**
  - No raw images or video are stored.
  - All inference runs locally on your machine.

---

## 6. Advanced Features (Now Implemented)

### 6.1. Full-Screen Capture (Default)
- EduGaze now always records the entire primary monitor for maximum reliability.
- Place the Zoom gallery on your main screen before clicking **Start**.
- Removes the fragile region picker and guarantees consistent OCR + tracking.

### 6.2. Enhanced Tracking
- Improved **CentroidTracker** with better matching algorithms
- More stable IDs across frames with velocity-based prediction
- Handles up to 25 faces simultaneously (Zoom's maximum)

### 6.3. Face Mesh Analysis (MediaPipe)
- **Gaze Estimation**: Tracks where students are looking (gaze_x, gaze_y)
- **Blink Rate**: Monitors eye openness (0=closed, 1=open) for alertness
- **Head Pose**: Tracks head orientation (yaw, pitch, roll angles)
- Integrated into engagement scoring for more accurate metrics

### 6.4. Time-Series Graphs
- **"Graphs" tab** in the GUI shows real-time engagement trends
- Line charts for each student showing engagement over time
- Updates every 2 seconds with the last 100 data points
- Color-coded by student name

### 6.5. Automatic Excel Report Generation
- Excel report is **automatically generated** when you click **Stop**
- Saved as `{log_name}_report.xlsx` next to your CSV file
- Includes:
  - Raw log data
  - Per-student average engagement
  - Summary statistics (highest/lowest engagement)
- Shows a notification dialog when complete

---

## 7. Extending Further

- Integrate DeepSORT for even more stable tracking
- Add multiple camera support
- Implement cloud export (optional, privacy-preserving)
- Add support for other video conferencing platforms (Teams, Google Meet)


