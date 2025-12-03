import os
import threading
import time
from typing import Dict, List, Optional, Tuple
from collections import defaultdict

import cv2
import numpy as np
from PyQt5 import QtCore, QtGui, QtWidgets
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

from core.pipeline import ProcessingPipeline
from core.types import StudentSnapshot
from core.report import ReportGenerator
from core.logger import EngagementLogger
from ui.window_picker import WindowPickerDialog


class VideoLabel(QtWidgets.QLabel):
    """Simple QLabel subclass to preserve aspect ratio scaling."""

    def setImage(self, image: QtGui.QImage):
        if image is None or image.isNull():
            print("[VideoLabel] Received null or invalid QImage")
            return
        try:
            pixmap = QtGui.QPixmap.fromImage(image)
            if pixmap.isNull():
                print("[VideoLabel] Failed to create QPixmap from QImage")
                return
            scaled = pixmap.scaled(self.size(), QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation)
            self.setPixmap(scaled)
            # Force update
            self.update()
        except Exception as e:
            print(f"[VideoLabel] Error in setImage: {e}")
            import traceback
            traceback.print_exc()


class EngagementGraph(FigureCanvas):
    """Matplotlib widget for time-series engagement graphs."""

    def __init__(self, parent=None):
        self.fig = Figure(figsize=(6, 4), dpi=100)
        super().__init__(self.fig)
        self.setParent(parent)
        self.axes = self.fig.add_subplot(111)
        self.axes.set_xlabel("Time")
        self.axes.set_ylabel("Engagement Score")
        self.axes.set_title("Engagement Over Time")
        self.axes.set_ylim(0, 1)
        self.fig.tight_layout()
        
        # Data storage
        self.times: List[float] = []
        self.data_series: Dict[int, List[float]] = defaultdict(list)  # student_id -> scores
        
    def update_graph(self, current_time: float, students: Dict[int, StudentSnapshot]):
        """Update graph with latest data."""
        self.times.append(current_time)
        
        for sid, student in students.items():
            if student.last_engagement is not None:
                self.data_series[sid].append(student.last_engagement)
            else:
                # Extend last value or use 0
                if self.data_series[sid]:
                    self.data_series[sid].append(self.data_series[sid][-1])
                else:
                    self.data_series[sid].append(0.0)
        
        # Keep only last 100 points for performance
        max_points = 100
        if len(self.times) > max_points:
            self.times = self.times[-max_points:]
            for sid in self.data_series:
                self.data_series[sid] = self.data_series[sid][-max_points:]
        
        # Clear and redraw
        self.axes.clear()
        self.axes.set_xlabel("Time (seconds)")
        self.axes.set_ylabel("Engagement Score")
        self.axes.set_title("Engagement Over Time")
        self.axes.set_ylim(0, 1)
        
        # Plot each student
        for sid, scores in self.data_series.items():
            if len(scores) == len(self.times):
                student_name = students.get(sid, StudentSnapshot(id=sid, name=f"Student-{sid}")).name
                # Normalize time to start from 0
                times_norm = [t - self.times[0] if self.times else t for t in self.times]
                self.axes.plot(times_norm, scores, label=student_name, linewidth=2)
        
        if self.data_series:
            self.axes.legend(loc='upper right', fontsize=8)
        
        self.axes.grid(True, alpha=0.3)
        self.fig.tight_layout()
        self.draw()


class MainWindow(QtWidgets.QMainWindow):
    frame_updated = QtCore.pyqtSignal(object)
    stats_updated = QtCore.pyqtSignal(dict)

    def __init__(self):
        super().__init__()
        self.setWindowTitle("EduGaze – Multi-Student Engagement Analyzer")
        self.resize(1400, 900)

        self.pipeline = ProcessingPipeline()
        self._worker_thread: threading.Thread | None = None
        self._running = False
        self._session_start_time: Optional[float] = None
        self._selected_region: Optional[Tuple[int, int, int, int]] = None
        self._selected_window_hwnd: Optional[int] = None  # Window handle for dynamic tracking
        self._selected_window_title: Optional[str] = None  # Window title for fallback search

        self._build_ui()
        self._apply_styles()
        self._set_session_state("idle")

        # Use QueuedConnection to ensure thread-safe signal delivery
        self.frame_updated.connect(self._on_frame_updated, QtCore.Qt.QueuedConnection)
        self.stats_updated.connect(self._on_stats_updated, QtCore.Qt.QueuedConnection)

    def _build_ui(self):
        central = QtWidgets.QWidget(self)
        self.setCentralWidget(central)

        main_layout = QtWidgets.QHBoxLayout(central)

        # Left column -------------------------------------------------
        left_column = QtWidgets.QVBoxLayout()

        # Preview card
        preview_card = QtWidgets.QFrame()
        preview_card.setObjectName("Card")
        preview_layout = QtWidgets.QVBoxLayout(preview_card)

        preview_header = QtWidgets.QHBoxLayout()
        preview_title = QtWidgets.QLabel("Live Preview")
        preview_title.setObjectName("CardTitle")
        self.status_pill = QtWidgets.QLabel("Idle")
        self.status_pill.setObjectName("StatusPill")
        preview_header.addWidget(preview_title)
        preview_header.addStretch(1)
        preview_header.addWidget(self.status_pill)
        preview_layout.addLayout(preview_header)

        self.video_label = VideoLabel()
        self.video_label.setMinimumSize(720, 405)
        # Set placeholder text
        self.video_label.setText("Waiting for capture...")
        self.video_label.setAlignment(QtCore.Qt.AlignCenter)
        self.video_label.setStyleSheet("color: #8a94b8; font-size: 14px;")
        preview_layout.addWidget(self.video_label)

        left_column.addWidget(preview_card)

        # Controls card
        controls_card = QtWidgets.QFrame()
        controls_card.setObjectName("Card")
        controls_layout = QtWidgets.QGridLayout(controls_card)
        controls_layout.setVerticalSpacing(12)
        controls_layout.setHorizontalSpacing(10)
        controls_layout.setColumnStretch(0, 0)
        controls_layout.setColumnStretch(1, 1)
        controls_layout.setColumnStretch(2, 0)

        self.start_button = QtWidgets.QPushButton("Start Capture")
        self.stop_button = QtWidgets.QPushButton("Stop")
        self.stop_button.setEnabled(False)

        controls_layout.addWidget(self.start_button, 0, 0)
        controls_layout.addWidget(self.stop_button, 0, 1)
        
        # Screen/Window selection button
        self.select_screen_button = QtWidgets.QPushButton("📺 Select Screen/Window...")
        controls_layout.addWidget(self.select_screen_button, 0, 2)

        log_label = QtWidgets.QLabel("Log file")
        log_label.setObjectName("FieldLabel")
        controls_layout.addWidget(log_label, 1, 0)
        self.output_path_edit = QtWidgets.QLineEdit(os.path.abspath("engagement_log.csv"))
        self.browse_button = QtWidgets.QPushButton("Browse…")
        controls_layout.addWidget(self.output_path_edit, 1, 1)
        controls_layout.addWidget(self.browse_button, 1, 2)

        self.session_hint = QtWidgets.QLabel("Click 'Select Screen/Window...' to choose what to capture. Default: Full screen.")
        self.session_hint.setWordWrap(True)
        self.session_hint.setObjectName("HintLabel")
        controls_layout.addWidget(self.session_hint, 2, 0, 1, 3)

        left_column.addWidget(controls_card)
        left_column.addStretch(1)

        # Right column ------------------------------------------------
        right_tabs = QtWidgets.QTabWidget()
        right_tabs.setObjectName("ModernTabs")
        
        # Tab 1: Student list
        student_tab = QtWidgets.QWidget()
        student_layout = QtWidgets.QVBoxLayout(student_tab)
        table_title = QtWidgets.QLabel("Detected Students")
        table_title.setObjectName("CardTitle")
        student_layout.addWidget(table_title)
        self.table = QtWidgets.QTableWidget(0, 7)
        self.table.setHorizontalHeaderLabels(
            ["Rank", "ID", "Name", "Emotion", "Engagement", "Motion", "Brightness"]
        )
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.verticalHeader().setVisible(False)
        self.table.setAlternatingRowColors(True)
        student_layout.addWidget(self.table)
        right_tabs.addTab(student_tab, "Students")
        
        # Tab 2: Engagement graphs
        graph_tab = QtWidgets.QWidget()
        graph_layout = QtWidgets.QVBoxLayout(graph_tab)
        graph_title = QtWidgets.QLabel("Engagement Trends")
        graph_title.setObjectName("CardTitle")
        graph_layout.addWidget(graph_title)
        self.engagement_graph = EngagementGraph(graph_tab)
        graph_layout.addWidget(self.engagement_graph)
        right_tabs.addTab(graph_tab, "Graphs")

        main_layout.addLayout(left_column, stretch=3)
        main_layout.addWidget(right_tabs, stretch=2)

        # Connections
        self.start_button.clicked.connect(self.start_capture)
        self.stop_button.clicked.connect(self.stop_capture)
        self.browse_button.clicked.connect(self.choose_output_path)
        self.select_screen_button.clicked.connect(self.select_capture_source)
    def _apply_styles(self):
        self.setStyleSheet(
            """
            QMainWindow {
                background-color: #0f111a;
                color: #f3f5fa;
            }
            QLabel {
                font-size: 13px;
            }
            QLabel#CardTitle {
                font-size: 16px;
                font-weight: 600;
                color: #fefefe;
            }
            QLabel#FieldLabel {
                font-size: 11px;
                text-transform: uppercase;
                letter-spacing: 0.12em;
                color: #8a94b8;
            }
            QLabel#HintLabel {
                color: #7c86ad;
                font-size: 12px;
            }
            QFrame#Card {
                background-color: #191d2f;
                border: 1px solid #262b3f;
                border-radius: 14px;
            }
            QPushButton {
                background-color: #2e3353;
                border: 1px solid #3a4070;
                border-radius: 10px;
                padding: 10px 18px;
                color: #fefefe;
                font-weight: 600;
            }
            QPushButton:hover:!disabled {
                background-color: #3a4070;
            }
            QPushButton:disabled {
                background-color: #1c1f33;
                color: #5f668a;
                border-color: #1f2338;
            }
            QLineEdit {
                background-color: #101325;
                border: 1px solid #2a3054;
                border-radius: 8px;
                padding: 8px 10px;
                color: #fefefe;
                font-size: 13px;
            }
            QTabWidget::pane {
                border: 1px solid #262b3f;
                border-radius: 12px;
                background-color: #191d2f;
            }
            QTabBar::tab {
                background: #15192a;
                color: #c8cee9;
                padding: 10px 18px;
                border-top-left-radius: 8px;
                border-top-right-radius: 8px;
                margin-right: 4px;
            }
            QTabBar::tab:selected {
                background: #2b3050;
                color: #ffffff;
            }
            QTableWidget {
                background: #101325;
                alternate-background-color: #161a33;
                border: 1px solid #262b3f;
                border-radius: 10px;
                gridline-color: #1d223a;
                color: #f3f5fa;
            }
            QTableWidget::item {
                color: #f3f5fa;
                background-color: #101325;
            }
            QTableWidget::item:alternate {
                background-color: #161a33;
            }
            QTableWidget::item:selected {
                background-color: #2b3050;
                color: #ffffff;
            }
            QHeaderView::section {
                background: #181c31;
                color: #c8cee9;
                padding: 7px;
                border: none;
            }
            """
        )

    def _set_session_state(self, state: str):
        state_map = {
            "idle": ("Idle", "#3a3f5c"),
            "running": ("Capturing", "#23c4a2"),
            "stopped": ("Stopped", "#ff8a65"),
        }
        text, color = state_map.get(state, ("Idle", "#3a3f5c"))
        if hasattr(self, "status_pill"):
            self.status_pill.setText(text)
            self.status_pill.setStyleSheet(
                f"""
                QLabel {{
                    background-color: {color};
                    color: #0f111a;
                    border-radius: 12px;
                    padding: 4px 14px;
                    font-weight: 600;
                    font-size: 12px;
                }}
                """
            )

    def choose_output_path(self):
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Select log file", self.output_path_edit.text(), "CSV Files (*.csv);;All Files (*)"
        )
        if path:
            self.output_path_edit.setText(path)
    
    def _show_error(self, title: str, message: str):
        """Show error message with proper styling for dark theme."""
        msg = QtWidgets.QMessageBox(self)
        msg.setIcon(QtWidgets.QMessageBox.Warning)
        msg.setWindowTitle(title)
        msg.setText(message)
        msg.setStyleSheet("""
            QMessageBox {
                background-color: #0f111a;
                color: #f3f5fa;
            }
            QMessageBox QLabel {
                color: #f3f5fa !important;
                background-color: transparent;
                padding: 10px;
            }
            QMessageBox QPushButton {
                background-color: #2e3353;
                border: 1px solid #3a4070;
                border-radius: 8px;
                padding: 8px 16px;
                color: #fefefe !important;
                min-width: 80px;
            }
            QMessageBox QPushButton:hover {
                background-color: #3a4070;
            }
        """)
        msg.exec_()
    
    def select_capture_source(self):
        """Open window/monitor picker dialog."""
        dialog = WindowPickerDialog(self)
        if dialog.exec_() == QtWidgets.QDialog.Accepted:
            region = dialog.get_region()
            selection_type = dialog.get_selection_type()
            window_hwnd = dialog.get_window_hwnd()  # Get window handle if window selected
            window_title = dialog.get_window_title()  # Get window title if window selected
            
            if region:
                try:
                    left, top, width, height = region
                    self._selected_region = region
                    self._selected_window_hwnd = window_hwnd  # Store window handle for dynamic tracking
                    self._selected_window_title = window_title  # Store window title for fallback
                    
                    # Update hint label
                    if selection_type == "monitor":
                        self.session_hint.setText(f"✅ Selected: Monitor ({width}×{height})")
                    elif selection_type == "window":
                        dynamic_note = " (Dynamic tracking - works in fullscreen)" if window_hwnd else ""
                        self.session_hint.setText(f"✅ Selected: Window ({width}×{height}){dynamic_note}")
                    elif selection_type == "region":
                        self.session_hint.setText(f"✅ Selected: Custom Region ({width}×{height})")
                    else:
                        self.session_hint.setText(f"✅ Selected: {width}×{height} at ({left}, {top})")
                    
                    print(f"[MainWindow] Selected capture region: {region} (type: {selection_type}, hwnd: {window_hwnd})")
                except (ValueError, TypeError) as e:
                    print(f"[MainWindow] ERROR: Invalid region format: {region}, error: {e}")
                    self._show_error(
                        "Invalid Selection",
                        f"Invalid region format received from picker.\n\n"
                        f"Details: {str(e)}\n\n"
                        f"Please try selecting again."
                    )
                    self._selected_region = None
                    self._selected_window_hwnd = None
                    self.session_hint.setText("⚠️ Invalid selection. Please try again.")
            else:
                print("[MainWindow] WARNING: Dialog accepted but no region was returned")
                self._selected_region = None
                self._selected_window_hwnd = None
                self.session_hint.setText("⚠️ No selection. Will use full screen by default.")

    def start_capture(self):
        if self._running:
            return
        log_path = self.output_path_edit.text().strip()
        if not log_path:
            self._show_error("Invalid Path", "Please provide a valid log file path.")
            return

        # Set capture region if selected, otherwise use full screen (None)
        try:
            # For windows, use dynamic tracking; for monitors/regions, use static region
            if self._selected_window_hwnd is not None:
                # Use dynamic window tracking with title for fallback
                self.pipeline.capturer.set_window_hwnd(
                    self._selected_window_hwnd,
                    title=self._selected_window_title
                )
                print(f"[MainWindow] Using dynamic window tracking for HWND: {self._selected_window_hwnd}, title: '{self._selected_window_title}'")
            else:
                # Use static region or monitor
                self.pipeline.capturer.set_region(self._selected_region)
                self.pipeline.capturer.set_window_hwnd(None)  # Clear window tracking
            
            if self._selected_region:
                left, top, width, height = self._selected_region
                print(f"[MainWindow] Starting capture with region: {width}×{height} at ({left}, {top})")
                if width <= 0 or height <= 0:
                    self._show_error(
                        "Invalid Capture Region",
                        f"The selected region is invalid (width={width}, height={height}).\n\n"
                        "Please select a valid screen, window, or region using the 'Select Screen/Window...' button."
                    )
                    return
            else:
                print("[MainWindow] Starting capture with full screen (default)")
        except Exception as e:
            print(f"[MainWindow] ERROR setting capture region: {e}")
            self._show_error(
                "Capture Setup Error",
                f"Failed to set capture region:\n\n{str(e)}\n\n"
                "Please try selecting a different screen/window or restart the application."
            )
            return
        
        # Test screen capture before starting
        try:
            test_frame = self.pipeline.capturer.grab()
            if test_frame is None or test_frame.size == 0:
                self._show_error(
                    "Capture Error",
                    "Failed to capture screen. Please check:\n\n"
                    "• Your display is active\n"
                    "• No other application is blocking screen capture\n"
                    "• The selected window/region is visible\n"
                    "• Try running as administrator if issues persist"
                )
                return
            print(f"[MainWindow] Test capture successful: {test_frame.shape}")
        except Exception as e:
            import traceback
            traceback.print_exc()
            self._show_error(
                "Capture Initialization Error",
                f"Failed to initialize screen capture:\n\n{str(e)}\n\n"
                "Please check your display settings and try again."
            )
            return

        self.pipeline.start(log_path)
        self._running = True
        self._session_start_time = time.time()
        self.start_button.setEnabled(False)
        self.stop_button.setEnabled(True)
        self._set_session_state("running")

        # Reset graph
        self.engagement_graph.times.clear()
        self.engagement_graph.data_series.clear()

        self._worker_thread = threading.Thread(target=self._run_loop, daemon=True)
        self._worker_thread.start()
        print("[MainWindow] Capture started successfully")

    def stop_capture(self):
        if not self._running:
            return
        self._running = False
        self.pipeline.stop()
        self.start_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        self._set_session_state("stopped")

        # Generate Excel report automatically
        self._generate_excel_report()

    def _generate_excel_report(self):
        """Automatically generate Excel report when session ends."""
        log_path = self.output_path_edit.text().strip()
        if not log_path or not os.path.exists(log_path):
            return
        
        try:
            # Generate report path
            base_name = os.path.splitext(log_path)[0]
            report_path = f"{base_name}_report.xlsx"
            
            logger = EngagementLogger(log_path)
            df = logger.to_dataframe()
            
            if df.empty:
                QtWidgets.QMessageBox.information(
                    self, "No Data", "No engagement data was logged."
                )
                return
            
            report = ReportGenerator(df)
            report.export_excel(report_path)
            
            QtWidgets.QMessageBox.information(
                self,
                "Report Generated",
                f"Excel report saved to:\n{report_path}",
            )
        except Exception as e:
            QtWidgets.QMessageBox.warning(
                self, "Report Error", f"Failed to generate report:\n{str(e)}"
            )
        finally:
            if not self._running:
                self._set_session_state("idle")

    def closeEvent(self, event: QtGui.QCloseEvent):
        self.stop_capture()
        super().closeEvent(event)

    # Background loop
    def _run_loop(self):
        last_stats_time = 0.0
        last_graph_update = 0.0
        frame_count = 0
        no_frame_count = 0
        iteration = 0
        print("[MainWindow] Background loop started")
        while self._running:
            iteration += 1
            if iteration % 100 == 0:  # Log every 100 iterations (~1 second)
                print(f"[MainWindow] Loop iteration {iteration}, frames received: {frame_count}, no frames: {no_frame_count}")
            try:
                print(f"[MainWindow] Calling process_frame() for iteration {iteration}...")
                frame_bgr, students = self.pipeline.process_frame()
                print(f"[MainWindow] process_frame() returned, frame_bgr is None: {frame_bgr is None}")
                if iteration <= 10:  # Log first 10 iterations
                    print(f"[MainWindow] Iteration {iteration}: frame_bgr is None: {frame_bgr is None}, "
                          f"has size: {hasattr(frame_bgr, 'size') if frame_bgr is not None else False}, "
                          f"size: {frame_bgr.size if frame_bgr is not None and hasattr(frame_bgr, 'size') else 'N/A'}")
                if frame_bgr is not None and frame_bgr.size > 0:
                    print(f"[MainWindow] Frame is valid, proceeding to display...")
                    # Validate frame shape
                    if len(frame_bgr.shape) != 3 or frame_bgr.shape[2] != 3:
                        print(f"[MainWindow] Invalid frame shape: {frame_bgr.shape}")
                        time.sleep(0.033)
                        continue
                    
                    frame_count += 1
                    no_frame_count = 0
                    # Convert BGR to RGB for Qt
                    try:
                        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                        h, w, ch = frame_rgb.shape
                        if h <= 0 or w <= 0:
                            print(f"[MainWindow] Invalid frame dimensions: {w}x{h}")
                            time.sleep(0.033)
                            continue
                        bytes_per_line = ch * w
                        # Ensure data is contiguous and in correct format
                        if not frame_rgb.flags['C_CONTIGUOUS']:
                            frame_rgb = np.ascontiguousarray(frame_rgb)
                        
                        # Convert to uint8 if needed
                        if frame_rgb.dtype != np.uint8:
                            frame_rgb = frame_rgb.astype(np.uint8)
                        
                        # QImage constructor: QImage(data, width, height, bytesPerLine, format)
                        # Note: width and height parameters are swapped in some Qt versions
                        # Try both orders to ensure compatibility
                        try:
                            image = QtGui.QImage(
                                frame_rgb.data, w, h, bytes_per_line, QtGui.QImage.Format_RGB888
                            )
                            if image.isNull():
                                # Try with height, width swapped
                                print(f"[MainWindow] QImage creation failed with (w={w}, h={h}), trying swapped...")
                                image = QtGui.QImage(
                                    frame_rgb.data, h, w, bytes_per_line, QtGui.QImage.Format_RGB888
                                )
                        except Exception as qimg_e:
                            print(f"[MainWindow] QImage creation exception: {qimg_e}")
                            import traceback
                            traceback.print_exc()
                            time.sleep(0.033)
                            continue
                        
                        if image.isNull():
                            print(f"[MainWindow] Failed to create QImage from frame {w}x{h}, dtype={frame_rgb.dtype}")
                            time.sleep(0.033)
                            continue
                        
                        # Make a copy so Qt owns the memory
                        image = image.copy()
                        if image.isNull():
                            print("[MainWindow] Failed to copy QImage")
                            time.sleep(0.033)
                            continue
                        
                        self.frame_updated.emit(image)
                        if frame_count == 1:
                            print(f"[MainWindow] First frame displayed successfully: {w}x{h}, image size: {image.width()}x{image.height()}")
                    except Exception as conv_e:
                        print(f"[MainWindow] Error converting frame: {conv_e}")
                        import traceback
                        traceback.print_exc()
                        time.sleep(0.033)
                        continue
                else:
                    no_frame_count += 1
                    # Log warning if we haven't received frames for a while
                    if no_frame_count == 10:  # After ~0.1 second of no frames
                        print(f"[MainWindow] Warning: No frames received (count: {no_frame_count}, frame_bgr is None: {frame_bgr is None})")
                        if frame_bgr is not None:
                            print(f"[MainWindow] Frame exists but size is {frame_bgr.size if hasattr(frame_bgr, 'size') else 'unknown'}")
                    if no_frame_count == 100:  # After ~1 second of no frames
                        print(f"[MainWindow] Critical: No frames received for {no_frame_count} iterations")
                        # Show error message in preview
                        error_image = QtGui.QImage(720, 405, QtGui.QImage.Format_RGB32)
                        error_image.fill(QtGui.QColor(30, 30, 50))
                        painter = QtGui.QPainter(error_image)
                        painter.setPen(QtGui.QColor(255, 100, 100))
                        painter.setFont(QtGui.QFont("Arial", 14))
                        painter.drawText(error_image.rect(), QtCore.Qt.AlignCenter, 
                                       "No frames captured.\nCheck console for errors.")
                        painter.end()
                        self.frame_updated.emit(error_image)
                    # If no frame, wait a bit to avoid CPU spinning
                    time.sleep(0.033)  # ~30 FPS max
            except Exception as e:
                print(f"[MainWindow] Error in processing loop: {e}")
                import traceback
                traceback.print_exc()
                time.sleep(0.1)  # Wait before retrying
                continue

            # Small sleep to prevent excessive CPU usage
            time.sleep(0.01)
            
            now = time.time()
            if now - last_stats_time > 1.0:
                # Update stats table once per second
                snapshots: Dict[int, StudentSnapshot] = students
                ui_stats = {
                    sid: {
                        "name": s.name,
                        "emotion": s.last_emotion or "",
                        "engagement": s.last_engagement or 0.0,
                        "motion": s.motion_level or 0.0,
                        "brightness": s.brightness or 0.0,
                    }
                    for sid, s in snapshots.items()
                }
                self.stats_updated.emit(ui_stats)
                last_stats_time = now

            # Update graph less frequently (every 2 seconds)
            if now - last_graph_update > 2.0 and self._session_start_time:
                elapsed = now - self._session_start_time
                self.engagement_graph.update_graph(elapsed, students)
                last_graph_update = now

        # Finalize report when loop ends
        self.pipeline.finalize()

    @QtCore.pyqtSlot(object)
    def _on_frame_updated(self, image: QtGui.QImage):
        if image is None or image.isNull():
            print("[MainWindow] Received null or invalid QImage in _on_frame_updated")
            return
        try:
            self.video_label.setImage(image)
        except Exception as e:
            print(f"[MainWindow] Error in _on_frame_updated: {e}")
            import traceback
            traceback.print_exc()

    @QtCore.pyqtSlot(dict)
    def _on_stats_updated(self, stats: Dict[int, Dict]):
        sorted_stats = sorted(stats.items(), key=lambda kv: kv[1]["engagement"], reverse=True)
        self.table.setRowCount(len(sorted_stats))
        for row_idx, (sid, info) in enumerate(sorted_stats):
            # Create items and explicitly set text color
            rank_item = QtWidgets.QTableWidgetItem(str(row_idx + 1))
            rank_item.setForeground(QtGui.QColor("#f3f5fa"))
            self.table.setItem(row_idx, 0, rank_item)
            
            id_item = QtWidgets.QTableWidgetItem(str(sid))
            id_item.setForeground(QtGui.QColor("#f3f5fa"))
            self.table.setItem(row_idx, 1, id_item)
            
            name_item = QtWidgets.QTableWidgetItem(info["name"])
            name_item.setForeground(QtGui.QColor("#f3f5fa"))
            self.table.setItem(row_idx, 2, name_item)
            
            emotion_item = QtWidgets.QTableWidgetItem(info["emotion"])
            emotion_item.setForeground(QtGui.QColor("#f3f5fa"))
            self.table.setItem(row_idx, 3, emotion_item)
            
            engagement_item = QtWidgets.QTableWidgetItem(f"{info['engagement']:.2f}")
            engagement_item.setForeground(QtGui.QColor("#f3f5fa"))
            self.table.setItem(row_idx, 4, engagement_item)
            
            motion_item = QtWidgets.QTableWidgetItem(f"{info['motion']:.2f}")
            motion_item.setForeground(QtGui.QColor("#f3f5fa"))
            self.table.setItem(row_idx, 5, motion_item)
            
            brightness_item = QtWidgets.QTableWidgetItem(f"{info['brightness']:.2f}")
            brightness_item.setForeground(QtGui.QColor("#f3f5fa"))
            self.table.setItem(row_idx, 6, brightness_item)
