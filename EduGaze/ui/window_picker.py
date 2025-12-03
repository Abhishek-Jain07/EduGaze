"""
Enhanced window/monitor picker dialog similar to Zoom's screen sharing interface.
Allows users to select which screen, window, or region to capture with thumbnail previews.
"""
import ctypes
from ctypes import wintypes
from typing import Optional, Tuple, List
import mss
import numpy as np
import cv2

from PyQt5 import QtCore, QtGui, QtWidgets


class WindowPickerDialog(QtWidgets.QDialog):
    """Modern dialog for selecting capture source - similar to Zoom's screen sharing picker."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Select Screen or Window to Capture")
        self.setMinimumSize(700, 600)
        self.selected_region: Optional[Tuple[int, int, int, int]] = None
        self.selected_type: str = "none"  # "monitor", "window", "region", "fullscreen"
        self.selected_window_hwnd: Optional[int] = None  # Window handle for dynamic tracking
        self.selected_window_title: Optional[str] = None  # Window title for fallback search
        
        self._build_ui()
        self._apply_styles()
        self._populate_monitors()
        self._populate_windows()
        
    def _build_ui(self):
        """Build the modern UI similar to Zoom's picker."""
        layout = QtWidgets.QVBoxLayout(self)
        layout.setSpacing(20)
        layout.setContentsMargins(30, 30, 30, 30)
        
        # Title
        title = QtWidgets.QLabel("Select what to share")
        title.setObjectName("TitleLabel")
        layout.addWidget(title)
        
        # Info label
        info = QtWidgets.QLabel("Choose a screen, window, or draw a custom region to analyze")
        info.setObjectName("InfoLabel")
        layout.addWidget(info)
        layout.addSpacing(10)
        
        # Tabs for different selection modes
        tabs = QtWidgets.QTabWidget()
        tabs.setObjectName("PickerTabs")
        
        # Tab 1: Monitors
        monitor_tab = QtWidgets.QWidget()
        monitor_layout = QtWidgets.QVBoxLayout(monitor_tab)
        monitor_layout.setSpacing(15)
        
        monitor_label = QtWidgets.QLabel("Select a Monitor:")
        monitor_label.setObjectName("SectionLabel")
        monitor_layout.addWidget(monitor_label)
        
        self.monitor_list = QtWidgets.QListWidget()
        self.monitor_list.setObjectName("SelectionList")
        self.monitor_list.setIconSize(QtCore.QSize(200, 120))
        self.monitor_list.itemClicked.connect(self._on_monitor_clicked)
        self.monitor_list.itemDoubleClicked.connect(self._on_monitor_selected)
        monitor_layout.addWidget(self.monitor_list)
        
        tabs.addTab(monitor_tab, "📺 Monitors")
        
        # Tab 2: Windows
        window_tab = QtWidgets.QWidget()
        window_layout = QtWidgets.QVBoxLayout(window_tab)
        window_layout.setSpacing(15)
        
        window_label = QtWidgets.QLabel("Select a Window:")
        window_label.setObjectName("SectionLabel")
        window_layout.addWidget(window_label)
        
        # Search box for filtering windows
        self.window_search = QtWidgets.QLineEdit()
        self.window_search.setPlaceholderText("Search windows...")
        self.window_search.textChanged.connect(self._filter_windows)
        window_layout.addWidget(self.window_search)
        
        self.window_list = QtWidgets.QListWidget()
        self.window_list.setObjectName("SelectionList")
        self.window_list.setIconSize(QtCore.QSize(200, 120))
        self.window_list.itemClicked.connect(self._on_window_clicked)
        self.window_list.itemDoubleClicked.connect(self._on_window_selected)
        window_layout.addWidget(self.window_list)
        
        tabs.addTab(window_tab, "🪟 Windows")
        
        # Tab 3: Custom Region
        region_tab = QtWidgets.QWidget()
        region_layout = QtWidgets.QVBoxLayout(region_tab)
        region_layout.setSpacing(15)
        
        region_label = QtWidgets.QLabel("Draw a Custom Region:")
        region_label.setObjectName("SectionLabel")
        region_layout.addWidget(region_label)
        
        region_info = QtWidgets.QLabel("Click the button below to draw a region on your screen")
        region_info.setObjectName("InfoLabel")
        region_layout.addWidget(region_info)
        
        self.draw_region_btn = QtWidgets.QPushButton("📐 Draw Region on Screen")
        self.draw_region_btn.clicked.connect(self._pick_region)
        region_layout.addWidget(self.draw_region_btn)
        
        region_layout.addStretch()
        tabs.addTab(region_tab, "✏️ Custom Region")
        
        layout.addWidget(tabs)
        
        # Selected region display
        self.selected_label = QtWidgets.QLabel("No selection")
        self.selected_label.setObjectName("SelectedLabel")
        layout.addWidget(self.selected_label)
        
        # Buttons
        button_box = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel
        )
        button_box.accepted.connect(self._validate_and_accept)
        button_box.rejected.connect(self.reject)
        layout.addWidget(button_box)
        
    def _apply_styles(self):
        """Apply modern styling."""
        self.setStyleSheet("""
            QDialog {
                background-color: #0f111a;
                color: #f3f5fa;
            }
            QLabel#TitleLabel {
                font-size: 20px;
                font-weight: 600;
                color: #fefefe;
                padding-bottom: 5px;
            }
            QLabel#InfoLabel {
                font-size: 13px;
                color: #8a94b8;
                padding-bottom: 10px;
            }
            QLabel#SectionLabel {
                font-size: 14px;
                font-weight: 600;
                color: #f3f5fa;
            }
            QLabel#SelectedLabel {
                font-size: 12px;
                color: #23c4a2;
                padding: 10px;
                background-color: #191d2f;
                border-radius: 8px;
                border: 1px solid #262b3f;
            }
            QTabWidget::pane {
                border: 1px solid #262b3f;
                border-radius: 12px;
                background-color: #191d2f;
            }
            QTabBar::tab {
                background: #15192a;
                color: #c8cee9;
                padding: 10px 20px;
                border-top-left-radius: 8px;
                border-top-right-radius: 8px;
                margin-right: 2px;
            }
            QTabBar::tab:selected {
                background: #2b3050;
                color: #ffffff;
            }
            QListWidget#SelectionList {
                background: #101325;
                border: 1px solid #262b3f;
                border-radius: 10px;
                padding: 5px;
                color: #f3f5fa;
                font-size: 13px;
            }
            QListWidget#SelectionList::item {
                padding: 12px;
                border-radius: 6px;
                margin: 2px;
                min-height: 140px;
            }
            QListWidget#SelectionList::item:selected {
                background-color: #2b3050;
                color: #ffffff;
                border: 2px solid #23c4a2;
            }
            QListWidget#SelectionList::item:hover {
                background-color: #1a1f3a;
            }
            QLineEdit {
                background-color: #101325;
                border: 1px solid #2a3054;
                border-radius: 8px;
                padding: 10px;
                color: #fefefe;
                font-size: 13px;
            }
            QPushButton {
                background-color: #2e3353;
                border: 1px solid #3a4070;
                border-radius: 10px;
                padding: 12px 20px;
                color: #fefefe;
                font-weight: 600;
                font-size: 13px;
            }
            QPushButton:hover {
                background-color: #3a4070;
            }
            QPushButton:pressed {
                background-color: #252a48;
            }
            QDialogButtonBox QPushButton {
                min-width: 100px;
            }
            QMessageBox {
                background-color: #0f111a;
                color: #f3f5fa;
            }
            QMessageBox QLabel {
                color: #f3f5fa;
                background-color: #191d2f;
                padding: 10px;
            }
            QMessageBox QPushButton {
                min-width: 80px;
            }
        """)
    
    def _capture_window_thumbnail(self, hwnd) -> Optional[QtGui.QPixmap]:
        """Capture a thumbnail of a window."""
        try:
            import win32gui
            import win32ui
            import win32con
            
            # Get window rect
            rect = win32gui.GetWindowRect(hwnd)
            width = rect[2] - rect[0]
            height = rect[3] - rect[1]
            
            if width <= 0 or height <= 0:
                return None
            
            # Limit thumbnail size for performance
            max_thumb_size = 400
            scale = min(1.0, max_thumb_size / max(width, height))
            thumb_width = int(width * scale)
            thumb_height = int(height * scale)
            
            # Use mss to capture the window region
            try:
                sct = mss.mss()
                bbox = {
                    "left": rect[0],
                    "top": rect[1],
                    "width": width,
                    "height": height,
                }
                img = sct.grab(bbox)
                frame = np.array(img)[:, :, :3]  # BGRA -> BGR
                
                # Resize for thumbnail
                if scale < 1.0:
                    frame = cv2.resize(frame, (thumb_width, thumb_height), interpolation=cv2.INTER_AREA)
                
                # Convert to QPixmap
                height, width = frame.shape[:2]
                bytes_per_line = 3 * width
                q_img = QtGui.QImage(frame.data, width, height, bytes_per_line, QtGui.QImage.Format_RGB888).rgbSwapped()
                pixmap = QtGui.QPixmap.fromImage(q_img)
                return pixmap
            except Exception as e:
                print(f"[WindowPicker] Error capturing window thumbnail with mss: {e}")
                return None
                
        except ImportError:
            return None
        except Exception as e:
            print(f"[WindowPicker] Error capturing window thumbnail: {e}")
            return None
    
    def _capture_monitor_thumbnail(self, monitor_info: dict) -> Optional[QtGui.QPixmap]:
        """Capture a thumbnail of a monitor."""
        try:
            sct = mss.mss()
            img = sct.grab(monitor_info)
            frame = np.array(img)[:, :, :3]  # BGRA -> BGR
            
            # Resize for thumbnail (max 400px on longest side)
            max_thumb_size = 400
            height, width = frame.shape[:2]
            scale = min(1.0, max_thumb_size / max(width, height))
            
            if scale < 1.0:
                thumb_width = int(width * scale)
                thumb_height = int(height * scale)
                frame = cv2.resize(frame, (thumb_width, thumb_height), interpolation=cv2.INTER_AREA)
            
            # Convert to QPixmap
            height, width = frame.shape[:2]
            bytes_per_line = 3 * width
            q_img = QtGui.QImage(frame.data, width, height, bytes_per_line, QtGui.QImage.Format_RGB888).rgbSwapped()
            pixmap = QtGui.QPixmap.fromImage(q_img)
            return pixmap
        except Exception as e:
            print(f"[WindowPicker] Error capturing monitor thumbnail: {e}")
            return None
    
    def _populate_monitors(self):
        """Populate monitor list with available screens and thumbnails."""
        try:
            sct = mss.mss()
            monitors = sct.monitors
            
            # Skip monitor 0 (all monitors combined)
            for i in range(1, len(monitors)):
                mon = monitors[i]
                width = mon["width"]
                height = mon["height"]
                left = mon["left"]
                top = mon["top"]
                
                # Determine if this is primary monitor
                is_primary = (i == 1)
                label = f"Monitor {i}"
                if is_primary:
                    label += " (Primary)"
                label += f"\n{width}×{height} at ({left}, {top})"
                
                # Capture thumbnail
                thumbnail = self._capture_monitor_thumbnail(mon)
                
                item = QtWidgets.QListWidgetItem(label)
                item.setData(QtCore.Qt.UserRole, (i, mon))
                if thumbnail:
                    # Scale thumbnail to fit icon size
                    scaled_thumb = thumbnail.scaled(200, 120, QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation)
                    item.setIcon(QtGui.QIcon(scaled_thumb))
                else:
                    # Fallback icon
                    item.setIcon(QtGui.QIcon())
                self.monitor_list.addItem(item)
                
        except Exception as e:
            print(f"[WindowPicker] Error populating monitors: {e}")
            item = QtWidgets.QListWidgetItem("Error loading monitors. Using primary screen.")
            self.monitor_list.addItem(item)
    
    def _populate_windows(self):
        """Populate window list with available Windows and thumbnails."""
        self._all_windows = []
        
        try:
            import win32gui
            
            def enum_handler(hwnd, ctx):
                if win32gui.IsWindowVisible(hwnd):
                    title = win32gui.GetWindowText(hwnd)
                    if title and title.strip():
                        # Filter out empty titles and some system windows
                        # Skip very small windows
                        try:
                            rect = win32gui.GetWindowRect(hwnd)
                            width = rect[2] - rect[0]
                            height = rect[3] - rect[1]
                            if width > 50 and height > 50:  # Filter tiny windows
                                self._all_windows.append((hwnd, title))
                        except:
                            pass
            
            win32gui.EnumWindows(enum_handler, None)
            
            # Sort by title
            self._all_windows.sort(key=lambda x: x[1].lower())
            
            # Add windows with thumbnails
            for hwnd, title in self._all_windows:
                # Capture thumbnail
                thumbnail = self._capture_window_thumbnail(hwnd)
                
                item = QtWidgets.QListWidgetItem(title)
                item.setData(QtCore.Qt.UserRole, hwnd)
                if thumbnail:
                    # Scale thumbnail to fit icon size
                    scaled_thumb = thumbnail.scaled(200, 120, QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation)
                    item.setIcon(QtGui.QIcon(scaled_thumb))
                else:
                    # Fallback icon
                    item.setIcon(QtGui.QIcon())
                self.window_list.addItem(item)
                
        except ImportError:
            item = QtWidgets.QListWidgetItem("Window selection requires pywin32. Please install it.")
            item.setFlags(QtCore.Qt.NoItemFlags)
            self.window_list.addItem(item)
        except Exception as e:
            print(f"[WindowPicker] Error populating windows: {e}")
            item = QtWidgets.QListWidgetItem(f"Error loading windows: {str(e)}")
            self.window_list.addItem(item)
    
    def _filter_windows(self, text: str):
        """Filter window list based on search text."""
        if not hasattr(self, '_all_windows'):
            return
            
        search_text = text.lower()
        self.window_list.clear()
        
        for hwnd, title in self._all_windows:
            if search_text in title.lower():
                # Capture thumbnail (might be slow, but ensures fresh thumbnails)
                thumbnail = self._capture_window_thumbnail(hwnd)
                
                item = QtWidgets.QListWidgetItem(title)
                item.setData(QtCore.Qt.UserRole, hwnd)
                if thumbnail:
                    scaled_thumb = thumbnail.scaled(200, 120, QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation)
                    item.setIcon(QtGui.QIcon(scaled_thumb))
                self.window_list.addItem(item)
    
    def _on_monitor_clicked(self, item: QtWidgets.QListWidgetItem):
        """Handle monitor click (single click - preview selection)."""
        data = item.data(QtCore.Qt.UserRole)
        if not data:
            return
        
        # Clear window tracking when selecting monitor
        self.selected_window_hwnd = None
            
        monitor_idx, mon = data
        self.selected_region = (mon["left"], mon["top"], mon["width"], mon["height"])
        self.selected_type = "monitor"
        self.selected_label.setText(f"Selected: {item.text().split(chr(10))[0]} ({mon['width']}×{mon['height']})")
        self.monitor_list.setCurrentItem(item)
    
    def _on_monitor_selected(self, item: QtWidgets.QListWidgetItem):
        """Handle monitor double-click (accept immediately)."""
        self._on_monitor_clicked(item)
        # Optionally auto-accept on double-click
        # self.accept()
    
    def _on_window_clicked(self, item: QtWidgets.QListWidgetItem):
        """Handle window click (single click - preview selection)."""
        try:
            import win32gui
            
            hwnd = item.data(QtCore.Qt.UserRole)
            if not hwnd:
                return
            
            # Store the window handle and title for dynamic tracking
            self.selected_window_hwnd = hwnd
            self.selected_window_title = item.text()  # Store title for fallback search
                
            rect = win32gui.GetWindowRect(hwnd)
            # Convert to (left, top, width, height)
            self.selected_region = (
                rect[0],
                rect[1],
                rect[2] - rect[0],
                rect[3] - rect[1],
            )
            self.selected_type = "window"
            self.selected_label.setText(f"Selected: {item.text()} ({rect[2] - rect[0]}×{rect[3] - rect[1]}) - Dynamic tracking enabled")
            self.window_list.setCurrentItem(item)
            
        except Exception as e:
            print(f"[WindowPicker] Error selecting window: {e}")
            self._show_error("Error", f"Failed to select window:\n{str(e)}")
    
    def _on_window_selected(self, item: QtWidgets.QListWidgetItem):
        """Handle window double-click (accept immediately)."""
        self._on_window_clicked(item)
        # Optionally auto-accept on double-click
        # self.accept()
    
    def _pick_region(self):
        """Open region drawing dialog."""
        # Clear window tracking when selecting custom region
        self.selected_window_hwnd = None
        
        self.hide()
        region_picker = RegionDrawDialog(self)
        if region_picker.exec_() == QtWidgets.QDialog.Accepted:
            self.selected_region = region_picker.get_region()
            if self.selected_region:
                x, y, w, h = self.selected_region
                self.selected_type = "region"
                self.selected_label.setText(f"Selected: Custom Region ({w}×{h})")
        self.show()
    
    def _show_error(self, title: str, message: str):
        """Show error message with proper styling."""
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
                color: #f3f5fa;
                background-color: transparent;
                padding: 10px;
            }
            QMessageBox QPushButton {
                background-color: #2e3353;
                border: 1px solid #3a4070;
                border-radius: 8px;
                padding: 8px 16px;
                color: #fefefe;
                min-width: 80px;
            }
            QMessageBox QPushButton:hover {
                background-color: #3a4070;
            }
        """)
        msg.exec_()
    
    def _validate_and_accept(self):
        """Validate selection before accepting."""
        if not self.selected_region:
            self._show_error(
                "No Selection",
                "Please select a monitor, window, or draw a custom region before continuing.\n\n"
                "Click on an item in the list to select it, then click OK."
            )
            return
        
        # Validate region is valid
        try:
            left, top, width, height = self.selected_region
            if width <= 0 or height <= 0:
                self._show_error(
                    "Invalid Region",
                    "Selected region is invalid. Please select a valid area."
                )
                return
        except (ValueError, TypeError) as e:
            self._show_error(
                "Invalid Selection",
                f"Selected region format is invalid: {str(e)}\n\nPlease try selecting again."
            )
            return
        
        self.accept()
    
    def get_region(self) -> Optional[Tuple[int, int, int, int]]:
        """Return selected region as (left, top, width, height)."""
        return self.selected_region
    
    def get_selection_type(self) -> str:
        """Return type of selection: 'monitor', 'window', 'region', or 'fullscreen'."""
        return self.selected_type
    
    def get_window_hwnd(self) -> Optional[int]:
        """Return the window handle (HWND) if a window is selected, None otherwise."""
        return self.selected_window_hwnd
    
    def get_window_title(self) -> Optional[str]:
        """Return the window title if a window is selected, None otherwise."""
        return self.selected_window_title


class RegionDrawDialog(QtWidgets.QDialog):
    """Full-screen overlay for drawing a capture region - similar to Zoom's region picker."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.start_pos: Optional[QtCore.QPoint] = None
        self.end_pos: Optional[QtCore.QPoint] = None
        self.drawing = False
        
        self.setWindowFlags(
            QtCore.Qt.FramelessWindowHint
            | QtCore.Qt.WindowStaysOnTopHint
            | QtCore.Qt.Tool
        )
        self.setAttribute(QtCore.Qt.WA_TranslucentBackground)
        self.setCursor(QtCore.Qt.CrossCursor)
        
        # Cover all screens
        self._setup_multiscreen()
        
        # Instructions label
        self.instructions = QtWidgets.QLabel("Click and drag to select a region. Press ESC to cancel.", self)
        self.instructions.setStyleSheet("""
            QLabel {
                background-color: rgba(15, 17, 26, 200);
                color: #fefefe;
                padding: 10px 20px;
                border-radius: 8px;
                font-size: 14px;
                font-weight: 600;
            }
        """)
        self.instructions.setAlignment(QtCore.Qt.AlignCenter)
        self.instructions.move(50, 50)
        self.instructions.adjustSize()
        
    def _setup_multiscreen(self):
        """Setup dialog to cover all screens."""
        screens = QtWidgets.QApplication.screens()
        if not screens:
            return
            
        # Get bounding rect of all screens
        min_x = min(s.geometry().left() for s in screens)
        min_y = min(s.geometry().top() for s in screens)
        max_x = max(s.geometry().right() for s in screens)
        max_y = max(s.geometry().bottom() for s in screens)
        
        width = max_x - min_x
        height = max_y - min_y
        
        self.setGeometry(min_x, min_y, width, height)

    def paintEvent(self, event):
        """Draw semi-transparent overlay with selected region highlighted."""
        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.Antialiasing)

        # Semi-transparent dark overlay
        painter.fillRect(self.rect(), QtGui.QColor(0, 0, 0, 150))

        if self.start_pos and self.end_pos:
            # Highlight selected region (clear area)
            rect = QtCore.QRect(self.start_pos, self.end_pos).normalized()
            
            # Clear the selected region (make it brighter)
            painter.setCompositionMode(QtGui.QPainter.CompositionMode_Clear)
            painter.fillRect(rect, QtCore.Qt.transparent)
            painter.setCompositionMode(QtGui.QPainter.CompositionMode_SourceOver)
            
            # Draw border around selected region
            painter.setPen(QtGui.QPen(QtGui.QColor(255, 100, 100), 4))
            painter.drawRect(rect)
            
            # Draw corner handles
            handle_size = 10
            corners = [
                rect.topLeft(),
                rect.topRight(),
                rect.bottomLeft(),
                rect.bottomRight(),
            ]
            for corner in corners:
                painter.fillRect(
                    corner.x() - handle_size // 2,
                    corner.y() - handle_size // 2,
                    handle_size,
                    handle_size,
                    QtGui.QColor(255, 100, 100)
                )
            
            # Show dimensions
            width = rect.width()
            height = rect.height()
            dim_text = f"{width} × {height}"
            font = QtGui.QFont()
            font.setBold(True)
            font.setPointSize(12)
            painter.setFont(font)
            painter.setPen(QtGui.QPen(QtGui.QColor(255, 255, 255), 2))
            
            # Position text above or below selection
            text_rect = QtCore.QRect(rect)
            text_rect.setHeight(30)
            text_rect.moveTop(rect.top() - 35 if rect.top() > 50 else rect.bottom() + 5)
            painter.drawText(text_rect, QtCore.Qt.AlignCenter, dim_text)

    def mousePressEvent(self, event):
        """Start drawing region."""
        if event.button() == QtCore.Qt.LeftButton:
            self.start_pos = event.pos()
            self.end_pos = event.pos()
            self.drawing = True
            self.update()

    def mouseMoveEvent(self, event):
        """Update region while drawing."""
        if self.drawing:
            self.end_pos = event.pos()
            self.update()

    def mouseReleaseEvent(self, event):
        """Finish drawing region."""
        if event.button() == QtCore.Qt.LeftButton and self.drawing:
            self.end_pos = event.pos()
            self.drawing = False
            
            # Ensure minimum size
            if self.start_pos and self.end_pos:
                rect = QtCore.QRect(self.start_pos, self.end_pos).normalized()
                if rect.width() < 100 or rect.height() < 100:
                    QtWidgets.QMessageBox.information(
                        self,
                        "Region Too Small",
                        "Please select a larger region (minimum 100×100 pixels)."
                    )
                    self.start_pos = None
                    self.end_pos = None
                    self.update()
                    return
                
                # Wait a moment to show selection, then accept
                QtCore.QTimer.singleShot(500, self.accept)

    def keyPressEvent(self, event):
        """Cancel on Escape."""
        if event.key() == QtCore.Qt.Key_Escape:
            self.reject()

    def get_region(self) -> Optional[Tuple[int, int, int, int]]:
        """Return selected region in screen coordinates."""
        if not (self.start_pos and self.end_pos):
            return None

        # Get global screen coordinates
        start_global = self.mapToGlobal(self.start_pos)
        end_global = self.mapToGlobal(self.end_pos)

        x1, y1 = start_global.x(), start_global.y()
        x2, y2 = end_global.x(), end_global.y()

        left = min(x1, x2)
        top = min(y1, y2)
        width = abs(x2 - x1)
        height = abs(y2 - y1)

        return (left, top, width, height)
