"""
Window and region picker dialog for selecting screen capture area.
"""
import ctypes
from ctypes import wintypes
from typing import Optional, Tuple

from PyQt5 import QtCore, QtGui, QtWidgets


class RegionPickerDialog(QtWidgets.QDialog):
    """Dialog for selecting a window or drawing a custom region."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Select Capture Region")
        self.setMinimumSize(400, 200)
        self.selected_region: Optional[Tuple[int, int, int, int]] = None

        layout = QtWidgets.QVBoxLayout(self)

        info_label = QtWidgets.QLabel("Choose capture region:")
        layout.addWidget(info_label)

        # Window selection button
        window_btn = QtWidgets.QPushButton("Select Window...")
        window_btn.clicked.connect(self._pick_window)
        layout.addWidget(window_btn)

        # Region selection button
        region_btn = QtWidgets.QPushButton("Draw Region...")
        region_btn.clicked.connect(self._pick_region)
        layout.addWidget(region_btn)

        # Full screen option
        fullscreen_btn = QtWidgets.QPushButton("Full Screen")
        fullscreen_btn.clicked.connect(self._pick_fullscreen)
        layout.addWidget(fullscreen_btn)

        # Show selected region info
        self.region_label = QtWidgets.QLabel("No region selected")
        layout.addWidget(self.region_label)

        # Buttons
        button_box = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel
        )
        button_box.accepted.connect(self.accept)
        button_box.rejected.connect(self.reject)
        layout.addWidget(button_box)

    def _pick_window(self):
        """Let user pick a window to capture."""
        try:
            import win32gui
        except ImportError:
            QtWidgets.QMessageBox.information(
                self,
                "Window Selection",
                "Window selection requires pywin32. Please use 'Draw Region' instead or install pywin32.",
            )
            return
        
        self.hide()
        window_picker = WindowPicker()
        if window_picker.pick():
            self.selected_region = window_picker.get_window_rect()
            if self.selected_region:
                x, y, w, h = self.selected_region
                self.region_label.setText(f"Window: ({x}, {y}) {w}x{h}")
        self.show()

    def _pick_region(self):
        """Let user draw a region on screen."""
        self.hide()
        region_picker = RegionDrawDialog()
        if region_picker.exec_() == QtWidgets.QDialog.Accepted:
            self.selected_region = region_picker.get_region()
            if self.selected_region:
                x, y, w, h = self.selected_region
                self.region_label.setText(f"Region: ({x}, {y}) {w}x{h}")
        self.show()

    def _pick_fullscreen(self):
        """Select full primary screen."""
        import mss
        sct = mss.mss()
        mon = sct.monitors[1]  # Primary monitor
        self.selected_region = (mon["left"], mon["top"], mon["width"], mon["height"])
        self.region_label.setText(f"Full Screen: {mon['width']}x{mon['height']}")

    def get_region(self) -> Optional[Tuple[int, int, int, int]]:
        """Return selected region as (left, top, width, height)."""
        return self.selected_region


class WindowPicker:
    """Helper class to list and pick Windows windows."""

    def __init__(self):
        self.window_rect: Optional[Tuple[int, int, int, int]] = None

    def pick(self) -> bool:
        """Show window selection dialog."""
        try:
            import win32gui

            windows = self._get_windows()
            if not windows:
                return False

            dialog = QtWidgets.QDialog()
            dialog.setWindowTitle("Select Window")
            dialog.resize(600, 400)
            layout = QtWidgets.QVBoxLayout(dialog)

            list_widget = QtWidgets.QListWidget()
            for hwnd, title in windows:
                list_widget.addItem(title)
            layout.addWidget(list_widget)

            button_box = QtWidgets.QDialogButtonBox(
                QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel
            )
            button_box.accepted.connect(dialog.accept)
            button_box.rejected.connect(dialog.reject)
            layout.addWidget(button_box)

            if dialog.exec_() == QtWidgets.QDialog.Accepted:
                current_item = list_widget.currentItem()
                if current_item:
                    idx = list_widget.row(current_item)
                    hwnd, _ = windows[idx]
                    rect = win32gui.GetWindowRect(hwnd)
                    # Convert to (left, top, width, height)
                    self.window_rect = (
                        rect[0],
                        rect[1],
                        rect[2] - rect[0],
                        rect[3] - rect[1],
                    )
                    return True
            return False
        except ImportError:
            # Fallback if pywin32 not available
            QtWidgets.QMessageBox.warning(
                None, "Error", "Window selection requires pywin32. Using full screen."
            )
            return False

    @staticmethod
    def _get_windows():
        """Get list of visible Windows."""
        try:
            import win32gui

            windows = []

            def enum_handler(hwnd, ctx):
                if win32gui.IsWindowVisible(hwnd):
                    title = win32gui.GetWindowText(hwnd)
                    if title:
                        windows.append((hwnd, title))

            win32gui.EnumWindows(enum_handler, None)
            return windows
        except ImportError:
            return []

    def get_window_rect(self) -> Optional[Tuple[int, int, int, int]]:
        """Return selected window rectangle."""
        return self.window_rect


class RegionDrawDialog(QtWidgets.QDialog):
    """Full-screen overlay for drawing a capture region."""

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

        # Get screen geometry
        screen = QtWidgets.QApplication.primaryScreen().geometry()
        self.setGeometry(screen)

    def paintEvent(self, event):
        """Draw semi-transparent overlay with selected region highlighted."""
        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.Antialiasing)

        # Semi-transparent overlay
        painter.fillRect(self.rect(), QtGui.QColor(0, 0, 0, 100))

        if self.start_pos and self.end_pos:
            # Highlight selected region
            rect = QtCore.QRect(self.start_pos, self.end_pos).normalized()
            painter.fillRect(rect, QtGui.QColor(255, 255, 255, 0))
            painter.setPen(QtGui.QPen(QtGui.QColor(255, 0, 0), 3))
            painter.drawRect(rect)

    def mousePressEvent(self, event):
        """Start drawing region."""
        if event.button() == QtCore.Qt.LeftButton:
            self.start_pos = event.pos()
            self.end_pos = event.pos()
            self.drawing = True

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
            if self.start_pos and self.end_pos:
                # Wait a moment to show selection, then accept
                QtCore.QTimer.singleShot(300, self.accept)

    def keyPressEvent(self, event):
        """Cancel on Escape."""
        if event.key() == QtCore.Qt.Key_Escape:
            self.reject()

    def get_region(self) -> Optional[Tuple[int, int, int, int]]:
        """Return selected region in screen coordinates."""
        if not (self.start_pos and self.end_pos):
            return None

        screen = QtWidgets.QApplication.primaryScreen().geometry()
        start = self.mapToGlobal(self.start_pos)
        end = self.mapToGlobal(self.end_pos)

        x1, y1 = start.x(), start.y()
        x2, y2 = end.x(), end.y()

        left = min(x1, x2)
        top = min(y1, y2)
        width = abs(x2 - x1)
        height = abs(y2 - y1)

        return (left, top, width, height)

