import threading
from typing import Optional, Tuple

import mss
import numpy as np


class ScreenCapturer:
    """
    Wrapper around mss to capture the primary screen or a fixed region.
    For this PoC we capture the primary monitor; region selection can be added later.
    """

    def __init__(self, monitor_index: int = 1, region: Optional[Tuple[int, int, int, int]] = None):
        """
        :param monitor_index: mss monitor index (1 = primary)
        :param region: optional (left, top, width, height) in screen coordinates (absolute)
        """
        self.monitor_index = monitor_index
        self.region = region  # Now in absolute screen coordinates
        self._local = threading.local()
    
    def set_region(self, region: Optional[Tuple[int, int, int, int]]):
        """Update capture region dynamically."""
        self.region = region

    def _get_sct(self) -> mss.mss:
        """mss is not thread-safe on Windows; keep one instance per thread."""
        if not hasattr(self._local, "sct"):
            try:
                self._local.sct = mss.mss()
                # Test that it works
                monitors = self._local.sct.monitors
                if not monitors or len(monitors) < 2:
                    print(f"[ScreenCapturer] Warning: Only {len(monitors)} monitor(s) detected")
                else:
                    print(f"[ScreenCapturer] Initialized mss instance in thread, {len(monitors)} monitor(s) available")
            except Exception as e:
                print(f"[ScreenCapturer] Failed to create mss instance: {e}")
                import traceback
                traceback.print_exc()
                raise
        return self._local.sct

    def grab(self) -> np.ndarray:
        try:
            sct = self._get_sct()
            if self.region is not None:
                # Region is in absolute screen coordinates
                left, top, width, height = self.region
                bbox = {
                    "left": left,
                    "top": top,
                    "width": width,
                    "height": height,
                }
            else:
                # Default: primary monitor
                monitors = sct.monitors
                index = self.monitor_index
                if index >= len(monitors):
                    index = 1
                bbox = monitors[index]
            img = sct.grab(bbox)
            if img is None:
                print("[ScreenCapturer] Warning: mss.grab() returned None")
                return None
            frame = np.array(img)[:, :, :3]  # BGRA -> BGR
            if frame.size == 0:
                print("[ScreenCapturer] Warning: Frame is empty")
                return None
            return frame
        except Exception as e:
            print(f"[ScreenCapturer] Error during grab: {e}")
            import traceback
            traceback.print_exc()
            return None


