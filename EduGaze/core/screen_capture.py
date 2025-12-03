import threading
from typing import Optional, Tuple

import mss
import numpy as np


class ScreenCapturer:
    """
    Wrapper around mss to capture the primary screen, a fixed region, or a dynamic window.
    Supports dynamic window tracking for windows that change size/position (e.g., fullscreen).
    """

    def __init__(self, monitor_index: int = 1, region: Optional[Tuple[int, int, int, int]] = None):
        """
        :param monitor_index: mss monitor index (1 = primary)
        :param region: optional (left, top, width, height) in screen coordinates (absolute)
        """
        self.monitor_index = monitor_index
        self.region = region  # Static region in absolute screen coordinates
        self.window_hwnd = None  # Window handle for dynamic window tracking
        self.window_title = None  # Window title for fallback search if handle becomes invalid
        self._local = threading.local()
        self._revalidation_counter = 0  # Counter for periodic window re-validation
        self._revalidation_interval = 15  # Re-validate window every 15 frames (~0.5 seconds at 30fps)
        self._consecutive_failures = 0  # Track consecutive failures to trigger aggressive re-finding
    
    def set_region(self, region: Optional[Tuple[int, int, int, int]]):
        """Update capture region dynamically (static region mode)."""
        self.region = region
        self.window_hwnd = None  # Clear window tracking when setting static region
    
    def set_window_hwnd(self, hwnd: Optional[int], title: Optional[str] = None):
        """Set window handle for dynamic window tracking.
        
        :param hwnd: Window handle (HWND)
        :param title: Optional window title for fallback search if handle becomes invalid
        """
        self.window_hwnd = hwnd
        self.window_title = title
        if hwnd is None:
            self.region = None
            self.window_title = None
    
    def _get_window_rect(self) -> Optional[Tuple[int, int, int, int]]:
        """Dynamically get the current rect of the tracked window.
        Handles both windowed and fullscreen modes.
        Periodically re-validates and re-finds the window to handle fullscreen transitions.
        """
        if self.window_hwnd is None:
            return None
        
        try:
            import win32gui
            import win32con
            
            # Periodic re-validation: every N frames, check if window is still valid and try to re-find if needed
            self._revalidation_counter += 1
            should_revalidate = (self._revalidation_counter % self._revalidation_interval == 0)
            
            if should_revalidate:
                # Always check if window is still valid, even if IsWindow returns True
                # Sometimes the window handle becomes stale but IsWindow still returns True
                try:
                    current_title = win32gui.GetWindowText(self.window_hwnd)
                    current_rect = win32gui.GetWindowRect(self.window_hwnd)
                    current_width = current_rect[2] - current_rect[0]
                    current_height = current_rect[3] - current_rect[1]
                    
                    # If window dimensions are suspiciously small or title doesn't match, try to re-find
                    if (current_width < 100 or current_height < 100) or \
                       (self.window_title and self.window_title.lower() not in current_title.lower() and 
                        current_title.lower() not in self.window_title.lower()):
                        print(f"[ScreenCapturer] Periodic check: Window seems wrong (title: '{current_title}', size: {current_width}×{current_height}), re-finding...")
                        if self.window_title:
                            new_hwnd = self._find_window_by_title(self.window_title, search_children=True)
                            if new_hwnd and new_hwnd != self.window_hwnd:
                                print(f"[ScreenCapturer] Found window with new HWND: {new_hwnd} (was {self.window_hwnd})")
                                self.window_hwnd = new_hwnd
                except:
                    # If we can't get window info, it might be invalid
                    if self.window_title:
                        print(f"[ScreenCapturer] Periodic check: Cannot access window {self.window_hwnd}, re-finding...")
                        new_hwnd = self._find_window_by_title(self.window_title, search_children=True)
                        if new_hwnd and new_hwnd != self.window_hwnd:
                            print(f"[ScreenCapturer] Found window with new HWND: {new_hwnd} (was {self.window_hwnd})")
                            self.window_hwnd = new_hwnd
            
            # Check if window still exists
            if not win32gui.IsWindow(self.window_hwnd):
                print(f"[ScreenCapturer] Warning: Window handle {self.window_hwnd} is no longer valid")
                # Try to find window by title if we have it
                if self.window_title:
                    print(f"[ScreenCapturer] Attempting to re-find window by title: '{self.window_title}'")
                    new_hwnd = self._find_window_by_title(self.window_title, search_children=True)
                    if new_hwnd:
                        print(f"[ScreenCapturer] Found window again with new HWND: {new_hwnd}")
                        self.window_hwnd = new_hwnd
                    else:
                        print(f"[ScreenCapturer] Could not re-find window by title")
                        self.window_hwnd = None
                        return None
                else:
                    self.window_hwnd = None
                    return None
            
            # Check if window is visible
            if not win32gui.IsWindowVisible(self.window_hwnd):
                # Window might be minimized or hidden, but still try to get its rect
                # In fullscreen mode, sometimes the window appears invisible to IsWindowVisible
                pass
            
            # Get window placement to check if minimized
            try:
                placement = win32gui.GetWindowPlacement(self.window_hwnd)
                show_cmd = placement[1]  # SW_SHOWMINIMIZED = 2
                if show_cmd == win32con.SW_SHOWMINIMIZED:
                    # Window is minimized, restore it to get proper rect
                    # But don't actually restore - just get the restore rect
                    restore_rect = placement[4]  # (left, top, right, bottom)
                    if restore_rect and len(restore_rect) == 4:
                        left, top, right, bottom = restore_rect
                        width = right - left
                        height = bottom - top
                        if width > 0 and height > 0:
                            return (left, top, width, height)
            except:
                pass  # Fall through to GetWindowRect
            
            # Try GetWindowRect first (includes window borders)
            try:
                rect = win32gui.GetWindowRect(self.window_hwnd)
                left, top, right, bottom = rect
                width = right - left
                height = bottom - top
                
                # For fullscreen windows, sometimes GetWindowRect returns the whole screen
                # Check if this looks like a fullscreen window (covers most of primary monitor)
                if width > 1000 and height > 700:  # Reasonable fullscreen threshold
                    # Try GetClientRect + ClientToScreen for more accurate fullscreen capture
                    try:
                        client_rect = win32gui.GetClientRect(self.window_hwnd)
                        cl_left, cl_top, cl_right, cl_bottom = client_rect
                        client_width = cl_right - cl_left
                        client_height = cl_bottom - cl_top
                        
                        # Convert client coords to screen coords
                        point1 = win32gui.ClientToScreen(self.window_hwnd, (cl_left, cl_top))
                        point2 = win32gui.ClientToScreen(self.window_hwnd, (cl_right, cl_bottom))
                        
                        screen_left = point1[0]
                        screen_top = point1[1]
                        screen_width = point2[0] - point1[0]
                        screen_height = point2[1] - point1[1]
                        
                        if screen_width > 0 and screen_height > 0:
                            # Use client rect for more accurate capture
                            return (screen_left, screen_top, screen_width, screen_height)
                    except:
                        pass  # Fall back to window rect
                
                # Use window rect (standard case)
                if width > 0 and height > 0:
                    # If window is suspiciously small, try to re-find it
                    if width < 100 or height < 100:
                        print(f"[ScreenCapturer] Warning: Window rect is suspiciously small: {width}×{height}")
                        if self.window_title:
                            print(f"[ScreenCapturer] Attempting to re-find window due to small size...")
                            new_hwnd = self._find_window_by_title(self.window_title, search_children=True)
                            if new_hwnd and new_hwnd != self.window_hwnd:
                                print(f"[ScreenCapturer] Found larger window with new HWND: {new_hwnd}")
                                self.window_hwnd = new_hwnd
                                # Try again with new handle
                                try:
                                    new_rect = win32gui.GetWindowRect(new_hwnd)
                                    new_left, new_top, new_right, new_bottom = new_rect
                                    new_width = new_right - new_left
                                    new_height = new_bottom - new_top
                                    if new_width > 0 and new_height > 0:
                                        return (new_left, new_top, new_width, new_height)
                                except:
                                    pass
                    
                    return (left, top, width, height)
            except Exception as e:
                print(f"[ScreenCapturer] Error in GetWindowRect: {e}")
                # Try to re-find window if we get an error
                if self.window_title:
                    print(f"[ScreenCapturer] Attempting to re-find window after error...")
                    new_hwnd = self._find_window_by_title(self.window_title, search_children=True)
                    if new_hwnd and new_hwnd != self.window_hwnd:
                        print(f"[ScreenCapturer] Found window with new HWND: {new_hwnd}")
                        self.window_hwnd = new_hwnd
                        # Try one more time
                        try:
                            new_rect = win32gui.GetWindowRect(new_hwnd)
                            new_left, new_top, new_right, new_bottom = new_rect
                            new_width = new_right - new_left
                            new_height = new_bottom - new_top
                            if new_width > 0 and new_height > 0:
                                return (new_left, new_top, new_width, new_height)
                        except:
                            pass
                return None
            
            return None
        except ImportError:
            print("[ScreenCapturer] Warning: pywin32 not available for window tracking")
            self.window_hwnd = None
            return None
        except Exception as e:
            print(f"[ScreenCapturer] Error getting window rect: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def _find_window_by_title(self, title: str, search_children: bool = False) -> Optional[int]:
        """Find a window by its title. Returns the first matching HWND or None.
        
        :param title: Window title to search for (partial match supported)
        :param search_children: If True, also searches child windows (useful for fullscreen overlays)
        """
        try:
            import win32gui
            
            found_hwnd = None
            title_lower = title.lower()
            
            def enum_handler(hwnd, ctx):
                nonlocal found_hwnd
                if found_hwnd:
                    return  # Already found
                    
                try:
                    if win32gui.IsWindowVisible(hwnd):
                        window_title = win32gui.GetWindowText(hwnd)
                        if window_title:
                            window_title_lower = window_title.lower()
                            # Check for exact match or partial match
                            if (title_lower in window_title_lower or 
                                window_title_lower in title_lower or
                                any(word in window_title_lower for word in title_lower.split() if len(word) > 3)):
                                # Prefer windows that are not minimized
                                try:
                                    placement = win32gui.GetWindowPlacement(hwnd)
                                    show_cmd = placement[1]
                                    # Prefer non-minimized windows
                                    if show_cmd != 2:  # SW_SHOWMINIMIZED
                                        found_hwnd = hwnd
                                        return
                                except:
                                    found_hwnd = hwnd
                                    return
                except:
                    pass
            
            # Search top-level windows
            win32gui.EnumWindows(enum_handler, None)
            
            # If not found and search_children is True, search child windows
            if not found_hwnd and search_children:
                def enum_child_handler(hwnd, ctx):
                    nonlocal found_hwnd
                    if found_hwnd:
                        return
                    try:
                        window_title = win32gui.GetWindowText(hwnd)
                        if window_title:
                            window_title_lower = window_title.lower()
                            if (title_lower in window_title_lower or 
                                window_title_lower in title_lower):
                                # Check if it's a reasonable size (not a tiny control)
                                try:
                                    rect = win32gui.GetWindowRect(hwnd)
                                    width = rect[2] - rect[0]
                                    height = rect[3] - rect[1]
                                    if width > 200 and height > 200:  # Reasonable size
                                        found_hwnd = hwnd
                                except:
                                    pass
                    except:
                        pass
                
                # Enumerate all windows and check their children
                def enum_all_with_children(hwnd, ctx):
                    enum_child_handler(hwnd, ctx)
                    try:
                        win32gui.EnumChildWindows(hwnd, enum_child_handler, None)
                    except:
                        pass
                
                win32gui.EnumWindows(enum_all_with_children, None)
            
            if found_hwnd:
                try:
                    current_title = win32gui.GetWindowText(found_hwnd)
                    print(f"[ScreenCapturer] Found window: '{current_title}' (HWND: {found_hwnd})")
                except:
                    pass
            
            return found_hwnd
        except Exception as e:
            print(f"[ScreenCapturer] Error in _find_window_by_title: {e}")
            import traceback
            traceback.print_exc()
            return None

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
            
            # Priority 1: Dynamic window tracking (if window handle is set)
            if self.window_hwnd is not None:
                window_rect = self._get_window_rect()
                if window_rect:
                    left, top, width, height = window_rect
                    # Validate dimensions
                    if width > 0 and height > 0:
                        bbox = {
                            "left": left,
                            "top": top,
                            "width": width,
                            "height": height,
                        }
                        # Debug logging every 60 frames (roughly every 2 seconds at 30fps)
                        if hasattr(self, '_debug_frame_count'):
                            self._debug_frame_count += 1
                        else:
                            self._debug_frame_count = 0
                        
                        if self._debug_frame_count % 60 == 0:
                            try:
                                import win32gui
                                current_title = win32gui.GetWindowText(self.window_hwnd)
                                print(f"[ScreenCapturer] Dynamic window tracking: {width}×{height} at ({left}, {top}), title: '{current_title}'")
                            except:
                                print(f"[ScreenCapturer] Dynamic window tracking: {width}×{height} at ({left}, {top})")
                        # Reset failure counter on success
                        self._consecutive_failures = 0
                    else:
                        print(f"[ScreenCapturer] Warning: Invalid window dimensions: {width}×{height}")
                        window_rect = None
                
                if not window_rect:
                    # Window is invalid, increment failure counter
                    self._consecutive_failures += 1
                    
                    # If we've failed multiple times, try aggressive re-finding
                    if self._consecutive_failures >= 3 and self.window_title:
                        print(f"[ScreenCapturer] Multiple failures ({self._consecutive_failures}), trying aggressive re-find...")
                        # Try to find any window with similar title
                        new_hwnd = self._find_window_by_title(self.window_title, search_children=True)
                        if new_hwnd:
                            print(f"[ScreenCapturer] Aggressive re-find successful: {new_hwnd}")
                            self.window_hwnd = new_hwnd
                            self._consecutive_failures = 0
                            # Try one more time with new handle
                            window_rect = self._get_window_rect()
                    
                    if not window_rect:
                        # Still invalid, fall back to static region or monitor
                        print(f"[ScreenCapturer] Window rect invalid after {self._consecutive_failures} attempts, falling back to static region")
                        if self.region is not None:
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
            # Priority 2: Static region (if set)
            elif self.region is not None:
                # Region is in absolute screen coordinates
                left, top, width, height = self.region
                bbox = {
                    "left": left,
                    "top": top,
                    "width": width,
                    "height": height,
                }
            # Priority 3: Default: primary monitor
            else:
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


