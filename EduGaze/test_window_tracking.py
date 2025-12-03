"""
Diagnostic script to test window tracking when Zoom goes fullscreen.
Run this while Zoom is open to see what windows are available.
"""
import time
try:
    import win32gui
    
    def enum_windows():
        """List all visible windows."""
        windows = []
        
        def enum_handler(hwnd, ctx):
            if win32gui.IsWindowVisible(hwnd):
                title = win32gui.GetWindowText(hwnd)
                if title:
                    try:
                        rect = win32gui.GetWindowRect(hwnd)
                        width = rect[2] - rect[0]
                        height = rect[3] - rect[1]
                        windows.append((hwnd, title, width, height))
                    except:
                        pass
        
        win32gui.EnumWindows(enum_handler, None)
        return windows
    
    print("=" * 80)
    print("Window Tracking Diagnostic Tool")
    print("=" * 80)
    print("\nInstructions:")
    print("1. Open Zoom in windowed mode")
    print("2. Run this script")
    print("3. Note the HWND of the Zoom window")
    print("4. Switch Zoom to fullscreen")
    print("5. Run this script again")
    print("6. Check if the HWND changed\n")
    
    input("Press Enter when Zoom is in windowed mode...")
    
    print("\n--- WINDOWED MODE ---")
    windows_windowed = enum_windows()
    zoom_windows = [(hwnd, title, w, h) for hwnd, title, w, h in windows_windowed if "zoom" in title.lower()]
    
    print(f"\nFound {len(zoom_windows)} Zoom-related windows:")
    for hwnd, title, width, height in zoom_windows:
        print(f"  HWND: {hwnd:10d} | {width:5d}×{height:5d} | {title}")
    
    if zoom_windows:
        selected_hwnd = zoom_windows[0][0]
        print(f"\nSelected HWND: {selected_hwnd}")
        print(f"Window title: {zoom_windows[0][1]}")
        
        input("\nNow switch Zoom to FULLSCREEN and press Enter...")
        
        print("\n--- FULLSCREEN MODE ---")
        windows_fullscreen = enum_windows()
        zoom_windows_fs = [(hwnd, title, w, h) for hwnd, title, w, h in windows_fullscreen if "zoom" in title.lower()]
        
        print(f"\nFound {len(zoom_windows_fs)} Zoom-related windows:")
        for hwnd, title, width, height in zoom_windows_fs:
            print(f"  HWND: {hwnd:10d} | {width:5d}×{height:5d} | {title}")
        
        print(f"\nOriginal HWND: {selected_hwnd}")
        if selected_hwnd in [hwnd for hwnd, _, _, _ in zoom_windows_fs]:
            print("✅ Original HWND still exists in fullscreen!")
        else:
            print("❌ Original HWND NOT found in fullscreen!")
            if zoom_windows_fs:
                print(f"   New HWND might be: {zoom_windows_fs[0][0]}")
        
        # Check if window is still valid
        if win32gui.IsWindow(selected_hwnd):
            try:
                current_title = win32gui.GetWindowText(selected_hwnd)
                current_rect = win32gui.GetWindowRect(selected_hwnd)
                current_width = current_rect[2] - current_rect[0]
                current_height = current_rect[3] - current_rect[1]
                print(f"\nOriginal window status:")
                print(f"  IsWindow: True")
                print(f"  Title: '{current_title}'")
                print(f"  Size: {current_width}×{current_height}")
            except Exception as e:
                print(f"\nOriginal window status: Error accessing - {e}")
        else:
            print(f"\nOriginal window status: IsWindow returns False")
    
    print("\n" + "=" * 80)
    
except ImportError:
    print("ERROR: pywin32 is not installed. Please install it with: pip install pywin32")
except Exception as e:
    print(f"ERROR: {e}")
    import traceback
    traceback.print_exc()



