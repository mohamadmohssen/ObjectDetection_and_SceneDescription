import cv2
import time
import threading

from detector           import YOLODetector
from caption_model      import CaptionModel
from utils.preprocessor import convert_to_pil
from caption_enhancer   import CaptionEnhancer, MODES
from utils.camera       import open_camera, camera_zoom
from utils.tracker      import ObjectTracker
from utils.visualizer   import (
    build_frame,
    get_mode_button_rect,
    get_vision_button_rect,
    get_stats_button_rect,
    get_stats_reset_btn_rect,
    get_dropdown_item_at,
    CAPTION_PANEL_HEIGHT,
    STATS_PANEL_W,
)

 
# Camera preference
 
CAMERA_INDEX  = 0
PREFERRED_W   = 1920
PREFERRED_H   = 1080
PREFERRED_FPS = 30
 
#camera zooms
ZOOM_WIDTH = 1920
ZOOM_HEIGHT = 1080
PERFORM_ZOOM = False

# Initializing models
 
detector  = YOLODetector()
captioner = CaptionModel()
enhancer  = CaptionEnhancer()

 
tracker = ObjectTracker()

 
# Camera 
 
try:
    cap, CAM_W, CAM_H, CAM_FPS = open_camera(
        index         = CAMERA_INDEX,
        preferred_w   = PREFERRED_W,
        preferred_h   = PREFERRED_H,
        preferred_fps = PREFERRED_FPS,
    )
except RuntimeError as e:
    print(e)
    exit(1)

 
# Screen resolution
 
def get_screen_resolution():
    try:
        import tkinter as tk
        root = tk.Tk()
        root.withdraw()
        w, h = root.winfo_screenwidth(), root.winfo_screenheight()
        root.destroy()
        if w > 100 and h > 100:
            return w, h
    except Exception:
        pass
    return 1280, 720


SCREEN_W, SCREEN_H = get_screen_resolution()

WIN_W   = SCREEN_W
WIN_H   = SCREEN_H
VIDEO_H = WIN_H - CAPTION_PANEL_HEIGHT

print(f"[INFO] Screen    : {SCREEN_W}×{SCREEN_H}")
print(f"[INFO] Camera    : {CAM_W}×{CAM_H} @ {CAM_FPS:.0f} fps")
print(f"[INFO] Video area: {WIN_W}×{VIDEO_H}  (letterboxed)")

 
# Window

WIN_NAME = "YOLO Scene Captioning"
cv2.namedWindow(WIN_NAME, cv2.WINDOW_NORMAL)
cv2.resizeWindow(WIN_NAME, WIN_W, WIN_H)
cv2.moveWindow(WIN_NAME, 0, 0)

 
# UI state
last_caption     = ""
last_time        = 0.0
previous_caption = ""
CAPTION_INTERVAL = 4       # caption refreshes by seconds

lock           = threading.Lock()
active_threads = 0
MAX_THREADS    = 1

dropdown_open    = False
hover_index      = -1

stats_open       = False      
stats_reset_hover = False     
stats_btn_hover   = False 

 
# Mouse callback
 
def on_mouse(event, mx, my, flags, param):
    global dropdown_open, hover_index
    global stats_open, stats_reset_hover, stats_btn_hover

    # Hover tracking
    if event == cv2.EVENT_MOUSEMOVE:
        if dropdown_open:
            hover_index = get_dropdown_item_at(WIN_W, mx, my)

        sx1, sy1, sx2, sy2 = get_stats_button_rect(WIN_W)
        stats_btn_hover = (sx1 <= mx <= sx2 and sy1 <= my <= sy2)

        if stats_open:
            rx1, ry1, rx2, ry2 = get_stats_reset_btn_rect(WIN_W, WIN_H)
            stats_reset_hover = (rx1 <= mx <= rx2 and ry1 <= my <= ry2)
        else:
            stats_reset_hover = False

    elif event == cv2.EVENT_LBUTTONDOWN:

        vx1, vy1, vx2, vy2 = get_vision_button_rect()
        if vx1 <= mx <= vx2 and vy1 <= my <= vy2:
            detector.toggle()
            dropdown_open = False
            return

        sx1, sy1, sx2, sy2 = get_stats_button_rect(WIN_W)
        if sx1 <= mx <= sx2 and sy1 <= my <= sy2:
            stats_open = not stats_open
            new_w = WIN_W + (STATS_PANEL_W if stats_open else 0)
            cv2.resizeWindow(WIN_NAME, new_w, WIN_H)
            dropdown_open = False
            return

        if stats_open:
            rx1, ry1, rx2, ry2 = get_stats_reset_btn_rect(WIN_W, WIN_H)
            if rx1 <= mx <= rx2 and ry1 <= my <= ry2:
                tracker.reset()
                return

        bx1, by1, bx2, by2 = get_mode_button_rect(WIN_W)
        if dropdown_open:
            idx = get_dropdown_item_at(WIN_W, mx, my)
            if 0 <= idx < len(MODES):
                enhancer.set_mode(list(MODES.keys())[idx])
            dropdown_open = False
            hover_index   = -1
        else:
            if bx1 <= mx <= bx2 and by1 <= my <= by2:
                dropdown_open = True


cv2.setMouseCallback(WIN_NAME, on_mouse)

 
# Caption worker
 
def caption_worker(frame, detected_labels):
    global last_caption, previous_caption, active_threads
    try:
        pil_frame   = convert_to_pil(frame).resize((224, 224))
        raw_caption = captioner.caption(pil_frame)
        enhanced    = enhancer.enhance(raw_caption, detected_labels)
        if enhanced:
            previous_caption = last_caption
            last_caption = enhanced
            print("[INFO] Caption:", last_caption)
    finally:
        with lock:
            active_threads -= 1

 
# Main loop
 
print("[INFO] Starting… press Q to quit, Esc to close dropdown.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("[ERROR] Failed to read frame")
        break

    frame = cv2.flip(frame, 1)
    if PERFORM_ZOOM:
        frame = camera_zoom(frame, ZOOM_WIDTH, ZOOM_HEIGHT)

    # YOLO detection
    results         = detector.detect(frame)
    frame           = detector.draw_boxes(frame, results)
    detected_labels = detector.get_labels(results)

    # Update tracker counters
    tracker.update(detected_labels)

    # Caption thread
    current_time = time.time()
    with lock:
        can_run = (active_threads < MAX_THREADS
                   and (current_time - last_time) > CAPTION_INTERVAL)
    if can_run:
        with lock:
            active_threads += 1
            last_time = current_time
        threading.Thread(
            target = caption_worker,
            args   = (frame.copy(), detected_labels),
            daemon = True,
        ).start()

    # Compose and display
    canvas = build_frame(
        video_frame      = frame,
        caption          = last_caption,
        previous_caption = previous_caption,
        mode_key         = enhancer.current_mode,
        dropdown_open    = dropdown_open,
        hover_index      = hover_index,
        vision_mode      = detector.mode,
        target_w         = WIN_W,
        target_h         = WIN_H,
        # stats
        stats_open       = stats_open,
        top_counts       = tracker.top_counts(),
        seconds_left     = tracker.seconds_until_reset(),
        reset_btn_hover  = stats_reset_hover,
        stats_btn_hover  = stats_btn_hover,
    )

    cv2.imshow(WIN_NAME, canvas)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    if key == 27:
        dropdown_open = False
        hover_index   = -1

cap.release()
cv2.destroyAllWindows()