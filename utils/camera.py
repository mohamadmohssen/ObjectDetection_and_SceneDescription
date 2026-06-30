import cv2
import sys
# FOURCC codes
FOURCC_MJPG = cv2.VideoWriter_fourcc(*'MJPG')
FOURCC_YUYV = cv2.VideoWriter_fourcc(*'YUYV')

CAM_W   = 1920 
CAM_H   =  1080
CAM_FPS =   30

FALLBACK_MODES = [
    (FOURCC_MJPG, 1920, 1080, 60),
    (FOURCC_MJPG, 1920, 1080, 30),
    (FOURCC_MJPG, 1280,  720, 60),
    (FOURCC_MJPG, 1280,  720, 30),
    (FOURCC_YUYV, 1280,  720, 30),
    (FOURCC_YUYV,  640,  480, 30),
]


def _try_mode(cap: cv2.VideoCapture,
              fourcc: int, w: int, h: int, fps: int) -> tuple[bool, int, int, float]:

    cap.set(cv2.CAP_PROP_FOURCC,        fourcc)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,   w)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT,  h)
    cap.set(cv2.CAP_PROP_FPS,           fps)

    # 3 frames at first to adjust the camera
    for _ in range(3):
        cap.grab()

    ok, frame = cap.read()
    if not ok or frame is None:
        return False, 0, 0, 0.0

    fh, fw = frame.shape[:2]
    if fw == 0 or fh == 0:
        return False, 0, 0, 0.0

    actual_fps = cap.get(cv2.CAP_PROP_FPS)
    return True, fw, fh, actual_fps


def _platform_backend():
    if sys.platform.startswith("linux"):
        return cv2.CAP_V4L2
    elif sys.platform == "darwin":
        return cv2.CAP_AVFOUNDATION
    elif sys.platform.startswith("win"):
        return cv2.CAP_DSHOW
    return cv2.CAP_ANY


def open_camera(
    index:         int = 0,
    preferred_w:   int = CAM_W,
    preferred_h:   int = CAM_H,
    preferred_fps: int = CAM_FPS,
) -> tuple[cv2.VideoCapture, int, int, float]:
   
    cap = cv2.VideoCapture(index, _platform_backend())
    if not cap.isOpened():
        cap = cv2.VideoCapture(index)
    if not cap.isOpened():
        raise RuntimeError(f"[CAMERA] Cannot open camera at index {index} (/dev/video{index}).")

    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    preferred_fourcc = FOURCC_MJPG
    modes_to_try = [(preferred_fourcc, preferred_w, preferred_h, preferred_fps)] + [
        m for m in FALLBACK_MODES
        if m != (preferred_fourcc, preferred_w, preferred_h, preferred_fps)
    ]

    for fourcc, w, h, fps in modes_to_try:
        fmt_name = "MJPG" if fourcc == FOURCC_MJPG else "YUYV"
        ok, aw, ah, afps = _try_mode(cap, fourcc, w, h, fps)
        if not ok:
            print(f"[CAMERA] {fmt_name} {w}×{h}@{fps}, no frame, skipping.")
            continue

        marker = "Worked with" if (aw == w and ah == h) else "Fallback to"
        print(f"[CAMERA] {marker} {fmt_name} {w}×{h}@{fps} → got {aw}×{ah}@{afps:.0f}fps")
        return cap, aw, ah, afps

    cap.release()
    raise RuntimeError("[CAMERA] Exhausted all modes — no usable camera stream found.")


def auto_detect_camera(device_path: str | None = None) -> int:

    if device_path is None:
        return 0
    try:
        return int(device_path.replace('/dev/video', ''))
    except ValueError:
        return 0
    
def camera_zoom(frame, crop_w, crop_h):
    h, w = frame.shape[:2]
    crop_w = max(1, min(crop_w, w))
    crop_h = max(1, min(crop_h, h))
    x1 = (w - crop_w)//2
    y1= (h - crop_h) //2
    crop = frame[y1:y1+crop_h, x1: x1+crop_w]
    return cv2.resize(crop, (w,h), interpolation = cv2.INTER_LINEAR)