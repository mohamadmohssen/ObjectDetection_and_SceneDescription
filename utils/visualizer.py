import os
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from caption_enhancer import MODES
import platform

CAPTION_PANEL_HEIGHT = 140

#Modes button
MODE_BTN_W, MODE_BTN_H = 320, 70
MODE_BTN_MARGIN        = 16
DROPDOWN_ITEM_H        = 70
DROPDOWN_W             = 260

#vision button
VIS_BTN_W, VIS_BTN_H = 280, 70
VIS_BTN_MARGIN        = 16

#Stats button
STATS_BTN_W, STATS_BTN_H = 200, 70      
STATS_BTN_MARGIN          = 16

#Stats panel
STATS_PANEL_W      = 320   
STATS_BAR_MAX_W    = 200   
STATS_BAR_H        = 28    
STATS_BAR_GAP      = 14    
STATS_PADDING      = 18    
STATS_RESET_BTN_H  = 44    

#Colors
COLOR_PANEL_BG      = (28,  28,  28)
COLOR_BTN_NORMAL    = (50,  50,  50)
COLOR_BTN_BORDER    = (120, 120, 120)
COLOR_BTN_HOVER     = (80,  80,  80)
COLOR_DROP_BG       = (40,  40,  40)
COLOR_DROP_SEL      = (60,  110,  60)
COLOR_TEXT_WHITE    = (255, 255, 255)
COLOR_TEXT_GRAY     = (170, 170, 170)
COLOR_ACCENT        = (90,  190,  90)
COLOR_SEG_ACTIVE    = (60,  130, 200)    
COLOR_DET_ACTIVE    = (180, 110,  40)    
COLOR_LETTERBOX     = (0,   0,    0)     
COLOR_STATS_BG      = (22,  22,  38)    
COLOR_STATS_BORDER  = (60,  80,  120)    
COLOR_STATS_BAR     = (70,  160, 230)   
COLOR_STATS_BAR_BG  = (45,  45,  60)   
COLOR_STATS_RESET   = (160,  50,  50)    
COLOR_STATS_RESET_H = (200,  70,  70)    

def _find_font():
    candidates = []
    system = platform.system()
    if system == "Linux":
        candidates += ["/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"]
    elif system == "Darwin":
        candidates += ["/System/Library/Fonts/Supplemental/Arial.ttf",
                        "/Library/Fonts/Arial.ttf"]
    elif system == "Windows":
        candidates += ["C:\\Windows\\Fonts\\arial.ttf"]
    for path in candidates:
        if os.path.exists(path):
            return path
    return None


def _put_text_centered(img, text, cx, cy, font, scale, color, thickness=2):
    text = text.encode('utf-8').decode('utf-8')
    (tw, th), _ = cv2.getTextSize(text, font, scale, thickness)
    cv2.putText(img, text, (cx - tw // 2, cy + th // 2),
                font, scale, color, thickness, cv2.LINE_AA)
    


def _put_text_pil(img, text, cx, cy, font_size=30, color=(255, 255, 255)):
    pil_img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil_img)
    font_path = _find_font()
    try:
        font = ImageFont.truetype(font_path, font_size) if font_path else ImageFont.load_default()
    except Exception:
        font = ImageFont.load_default()
    
    bbox = draw.textbbox((0, 0), text, font=font)
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]
    
    x = cx - text_width // 2
    y = cy - text_height // 2
    draw.text((x, y), text, font=font, fill=color)
    
    return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)


def _letterbox(frame: np.ndarray, target_w: int, target_h: int) -> np.ndarray:
    """
    Scale 'frame' to fit inside a (target_w × target_h) canvas while
    keeping the original aspect ratio.  Empty space is filled with black.
    """
    src_h, src_w = frame.shape[:2]

    if src_w == 0 or src_h == 0:
        return np.zeros((target_h, target_w, 3), dtype=np.uint8)

    scale = min(target_w / src_w, target_h / src_h)
    new_w = int(src_w * scale)
    new_h = int(src_h * scale)

    resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    canvas = np.zeros((target_h, target_w, 3), dtype=np.uint8)
    y_off = (target_h - new_h) // 2
    x_off = (target_w - new_w) // 2
    canvas[y_off:y_off + new_h, x_off:x_off + new_w] = resized
    return canvas

def build_frame(
    video_frame:    np.ndarray,
    caption:        str,
    previous_caption: str,
    mode_key:       str,
    dropdown_open:  bool,
    hover_index:    int,
    vision_mode:    str,
    target_w:       int,
    target_h:       int,
    stats_open:     bool                   = False,
    top_counts:     list[tuple[str, int]]  = None,
    seconds_left:   int                    = 0,
    reset_btn_hover: bool                  = False,
    stats_btn_hover: bool                  = False,
) -> np.ndarray:

    if top_counts is None:
        top_counts = []

    canvas_w = target_w + (STATS_PANEL_W if stats_open else 0)
    video_area_h = target_h - CAPTION_PANEL_HEIGHT

    scaled_video = _letterbox(video_frame, target_w, video_area_h)

    #Caption panel
    panel = np.full((CAPTION_PANEL_HEIGHT, target_w, 3),
                    COLOR_PANEL_BG, dtype=np.uint8)
    cv2.line(panel, (0, 0), (target_w, 0), COLOR_ACCENT, 4)

    mode_label = MODES[mode_key]["label"]
    cv2.putText(panel, f"Mode: {mode_label}",
                (18, 32), cv2.FONT_HERSHEY_SIMPLEX,
                0.75, COLOR_TEXT_GRAY, 1, cv2.LINE_AA)

    if previous_caption:
        panel = _put_text_pil(panel, previous_caption,
                              target_w // 2, CAPTION_PANEL_HEIGHT // 2 - 25,
                              font_size=32, color=(170, 170, 170))

    if caption:
        panel = _put_text_pil(panel, caption,
                              target_w // 2, CAPTION_PANEL_HEIGHT // 2 + 18,
                              font_size=42, color=(255, 255, 255))

    video_col = np.vstack([scaled_video, panel])

    #stats panel
    if stats_open:
        side = _build_stats_panel(
            panel_w      = STATS_PANEL_W,
            panel_h      = target_h,
            top_counts   = top_counts,
            seconds_left = seconds_left,
            reset_hover  = reset_btn_hover,
        )
        canvas = np.hstack([video_col, side])
    else:
        canvas = video_col

    _draw_vision_button(canvas, VIS_BTN_MARGIN, VIS_BTN_MARGIN, vision_mode)

    btn_x = target_w - MODE_BTN_W - MODE_BTN_MARGIN
    btn_y = MODE_BTN_MARGIN
    _draw_mode_button(canvas, btn_x, btn_y, mode_label)

    if dropdown_open:
        _draw_dropdown(canvas, btn_x, btn_y, mode_key, hover_index)

    # Stats toggle button — centred between mode button and right video edge
    _draw_stats_button(canvas, target_w, stats_open, stats_btn_hover)

    return canvas


#stats panel builder
def _build_stats_panel(
    panel_w:      int,
    panel_h:      int,
    top_counts:   list[tuple[str, int]],
    seconds_left: int,
    reset_hover:  bool,
) -> np.ndarray:
    panel = np.full((panel_h, panel_w, 3), COLOR_STATS_BG, dtype=np.uint8)

    #border line
    cv2.line(panel, (0, 0), (0, panel_h), COLOR_STATS_BORDER, 3)

    #title
    y = STATS_PADDING + 24
    cv2.putText(panel, "Object Stats",
                (STATS_PADDING, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.85, COLOR_TEXT_WHITE, 2, cv2.LINE_AA)

    #subtitle
    mins, secs = divmod(seconds_left, 60)
    timer_text = f"Reset in {mins}:{secs:02d}"
    cv2.putText(panel, timer_text,
                (STATS_PADDING, y + 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.52, COLOR_TEXT_GRAY, 1, cv2.LINE_AA)

    #divider
    div_y = y + 46
    cv2.line(panel, (STATS_PADDING, div_y),
             (panel_w - STATS_PADDING, div_y), COLOR_STATS_BORDER, 1)

    #barcharts
    max_count = max((c for _, c in top_counts), default=1)
    bar_y = div_y + STATS_BAR_GAP + 10

    if not top_counts:
        cv2.putText(panel, "No detections yet",
                    (STATS_PADDING, bar_y + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, COLOR_TEXT_GRAY, 1, cv2.LINE_AA)
    else:
        for label, count in top_counts:
            # Label above bar
            cv2.putText(panel, label,
                        (STATS_PADDING, bar_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, COLOR_TEXT_WHITE, 1, cv2.LINE_AA)

            bar_y += STATS_BAR_H - 8

            #Background
            bar_bg_x2 = STATS_PADDING + STATS_BAR_MAX_W
            cv2.rectangle(panel,
                          (STATS_PADDING, bar_y),
                          (bar_bg_x2, bar_y + STATS_BAR_H),
                          COLOR_STATS_BAR_BG, -1)

            #Filled
            filled_w = int(STATS_BAR_MAX_W * count / max_count)
            if filled_w > 0:
                cv2.rectangle(panel,
                              (STATS_PADDING, bar_y),
                              (STATS_PADDING + filled_w, bar_y + STATS_BAR_H),
                              COLOR_STATS_BAR, -1)

            #Count label
            count_x = bar_bg_x2 + 8
            cv2.putText(panel, str(count),
                        (count_x, bar_y + STATS_BAR_H - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLOR_TEXT_WHITE, 1, cv2.LINE_AA)

            bar_y += STATS_BAR_H + STATS_BAR_GAP

    #Rese button
    reset_btn_y = panel_h - CAPTION_PANEL_HEIGHT - STATS_RESET_BTN_H - 20
    rbx1 = STATS_PADDING
    rby1 = reset_btn_y
    rbx2 = panel_w - STATS_PADDING
    rby2 = reset_btn_y + STATS_RESET_BTN_H

    btn_color = COLOR_STATS_RESET_H if reset_hover else COLOR_STATS_RESET
    cv2.rectangle(panel, (rbx1, rby1), (rbx2, rby2), btn_color, -1)
    cv2.rectangle(panel, (rbx1, rby1), (rbx2, rby2), COLOR_BTN_BORDER, 1)
    _put_text_centered(panel, "↺  Reset Counts",
                       (rbx1 + rbx2) // 2, (rby1 + rby2) // 2,
                       cv2.FONT_HERSHEY_SIMPLEX, 0.65, COLOR_TEXT_WHITE, 1)

    return panel


def _draw_stats_button(canvas, video_col_w: int, stats_open: bool, hovered: bool):

    x = video_col_w - MODE_BTN_W - STATS_BTN_W - MODE_BTN_MARGIN * 2 - 8
    y = STATS_BTN_MARGIN

    fill   = (60, 80, 130) if stats_open else ((70, 70, 90) if hovered else COLOR_BTN_NORMAL)
    border = (100, 140, 200) if stats_open else COLOR_BTN_BORDER
    label  = "[Stats] ON" if stats_open else "[Stats] OFF"
    cv2.rectangle(canvas,
                  (x + 3, y + 3), (x + STATS_BTN_W + 3, y + STATS_BTN_H + 3),
                  (0, 0, 0), -1)
    cv2.rectangle(canvas, (x, y), (x + STATS_BTN_W, y + STATS_BTN_H), fill, -1)
    cv2.rectangle(canvas, (x, y), (x + STATS_BTN_W, y + STATS_BTN_H), border, 2)

    _put_text_centered(canvas, label,
                       x + STATS_BTN_W // 2, y + STATS_BTN_H // 2,
                       cv2.FONT_HERSHEY_SIMPLEX, 0.65, COLOR_TEXT_WHITE, 1)


def _draw_vision_button(canvas, x, y, vision_mode: str):
    is_seg = (vision_mode == "segmentation")
    color  = COLOR_SEG_ACTIVE if is_seg else COLOR_DET_ACTIVE
    label  = "Segmentation"   if is_seg else "Detection"
    icon   = "[SEG]"          if is_seg else "[DET]"

    cv2.rectangle(canvas,
                  (x + 3, y + 3),
                  (x + VIS_BTN_W + 3, y + VIS_BTN_H + 3),
                  (0, 0, 0), -1)
    fill = tuple(max(0, c - 30) for c in color)
    cv2.rectangle(canvas, (x, y), (x + VIS_BTN_W, y + VIS_BTN_H), fill, -1)
    cv2.rectangle(canvas, (x, y), (x + VIS_BTN_W, y + VIS_BTN_H), color, 2)

    cv2.putText(canvas, icon,
                (x + 12, y + 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)
    cv2.putText(canvas, label,
                (x + 12, y + VIS_BTN_H - 14),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, COLOR_TEXT_WHITE, 1, cv2.LINE_AA)

    hint = "Detection" if is_seg else "Segmentation"
    cv2.putText(canvas, f"Switch to {hint}",
                (x + VIS_BTN_W + 8, y + VIS_BTN_H // 2 + 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, COLOR_TEXT_GRAY, 1, cv2.LINE_AA)


def _draw_mode_button(canvas, x, y, label: str):
    cv2.rectangle(canvas,
                  (x + 3, y + 3),
                  (x + MODE_BTN_W + 3, y + MODE_BTN_H + 3),
                  (0, 0, 0), -1)
    cv2.rectangle(canvas, (x, y), (x + MODE_BTN_W, y + MODE_BTN_H),
                  COLOR_BTN_NORMAL, -1)
    cv2.rectangle(canvas, (x, y), (x + MODE_BTN_W, y + MODE_BTN_H),
                  COLOR_BTN_BORDER, 2)
    cv2.putText(canvas, f"[Mode] {label}",
                (x + 14, y + MODE_BTN_H - 18),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, COLOR_TEXT_WHITE, 1, cv2.LINE_AA)
    ax  = x + MODE_BTN_W - 18
    ay  = y + MODE_BTN_H // 2
    pts = np.array([[ax, ay - 6], [ax + 11, ay - 6], [ax + 5, ay + 6]])
    cv2.fillPoly(canvas, [pts], COLOR_TEXT_GRAY)


def _draw_dropdown(canvas, btn_x, btn_y, current_key, hover_index):
    mode_items = list(MODES.items())
    n      = len(mode_items)
    drop_x = btn_x
    drop_y = btn_y + MODE_BTN_H + 6
    drop_h = n * DROPDOWN_ITEM_H

    cv2.rectangle(canvas,
                  (drop_x + 3, drop_y + 3),
                  (drop_x + DROPDOWN_W + 3, drop_y + drop_h + 3),
                  (0, 0, 0), -1)
    cv2.rectangle(canvas, (drop_x, drop_y),
                  (drop_x + DROPDOWN_W, drop_y + drop_h), COLOR_DROP_BG, -1)
    cv2.rectangle(canvas, (drop_x, drop_y),
                  (drop_x + DROPDOWN_W, drop_y + drop_h), COLOR_BTN_BORDER, 2)

    for i, (key, meta) in enumerate(mode_items):
        item_y = drop_y + i * DROPDOWN_ITEM_H
        if key == current_key:
            cv2.rectangle(canvas, (drop_x, item_y),
                          (drop_x + DROPDOWN_W, item_y + DROPDOWN_ITEM_H),
                          COLOR_DROP_SEL, -1)
        elif i == hover_index:
            cv2.rectangle(canvas, (drop_x, item_y),
                          (drop_x + DROPDOWN_W, item_y + DROPDOWN_ITEM_H),
                          COLOR_BTN_HOVER, -1)
        if i > 0:
            cv2.line(canvas,
                     (drop_x + 10, item_y),
                     (drop_x + DROPDOWN_W - 10, item_y),
                     (65, 65, 65), 1)
        cv2.putText(canvas, meta["label"],
                    (drop_x + 16, item_y + DROPDOWN_ITEM_H - 13),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65,
                    COLOR_TEXT_WHITE, 1, cv2.LINE_AA)

def get_mode_button_rect(win_w: int):
    x = win_w - MODE_BTN_W - MODE_BTN_MARGIN
    y = MODE_BTN_MARGIN
    return x, y, x + MODE_BTN_W, y + MODE_BTN_H


def get_vision_button_rect():
    return (VIS_BTN_MARGIN, VIS_BTN_MARGIN,
            VIS_BTN_MARGIN + VIS_BTN_W, VIS_BTN_MARGIN + VIS_BTN_H)


def get_stats_button_rect(video_col_w: int):
    x = video_col_w - MODE_BTN_W - STATS_BTN_W - MODE_BTN_MARGIN * 2 - 8
    y = STATS_BTN_MARGIN
    return x, y, x + STATS_BTN_W, y + STATS_BTN_H


def get_stats_reset_btn_rect(video_col_w: int, win_h: int):
    reset_btn_y = win_h - CAPTION_PANEL_HEIGHT - STATS_RESET_BTN_H - 20
    rbx1 = video_col_w + STATS_PADDING
    rby1 = reset_btn_y
    rbx2 = video_col_w + STATS_PANEL_W - STATS_PADDING
    rby2 = reset_btn_y + STATS_RESET_BTN_H
    return rbx1, rby1, rbx2, rby2


def get_dropdown_item_at(win_w: int, mx: int, my: int) -> int:
    bx, by, bx2, by2 = get_mode_button_rect(win_w)
    drop_x = bx
    drop_y = by2 + 6
    drop_h = len(MODES) * DROPDOWN_ITEM_H
    if drop_x <= mx <= drop_x + DROPDOWN_W and drop_y <= my <= drop_y + drop_h:
        return (my - drop_y) // DROPDOWN_ITEM_H
    return -1



def get_button_rect(win_w: int):
    return get_mode_button_rect(win_w)