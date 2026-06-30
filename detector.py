import cv2
import numpy as np
from ultralytics import YOLO

_PALETTE = [
    (255,  56,  56), (255, 157,  51), (255, 225,  56), ( 81, 240,  77),
    ( 77, 240, 240), ( 77, 120, 255), (153,  77, 255), (255,  77, 204),
    (255, 153, 153), (204, 255, 153), (153, 255, 255), (204, 153, 255),
]


def _color(cls_id: int):
    return _PALETTE[cls_id % len(_PALETTE)]


def _draw_label(frame, text, x, y, color):
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale, thickness = 0.55, 1
    (tw, th), _ = cv2.getTextSize(text, font, scale, thickness)
    pad = 4
    by1 = max(y - th - pad * 2, 0)
    by2 = max(y, th + pad * 2)
    cv2.rectangle(frame, (x, by1), (x + tw + pad * 2, by2), color, -1)
    cv2.putText(frame, text, (x + pad, by2 - pad),
                font, scale, (255, 255, 255), thickness, cv2.LINE_AA)


DETECTION_MODEL    = "yolov8m.pt"
SEGMENTATION_MODEL = "yolov8m-seg.pt"
MODE_DETECTION     = "detection"
MODE_SEGMENTATION  = "segmentation"

#0.0 = invisible, 1.0 = fully opaque.
SEG_ALPHA = 0.25


class YOLODetector:
    def __init__(self):
        print("[INFO] Loading segmentation model...")
        self._seg_model = YOLO(SEGMENTATION_MODEL)
        print("[INFO] Loading detection model...")
        self._det_model = YOLO(DETECTION_MODEL)
        self.mode       = MODE_SEGMENTATION   # default
        print(f"[INFO] Detector ready. Mode: {self.mode}")

    @property
    def model(self):
        return self._seg_model if self.mode == MODE_SEGMENTATION else self._det_model

    def switch_mode(self, mode: str):
        if mode in (MODE_DETECTION, MODE_SEGMENTATION):
            self.mode = mode
            print(f"[INFO] Switched to: {self.mode}")

    def toggle(self):
        self.mode = (MODE_SEGMENTATION
                     if self.mode == MODE_DETECTION
                     else MODE_DETECTION)
        print(f"[INFO] Toggled to: {self.mode}")

    def detect(self, frame):
        return self.model(frame)[0]

    def draw_boxes(self, frame: np.ndarray, results) -> np.ndarray:
        if self.mode == MODE_SEGMENTATION and results.masks is not None:
            return self._draw_segmentation(frame, results)
        return self._draw_detection(frame, results)

    # Segmentation

    def _draw_segmentation(self, frame: np.ndarray, results) -> np.ndarray:
        h, w = frame.shape[:2]

        color_layer = np.zeros_like(frame, dtype=np.uint8)
        combined_mask = np.zeros((h, w), dtype=np.uint8)

        label_draws: list[tuple[int, int, int, str, tuple]] = []

        masks = results.masks.data.cpu().numpy()
        for i, mask in enumerate(masks):
            box  = results.boxes[i]
            conf = float(box.conf[0])
            if conf < 0.5:
                continue

            cls_id = int(box.cls[0])
            color  = _color(cls_id)
            label  = self.model.names[cls_id]
            # Resize mask to frame size
            mask_r   = cv2.resize(mask, (w, h))
            bin_mask = (mask_r > 0.5).astype(np.uint8)

            # Paint this object's colour into the layer
            color_layer[bin_mask == 1] = color
            combined_mask = cv2.bitwise_or(combined_mask, bin_mask)

            contours, _ = cv2.findContours(bin_mask, cv2.RETR_EXTERNAL,
                                            cv2.CHAIN_APPROX_SIMPLE)
            x1, y1 = int(box.xyxy[0][0]), int(box.xyxy[0][1])
            label_draws.append((x1, y1, contours, f"{label} {conf:.2f}", color))

        output = frame.copy()
        mask_bool = combined_mask.astype(bool)
        output[mask_bool] = cv2.addWeighted(
            color_layer, SEG_ALPHA,
            frame,       1.0 - SEG_ALPHA,
            0
        )[mask_bool]

        #labels on top of image
        for x1, y1, contours, text, color in label_draws:
            cv2.drawContours(output, contours, -1, color, 2)
            _draw_label(output, text, x1, y1, color)

        return output

    #Detection

    def _draw_detection(self, frame: np.ndarray, results) -> np.ndarray:
        output = frame.copy()
        for box in results.boxes:
            conf = float(box.conf[0])
            if conf < 0.5:
                continue
            cls_id = int(box.cls[0])
            color  = _color(cls_id)
            label  = self.model.names[cls_id]
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cv2.rectangle(output, (x1, y1), (x2, y2), color, 2)
            _draw_label(output, f"{label} {conf:.2f}", x1, y1, color)
        return output

    def get_labels(self, results) -> list[str]:
        labels = []
        for box in results.boxes:
            if float(box.conf[0]) > 0.5:

                labels.append(self.model.names[int(box.cls[0])])
        try:
            labels = [l for l in labels if l not in ['chair','tv']]
            
        except:
            pass
        
        # print(list(set(labels)))
        return list(set(labels))