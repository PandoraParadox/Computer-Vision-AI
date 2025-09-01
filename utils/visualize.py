utils_visualize_py = r""""""
import random
from typing import List, Optional
import numpy as np
import cv2
from contracts import Detection, Track

_COLOR_CACHE = {}


def _color_for_id(i: int):
    if i not in _COLOR_CACHE:
        random.seed(i + 12345)
        _COLOR_CACHE[i] = tuple(int(random.random() * 255) for _ in range(3))
    return _COLOR_CACHE[i]


def draw_detections(img: np.ndarray, dets: List[Detection], class_names: Optional[List[str]] = None):
    out = img.copy()
    for d in dets:
        x1, y1, x2, y2 = map(int, d.xyxy)
        color = _color_for_id(d.cls)
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
        label = d.label if d.label is not None else (class_names[d.cls] if class_names else str(d.cls))
        if d.conf is not None:
            label = f"{label} {d.conf:.2f}"
        cv2.putText(out, label, (x1, max(0, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
    return out


def draw_tracks(img: np.ndarray, tracks: List[Track], class_names: Optional[List[str]] = None):
    out = img.copy()
    for t in tracks:
        x1, y1, x2, y2 = map(int, t.xyxy)
        color = _color_for_id(t.track_id)
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
        label = t.label if t.label is not None else (class_names[t.cls] if class_names else str(t.cls))
        text = f"ID {t.track_id}: {label} {t.conf:.2f}"
        cv2.putText(out, text, (x1, max(0, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
    return out
"""
Save as: utils/visualize.py
"""