import numpy as np
from typing import Tuple, Union

ArrayLike = Union[np.ndarray, Tuple[float, float, float, float]]


def xywh_to_xyxy(box: ArrayLike) -> np.ndarray:
    if isinstance(box, tuple) or isinstance(box, list):
        box = np.array(box, dtype=float)
    x, y, w, h = box
    return np.array([x, y, x + w, y + h], dtype=float)


def xyxy_to_xywh(box: ArrayLike) -> np.ndarray:
    if isinstance(box, tuple) or isinstance(box, list):
        box = np.array(box, dtype=float)
    x1, y1, x2, y2 = box
    return np.array([x1, y1, x2 - x1, y2 - y1], dtype=float)


def box_iou(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    N, M = a.shape[0], b.shape[0]
    ious = np.zeros((N, M), dtype=float)
    for i in range(N):
        ax1, ay1, ax2, ay2 = a[i]
        aa = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
        for j in range(M):
            bx1, by1, bx2, by2 = b[j]
            xx1 = max(ax1, bx1)
            yy1 = max(ay1, by1)
            xx2 = min(ax2, bx2)
            yy2 = min(ay2, by2)
            inter = max(0.0, xx2 - xx1) * max(0.0, yy2 - yy1)
            bb = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
            union = aa + bb - inter
            ious[i, j] = inter / union if union > 0 else 0.0
    return ious


def nms(boxes: np.ndarray, scores: np.ndarray, iou_thres: float = 0.5):
    """Pure numpy NMS. boxes: (N,4) xyxy."""
    idxs = scores.argsort()[::-1]
    keep = []
    while idxs.size > 0:
        i = idxs[0]
        keep.append(i)
        if idxs.size == 1:
            break
        ious = box_iou(boxes[i:i+1], boxes[idxs[1:]])[0]
        idxs = idxs[1:][ious < iou_thres]
    return keep
