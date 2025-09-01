trackers_sort_py = r""""""
"""SORT tracker (Kalman Filter + Hungarian) – model-agnostic.
Reference: https://arxiv.org/abs/1602.00763
Simplified implementation for educational / portable use.
"""

from dataclasses import dataclass
from typing import List, Optional
import numpy as np
from scipy.optimize import linear_sum_assignment
from filterpy.kalman import KalmanFilter
from contracts import Detection, Track

# --- Helper conversions ----------------------------------------------------

def xyxy_to_z(box):
    """Measurement vector z = [cx, cy, s, r].
    s = area = w*h, r = aspect = w/h."""
    x1, y1, x2, y2 = box
    w = max(0., x2 - x1)
    h = max(0., y2 - y1)
    cx = x1 + w / 2.
    cy = y1 + h / 2.
    s = w * h
    r = w / h if h > 0 else 1.
    return np.array([cx, cy, s, r])


def z_to_xyxy(z):
    cx, cy, s, r = z
    w = np.sqrt(s * r)
    h = s / w if w > 0 else 0.
    x1 = cx - w / 2.
    y1 = cy - h / 2.
    x2 = cx + w / 2.
    y2 = cy + h / 2.
    return np.array([x1, y1, x2, y2])

# --- Track State -----------------------------------------------------------

class SortTrack:
    _count = 0

    def __init__(self, det: Detection, max_age=30, min_hits=3):
        self.kf = self._init_kf(xyxy_to_z(det.xyxy))
        self.time_since_update = 0
        self.hits = 1
        self.hit_streak = 1
        self.age = 0
        self.id = SortTrack._count
        SortTrack._count += 1
        self.last_det = det
        self.max_age = max_age
        self.min_hits = min_hits

    @staticmethod
    def _init_kf(z):
        kf = KalmanFilter(dim_x=7, dim_z=4)
        # state: [cx, cy, s, r, vx, vy, vs]
        kf.F = np.array([
            [1,0,0,0,1,0,0],
            [0,1,0,0,0,1,0],
            [0,0,1,0,0,0,1],
            [0,0,0,1,0,0,0],
            [0,0,0,0,1,0,0],
            [0,0,0,0,0,1,0],
            [0,0,0,0,0,0,1],
        ])
        kf.H = np.array([
            [1,0,0,0,0,0,0],
            [0,1,0,0,0,0,0],
            [0,0,1,0,0,0,0],
            [0,0,0,1,0,0,0],
        ])
        kf.R[2:,2:] *= 10.
        kf.P[4:,4:] *= 1000.
        kf.P *= 10.
        kf.x[:4] = z.reshape(-1,1)
        return kf

    def predict(self):
        self.kf.predict()
        self.age += 1
        if self.time_since_update > 0:
            self.hit_streak = 0
        self.time_since_update += 1
        return self

    def update(self, det: Detection):
        self.time_since_update = 0
        self.hits += 1
        self.hit_streak += 1
        self.last_det = det
        z = xyxy_to_z(det.xyxy)
        self.kf.update(z)

    def get_state(self):
        return z_to_xyxy(self.kf.x[:4].reshape(-1))

    def to_public(self) -> Track:
        # prefer last det conf/cls
        xyxy = tuple(map(float, self.get_state()))
        return Track(track_id=self.id,
                     xyxy=xyxy,
                     cls=self.last_det.cls,
                     conf=self.last_det.conf,
                     label=self.last_det.label,
                     is_confirmed=self.hits >= self.min_hits)

# --- SORT Manager ----------------------------------------------------------

class Sort:
    def __init__(self, max_age=30, min_hits=3, iou_threshold=0.3):
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.tracks: List[SortTrack] = []

    def update(self, detections: List[Detection]) -> List[Track]:
        # Predict all tracks forward
        for t in self.tracks:
            t.predict()

        # Build cost matrix using IOU
        det_boxes = np.array([d.xyxy for d in detections]) if detections else np.zeros((0,4))
        trk_boxes = np.array([t.get_state() for t in self.tracks]) if self.tracks else np.zeros((0,4))

        if det_boxes.shape[0] and trk_boxes.shape[0]:
            ious = iou_batch(trk_boxes, det_boxes)
            # Hungarian wants cost min; we use 1-IOU
            cost = 1 - ious
            row_idx, col_idx = linear_sum_assignment(cost)
        else:
            row_idx, col_idx = np.array([]), np.array([])

        unmatched_trks = list(range(len(self.tracks)))
        unmatched_dets = list(range(len(detections)))
        matches = []

        for r, c in zip(row_idx, col_idx):
            if 1 - cost[r, c] < self.iou_threshold:  # iou < thres
                continue
            matches.append((r, c))
            unmatched_trks.remove(r)
            unmatched_dets.remove(c)

        # update matched tracks
        for trk_idx, det_idx in matches:
            self.tracks[trk_idx].update(detections[det_idx])

        # create new tracks for unmatched detections
        for det_idx in unmatched_dets:
            self.tracks.append(SortTrack(detections[det_idx], self.max_age, self.min_hits))

        # age out dead tracks
        alive = []
        for t in self.tracks:
            if t.time_since_update < self.max_age:
                alive.append(t)
        self.tracks = alive

        # collect public tracks
        out = [t.to_public() for t in self.tracks if (t.hits >= self.min_hits or t.time_since_update == 0)]
        return out

# vectorized IoU helper
def iou_batch(a, b):
    iou = np.zeros((a.shape[0], b.shape[0]), dtype=float)
    for i in range(a.shape[0]):
        ax1, ay1, ax2, ay2 = a[i]
        aa = max(0., ax2-ax1) * max(0., ay2-ay1)
        for j in range(b.shape[0]):
            bx1, by1, bx2, by2 = b[j]
            xx1 = max(ax1, bx1)
            yy1 = max(ay1, by1)
            xx2 = min(ax2, bx2)
            yy2 = min(ay2, by2)
            inter = max(0., xx2-xx1) * max(0., yy2-yy1)
            bb = max(0., bx2-bx1) * max(0., by2-by1)
            union = aa + bb - inter
            iou[i,j] = inter/union if union>0 else 0.
    return iou
"""
Save as: trackers/sort.py
"""