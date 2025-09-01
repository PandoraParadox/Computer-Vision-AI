from typing import List
import numpy as np
from contracts import Detection, Track
from trackers.sort import Sort 

class DeepSort:
    def __init__(self, *args, **kwargs):
        self.sort = Sort(*args, **kwargs)

    def update(self, frame: np.ndarray, detections: List[Detection]) -> List[Track]:
        return self.sort.update(detections)

