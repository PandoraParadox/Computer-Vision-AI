import os
import argparse
import cv2
from typing import List
import numpy as np

from contracts import Detector, Track
from trackers.sort import Sort
from trackers.deep_sort.deep_sort import DeepSort
from utils.data_loader import iter_media
from utils.visualize import draw_tracks


def run(detector: Detector, source: str, tracker_type: str = 'sort', save_path: str = None, stride: int = 1):
    if tracker_type == 'deepsort':
        tracker = DeepSort()
    else:
        tracker = Sort()

    writer = None
    for i, frame in iter_media(source, stride=stride):
        # detection
        dets = detector.predict(frame[:, :, ::-1])  # convert BGR->RGB
        # tracking
        tracks = tracker.update(dets) if tracker_type == 'sort' else tracker.update(frame, dets)
        # visualize
        vis = draw_tracks(frame, tracks, detector.class_names)

        # write/show
        if save_path:
            if writer is None:
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                h, w = vis.shape[:2]
                writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), 30, (w, h))
            writer.write(vis)
        else:
            cv2.imshow('track', vis)
            if cv2.waitKey(1) == 27:
                break
    if writer is not None:
        writer.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=str, default='0')
    parser.add_argument('--weights', type=str, default='weights/best.pt')
    parser.add_argument('--tracker', type=str, default='sort', choices=['sort','deepsort'])
    parser.add_argument('--save', type=str, default='')
    args = parser.parse_args()

    from models.detector import load_detector  # implement this
    det = load_detector(args.weights)

    run(det, args.source, args.tracker, args.save or None)
