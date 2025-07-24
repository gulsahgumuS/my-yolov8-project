# sort.py
from collections import deque
import numpy as np
from filterpy.kalman import KalmanFilter

class KalmanBoxTracker:
    count = 0

    def __init__(self, bbox):
        self.kf = KalmanFilter(dim_x=7, dim_z=4)
        self.kf.F = np.array([
            [1, 0, 0, 0, 1, 0, 0],
            [0, 1, 0, 0, 0, 1, 0],
            [0, 0, 1, 0, 0, 0, 1],
            [0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 1]
        ])
        self.kf.H = np.array([
            [1, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0]
        ])
        self.kf.R[2:,2:] *= 10.
        self.kf.P[4:,4:] *= 1000.
        self.kf.P *= 10.
        self.kf.Q[-1,-1] *= 0.01
        self.kf.Q[4:,4:] *= 0.01
        self.kf.x[:4] = self.convert_bbox_to_z(bbox)
        self.id = KalmanBoxTracker.count
        KalmanBoxTracker.count += 1
        self.hits = 0
        self.no_losses = 0

    def update(self, bbox):
        self.kf.update(self.convert_bbox_to_z(bbox))
        self.hits += 1

    def predict(self):
        self.kf.predict()
        return self.kf.x

    def get_state(self):
        return self.convert_x_to_bbox(self.kf.x)

    @staticmethod
    def convert_bbox_to_z(bbox):
        x = (bbox[0] + bbox[2]) / 2.0
        y = (bbox[1] + bbox[3]) / 2.0
        s = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
        r = (bbox[2] - bbox[0]) / (bbox[3] - bbox[1])
        return np.array([[x], [y], [s], [r]])

    @staticmethod
    def convert_x_to_bbox(x):
        w = np.sqrt(x[2] * x[3])
        h = x[2] / w
        x1 = x[0] - w / 2.
        y1 = x[1] - h / 2.
        x2 = x[0] + w / 2.
        y2 = x[1] + h / 2.
        return [int(x1), int(y1), int(x2), int(y2)]

class Sort:
    def __init__(self):
        self.trackers = []

    def update(self, detections):
        updated_tracks = []
        for det in detections:
            tracker = KalmanBoxTracker(det)
            tracker.predict()
            tracker.update(det)
            updated_tracks.append((tracker.id, tracker.get_state()))
        return updated_tracks
