import cv2
import numpy as np
from ultralytics import YOLO
from filterpy.kalman import KalmanFilter

def iou(bb_test, bb_gt):
    xx1 = max(bb_test[0], bb_gt[0])
    yy1 = max(bb_test[1], bb_gt[1])
    xx2 = min(bb_test[2], bb_gt[2])
    yy2 = min(bb_test[3], bb_gt[3])
    w = max(0., xx2 - xx1)
    h = max(0., yy2 - yy1)
    inter = w * h
    area1 = (bb_test[2] - bb_test[0]) * (bb_test[3] - bb_test[1])
    area2 = (bb_gt[2] - bb_gt[0]) * (bb_gt[3] - bb_gt[1])
    o = inter / float(area1 + area2 - inter)
    return o

class Track:
    def __init__(self, bbox, track_id):
        self.kf = KalmanFilter(dim_x=7, dim_z=4)
        dt = 1.0
        self.kf.F = np.array([[1,0,0,0,dt,0,0],
                              [0,1,0,0,0,dt,0],
                              [0,0,1,0,0,0,dt],
                              [0,0,0,1,0,0,0],
                              [0,0,0,0,1,0,0],
                              [0,0,0,0,0,1,0],
                              [0,0,0,0,0,0,1]])
        self.kf.H = np.eye(4,7)

        self.kf.R[2:,2:] *= 10.
        self.kf.P[4:,4:] *= 1000.
        self.kf.Q[-1,-1] *= 0.01

        self.kf.x[:4] = np.array(bbox).reshape((4,1))

        self.id = track_id
        self.time_since_update = 0
        self.hit_streak = 0
        self.age = 0
        self.history = []

    def predict(self):
        self.kf.predict()
        self.age += 1
        if self.time_since_update > 0:
            self.hit_streak = 0
        self.time_since_update += 1
        self.history.append(self.kf.x)
        return self.kf.x

    def update(self, bbox):
        self.time_since_update = 0
        self.hit_streak += 1
        self.kf.update(np.array(bbox).reshape((4,1)))

    def get_state(self):
        return self.kf.x[:4].reshape(-1)

class Tracker:
    def __init__(self, iou_threshold=0.3, max_age=5, min_hits=3):
        self.iou_threshold = iou_threshold
        self.max_age = max_age
        self.min_hits = min_hits
        self.tracks = []
        self.next_id = 0

    def update(self, detections):
        # Predict existing tracks
        for track in self.tracks:
            track.predict()

        N = len(self.tracks)
        M = len(detections)

        iou_matrix = np.zeros((N, M), dtype=np.float32)

        for t, track in enumerate(self.tracks):
            for d, det in enumerate(detections):
                iou_matrix[t, d] = iou(track.get_state(), det)

        matched_indices = []

        if iou_matrix.size > 0:
            from scipy.optimize import linear_sum_assignment
            row_ind, col_ind = linear_sum_assignment(-iou_matrix)
            for r, c in zip(row_ind, col_ind):
                if iou_matrix[r, c] < self.iou_threshold:
                    continue
                matched_indices.append((r, c))

        unmatched_tracks = set(range(N)) - set(r for r, _ in matched_indices)
        unmatched_detections = set(range(M)) - set(c for _, c in matched_indices)

        # Update matched tracks with assigned detections
        for track_idx, det_idx in matched_indices:
            self.tracks[track_idx].update(detections[det_idx])

        # Create new tracks for unmatched detections
        for det_idx in unmatched_detections:
            self.tracks.append(Track(detections[det_idx], self.next_id))
            self.next_id += 1

        # Remove dead tracks
        self.tracks = [t for t in self.tracks if t.time_since_update <= self.max_age]

        # Return tracks which have hit_streak > min_hits (to avoid noise)
        results = []
        for track in self.tracks:
            if track.hit_streak >= self.min_hits:
                bbox = track.get_state()
                results.append((track.id, [int(x) for x in bbox]))
        return results


# --- Ana kod ---

model = YOLO('yolov8n.pt')
cap = cv2.VideoCapture("output.mp4")
tracker = Tracker()

if not cap.isOpened():
    print("Video açılamadı!")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame)[0]

    detections = []
    for box in results.boxes:
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        detections.append([x1, y1, x2, y2])

    tracked_objects = tracker.update(detections)

    for obj_id, bbox in tracked_objects:
        x1, y1, x2, y2 = map(int, bbox)
        cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,0), 2)
        cv2.putText(frame, f"ID: {obj_id}", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

    cv2.imshow("YOLOv8 + Kalman Tracker", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("Bitti.")
