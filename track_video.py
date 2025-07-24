import cv2
import numpy as np
from ultralytics import YOLO

def iou(bb1, bb2):
    # Intersection over Union hesapla
    x1 = max(bb1[0], bb2[0])
    y1 = max(bb1[1], bb2[1])
    x2 = min(bb1[2], bb2[2])
    y2 = min(bb1[3], bb2[3])

    inter_area = max(0, x2 - x1) * max(0, y2 - y1)
    bb1_area = (bb1[2] - bb1[0]) * (bb1[3] - bb1[1])
    bb2_area = (bb2[2] - bb2[0]) * (bb2[3] - bb2[1])
    union_area = bb1_area + bb2_area - inter_area
    return inter_area / union_area if union_area != 0 else 0

class TrackedObject:
    def __init__(self, id, bbox):
        self.id = id
        self.kf = cv2.KalmanFilter(8, 4)
        self.kf.measurementMatrix = np.eye(4, 8, dtype=np.float32)
        self.kf.transitionMatrix = np.eye(8, dtype=np.float32)
        self.kf.processNoiseCov = np.eye(8, dtype=np.float32) * 0.01
        self.kf.statePre[:4, 0] = np.array(bbox, dtype=np.float32)
        self.kf.statePost[:4, 0] = np.array(bbox, dtype=np.float32)
        self.missing = 0  # eşleşemediği kare sayısı

    def predict(self):
        pred = self.kf.predict()
        return pred[:4].reshape(-1)

    def update(self, bbox):
        meas = np.array(bbox, dtype=np.float32)
        self.kf.correct(meas)
        self.missing = 0

class KalmanMultiTracker:
    def __init__(self, max_missing=5, iou_threshold=0.3):
        self.trackers = []
        self.next_id = 0
        self.max_missing = max_missing
        self.iou_threshold = iou_threshold

    def update(self, detections):
        matches = [-1] * len(detections)
        updated_tracks = []

        # Tüm trackleri predict et
        predictions = [trk.predict() for trk in self.trackers]

        # Eşleştirme (IoU en yüksek)
        for d, det in enumerate(detections):
            best_iou = self.iou_threshold
            best_t = -1
            for t, trk in enumerate(self.trackers):
                pred_box = predictions[t]
                iou_score = iou(det, pred_box)
                if iou_score > best_iou:
                    best_iou = iou_score
                    best_t = t
            if best_t != -1:
                self.trackers[best_t].update(det)
                matches[d] = self.trackers[best_t].id

        # Yeni track ekle
        for i, m in enumerate(matches):
            if m == -1:
                new_trk = TrackedObject(self.next_id, detections[i])
                self.trackers.append(new_trk)
                matches[i] = self.next_id
                self.next_id += 1

        # Kaybolanları sil
        new_trackers = []
        for trk in self.trackers:
            trk.missing += 1
            if trk.missing <= self.max_missing:
                new_trackers.append(trk)
        self.trackers = new_trackers

        # Geriye aktif tüm tracklerin bbox ve ID'lerini ver
        result = []
        for trk in self.trackers:
            bbox = trk.predict()
            result.append((trk.id, bbox.astype(int).tolist()))
        return result

# YOLO modeli yükle
model = YOLO("yolov8n.pt")
tracker = KalmanMultiTracker()

cap = cv2.VideoCapture("video.mp4")
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter("output.mp4", fourcc, 30.0, (int(cap.get(3)), int(cap.get(4))))

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame)[0]
    detections = []
    for box in results.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        detections.append([x1, y1, x2, y2])

    tracks = tracker.update(detections)

    for obj_id, bbox in tracks:
        x1, y1, x2, y2 = bbox
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f"ID {obj_id}", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    out.write(frame)
    cv2.imshow("Tracking", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
out.release()
cv2.destroyAllWindows()
