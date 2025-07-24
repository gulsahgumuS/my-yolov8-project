from ultralytics import YOLO

model = YOLO("runs/detect/train5/weights/best.pt")

model.track(
    source="video.mp4",         
    tracker="bytetrack.yaml",   
    show=True,                  
    save=True                  
)
