from ultralytics import YOLO

def train():
    model = YOLO('yolov8n.pt')
    model.train(
        data='../data.yaml',
        epochs=10,
        imgsz=640,
        batch=16,
        device='cpu'  # Varsa '0' yaz GPU için
    )

if __name__ == "__main__":
    train()
