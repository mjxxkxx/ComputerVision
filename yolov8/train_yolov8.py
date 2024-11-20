from ultralytics import YOLO

def train_model():
    model = YOLO("./yolov8n.pt")
    model.train(
        data="D:/yolov8/data.yaml",
        epochs=50,
        batch=8,
        imgsz=640,
        lr0=0.01,
        lrf=0.0001,
        optimizer="AdamW",
        project="D:/yolov8/runs/train_augmented",
        name="train_augmented_with_data",
        device=0
    )

if __name__ == "__main__":
    train_model()