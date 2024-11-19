from ultralytics import YOLO

def train_model():
    # YOLO 모델 로드
    model = YOLO("yolov8n.pt")  # 필요한 경우 yolov8m.pt, yolov8l.pt 사용

    # 학습 시작
    model.train(
        data="D:/yolov8/data.yaml",
        epochs=50,  # 학습 에포크
        batch=8,  # 배치 크기
        imgsz=416,  # 이미지 크기
        save=True,
        cache=True,
        device=0,  # GPU 사용
        half=True,
        workers=0,
        project="D:/yolov8/runs/train_augmented",
        name="train_augmented"
    )

if __name__ == "__main__":
    train_model()