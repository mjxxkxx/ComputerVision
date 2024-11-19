from ultralytics import YOLO

def train_model():
    # YOLO 모델 로드
    model = YOLO("D:/yolov8/yolov8n.pt")  # YOLO 모델 경로

    # 학습 시작
    model.train(
        data="D:/yolov8/data.yaml",  # 업데이트된 데이터 구성 파일 경로
        epochs=50,  # 에포크 수
        batch=8,  # 배치 크기
        imgsz=640,  # 이미지 크기
        lr0=0.01,  # 초기 학습률
        lrf=0.01,
        optimizer="AdamW",  # 옵티마이저
        project="D:/yolov8/runs/train_augmented",
        name="train_augmented_with_data",
        device=0
    )

if __name__ == "__main__":
    train_model()