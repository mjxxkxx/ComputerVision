from ultralytics import YOLO

# 학습된 모델 로드
model = YOLO("D:/yolov8/runs/train_augmented/train_augmented/weights/best.pt")  # 재학습된 모델 가중치

# 평가 실행
results = model.val(
    data="D:/yolov8/data.yaml",  # 데이터셋 정의 파일
    batch=16,  # 배치 크기
    imgsz=640,  # 이미지 크기
    conf=0.25  # Confidence Threshold
)

print("평가 완료!")
