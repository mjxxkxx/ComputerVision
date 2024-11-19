from ultralytics import YOLO

# YOLOv8m 모델 로드
model = YOLO("D:/yolov8/yolov8m.pt")

# 모델 학습
model.train(
    data="D:/yolov8/data.yaml",
    epochs=100,
    batch=16,
    img_size=640
)

print("YOLOv8m 모델 학습 완료!")
