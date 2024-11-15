from ultralytics import YOLO

# 학습된 YOLOv8 모델 로드
model = YOLO("C:/Users/Administrator/Documents/GitHub/ComputerVision/yolov8/runs/detect/train_results/weights/best.pt")

# 테스트할 이미지 경로
test_image_path = "C:/Users/Administrator/Documents/GitHub/ComputerVision/yolov8/images/val/0--Parade/0_Parade_marchingband_1_849.jpg"

# 추론 실행
results = model(test_image_path)

# 결과 시각화
results.show()

# 결과 저장 (선택적)
results.save("C:/Users/Administrator/Documents/GitHub/ComputerVision/yolov8/results/")
