from ultralytics import YOLO

# 학습된 YOLOv8 모델 로드
model = YOLO("D:/yolov8/runs/detect/train_results/weights/best.pt")

# 테스트할 이미지 경로
test_image_path = "D:/yolov8/images/val/0--Parade/0_Parade_marchingband_1_849.jpg"

# 추론 실행
results = model(test_image_path)

# 결과 시각화
results.show()

# 결과 저장 (선택적)
results.save("D:/yolov8/results/")


# 50 epochs completed in 2.191 hours.
# Optimizer stripped from YOLOv8_WIDERFace\train_results\weights\last.pt, 6.2MB
# Optimizer stripped from YOLOv8_WIDERFace\train_results\weights\best.pt, 6.2MB

# Validating YOLOv8_WIDERFace\train_results\weights\best.pt...
# Ultralytics 8.3.31 🚀 Python-3.11.10 torch-2.5.1+cu118 CUDA:0 (NVIDIA GeForce RTX 2060, 6144MiB)
# Model summary (fused): 168 layers, 3,005,843 parameters, 0 gradients, 8.1 GFLOPs
#                  Class     Images  Instances      Box(P          R
#                    all       3225      39675      0.783       0.45      0.521      0.277
# Speed: 0.2ms preprocess, 1.1ms inference, 0.0ms loss, 1.1ms postprocess per image
# Results saved to YOLOv8_WIDERFace\train_results