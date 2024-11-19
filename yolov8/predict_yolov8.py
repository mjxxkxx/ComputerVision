from ultralytics import YOLO
import torch

# 랜덤성 제어
torch.manual_seed(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# 모델 로드
model = YOLO("D:/yolov8/YOLOv8_WIDERFace/train_results/weights/best.pt")

# 테스트할 Confidence Threshold와 IoU Threshold 설정
conf_values = [0.2] #0.25, 0.3, 0.35]  # 세분화된 Confidence Threshold
iou_values = [0.4, 0.45]         # 적정 IoU Threshold

# 결과 저장 디렉터리
output_dir = "runs/detect"

for conf in conf_values:
    for iou in iou_values:
        print(f"Evaluating for conf={conf}, iou={iou}")
        results = model.predict(
            source="D:/yolov8/images/val",
            conf=conf,
            iou=iou,
            save=True,
            save_txt=True,
            project=output_dir,
            name=f"predict_conf{conf}_iou{iou}",
            exist_ok=True
        )
