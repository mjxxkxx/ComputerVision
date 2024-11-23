# from ultralytics import YOLO
# from datetime import datetime

# # 새롭게 학습된 모델 로드
# model = YOLO("D:/yolov8/runs/train_augmented/train_augmented_with_data/weights/best.pt")

# # Confidence 및 IoU Threshold 설정
# conf_values = [0.2, 0.25, 0.3, 0.35]
# iou_values = [0.4, 0.45, 0.5]

# # 결과 저장
# timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
# project_dir = f"D:/yolov8/runs/detect_{timestamp}"
# print(f"Project directory: {project_dir}")

# for conf in conf_values:
#     for iou in iou_values:
#         print(f"Evaluating for conf={conf}, iou={iou}...")
#         model.predict(
#             source="D:/yolov8/images/val",
#             conf=conf,
#             iou=iou,
#             save=True,
#             save_txt=True,
#             project=project_dir,
#             name=f"predict_conf{conf}_iou{iou}",
#             exist_ok=True,
#         )
#         print(f"Results saved for conf={conf}, iou={iou}.")
