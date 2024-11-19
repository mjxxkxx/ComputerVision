import os
from sklearn.metrics import precision_score, recall_score, f1_score

# Ground Truth 경로 설정
gt_dir = "D:/yolov8/labels/val"

# Predict 경로 리스트
pred_dirs = [
    "D:/yolov8/runs/detect/predict_conf0.2_iou0.4",
    "D:/yolov8/runs/detect/predict_conf0.2_iou0.45",
    "D:/yolov8/runs/detect/predict_conf0.25_iou0.4",
    "D:/yolov8/runs/detect/predict_conf0.25_iou0.45",
    "D:/yolov8/runs/detect/predict_conf0.3_iou0.4",
    "D:/yolov8/runs/detect/predict_conf0.3_iou0.45",
    "D:/yolov8/runs/detect/predict_conf0.35_iou0.4",
    "D:/yolov8/runs/detect/predict_conf0.35_iou0.45",
]

# 함수 정의
def load_labels(label_dir):
    labels = {}
    for label_file in os.listdir(label_dir):
        if label_file.endswith(".txt"):
            with open(os.path.join(label_dir, label_file), 'r') as f:
                labels[label_file] = [line.strip() for line in f.readlines()]
    return labels

def calculate_metrics(gt_labels, pred_labels):
    all_files = set(gt_labels.keys()).union(set(pred_labels.keys()))
    y_true, y_pred = [], []

    for file in all_files:
        gt_lines = gt_labels.get(file, [])
        pred_lines = pred_labels.get(file, [])

        y_true.extend([1] * len(gt_lines))
        y_pred.extend([1] * len(pred_lines))

        if len(pred_lines) < len(gt_lines):
            y_pred.extend([0] * (len(gt_lines) - len(pred_lines)))
        elif len(pred_lines) > len(gt_lines):
            y_true.extend([0] * (len(pred_lines) - len(gt_lines)))

    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    return precision, recall, f1

# Evaluate 여러 Predict 디렉터리
for pred_dir in pred_dirs:
    print(f"\nEvaluating for directory: {pred_dir}")
    
    # GT 및 Pred 라벨 로드
    gt_labels = load_labels(gt_dir)
    pred_labels = load_labels(os.path.join(pred_dir, "labels"))
    
    # 메트릭 계산
    precision, recall, f1 = calculate_metrics(gt_labels, pred_labels)
    
    # 결과 출력
    print(f"Results for {pred_dir}:")
    print(f"- Precision: {precision:.4f}")
    print(f"- Recall: {recall:.4f}")
    print(f"- F1 Score: {f1:.4f}")
