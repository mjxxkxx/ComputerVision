import os
from sklearn.metrics import precision_score, recall_score, f1_score
import glob

# Ground Truth 및 Predict 경로
gt_dir = "D:/yolov8/labels/val"
pred_dirs = glob.glob("D:/yolov8/runs/detect_*/predict_conf*")

# 라벨 로드 함수
def load_labels(label_dir):
    labels = {}
    for label_file in os.listdir(label_dir):
        if label_file.endswith(".txt"):
            with open(os.path.join(label_dir, label_file), 'r') as f:
                labels[label_file] = [line.strip() for line in f.readlines()]
    return labels

# 메트릭 계산 함수
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

# Predict 디렉터리별 평가
for pred_dir in pred_dirs:
    print(f"\nEvaluating for directory: {pred_dir}")
    gt_labels = load_labels(gt_dir)
    pred_labels = load_labels(os.path.join(pred_dir, "labels"))

    precision, recall, f1 = calculate_metrics(gt_labels, pred_labels)
    print(f"Results for {pred_dir}:")
    print(f"- Precision: {precision:.4f}")
    print(f"- Recall: {recall:.4f}")
    print(f"- F1 Score: {f1:.4f}")
