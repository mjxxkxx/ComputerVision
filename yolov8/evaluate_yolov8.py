import os
from sklearn.metrics import precision_score, recall_score, f1_score

def load_labels(label_dir):
    """
    Load labels from a directory into a dictionary.
    :param label_dir: Directory containing label files.
    :return: Dictionary with filenames as keys and list of label lines as values.
    """
    labels = {}
    for label_file in os.listdir(label_dir):
        if label_file.endswith(".txt"):
            with open(os.path.join(label_dir, label_file), 'r') as f:
                labels[label_file] = [line.strip() for line in f.readlines()]
    return labels

def calculate_metrics(gt_labels, pred_labels):
    """
    Calculate precision, recall, and F1 score by comparing GT and predicted labels.
    :param gt_labels: Dictionary containing Ground Truth labels.
    :param pred_labels: Dictionary containing predicted labels.
    :return: Precision, Recall, F1 score.
    """
    # Get union of all files (GT + Predicted)
    all_files = set(gt_labels.keys()).union(set(pred_labels.keys()))
    
    # Initialize y_true and y_pred
    y_true = []
    y_pred = []
    
    for file in all_files:
        gt_lines = gt_labels.get(file, [])
        pred_lines = pred_labels.get(file, [])

        # Append 1 for each GT line to y_true
        y_true.extend([1] * len(gt_lines))
        # Append 1 for each Predicted line to y_pred
        y_pred.extend([1] * len(pred_lines))
        
        # Handle False Negatives (FN): GT exists but no corresponding prediction
        if len(pred_lines) < len(gt_lines):
            y_pred.extend([0] * (len(gt_lines) - len(pred_lines)))
        # Handle False Positives (FP): Prediction exists but no corresponding GT
        elif len(pred_lines) > len(gt_lines):
            y_true.extend([0] * (len(pred_lines) - len(gt_lines)))
    
    # Log lengths for debugging
    print(f"Length of y_true: {len(y_true)}, Length of y_pred: {len(y_pred)}")
    
    # Calculate metrics
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    
    print(f"Total True Positives (TP): {tp_count}")
    print(f"Total False Positives (FP): {fp_count}")
    print(f"Total False Negatives (FN): {fn_count}")

    return precision, recall, f1

# Main evaluation process
def main():
    # Set paths
    gt_dir = "C:/Users/Administrator/Documents/GitHub/ComputerVision/yolov8/labels/val"
    pred_dir = "C:/Users/Administrator/Documents/GitHub/ComputerVision/yolov8/runs/detect//predict_conf0.5_iou0.4/labels"

    # Load label data
    print("Loading Ground Truth labels...")
    gt_labels = load_labels(gt_dir)
    print(f"Loaded {len(gt_labels)} Ground Truth label files.")

    print("Loading Predicted labels...")
    pred_labels = load_labels(pred_dir)
    print(f"Loaded {len(pred_labels)} Predicted label files.")

    # Error handling for missing data
    if not gt_labels:
        print("Error: No Ground Truth labels found.")
        return
    if not pred_labels:
        print("Error: No Predicted labels found.")
        return

    # Evaluate metrics
    precision, recall, f1 = calculate_metrics(gt_labels, pred_labels)

    # Print results
    print("\nEvaluation Results:")
    print(f"- Precision: {precision:.4f}")
    print(f"- Recall: {recall:.4f}")
    print(f"- F1 Score: {f1:.4f}")

if __name__ == "__main__":
    main()


# Evaluation Results: (0.25)
# - Precision: 0.1102
# - Recall: 0.9553
# - F1 Score: 0.1976