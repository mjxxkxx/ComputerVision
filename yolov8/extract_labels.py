import os
from PIL import Image

def process_annotations(annotations_file, images_dir, labels_dir):
    os.makedirs(labels_dir, exist_ok=True)

    with open(annotations_file, "r") as f:
        lines = f.readlines()

    current_image = None
    for line in lines:
        line = line.strip()

        if ".jpg" in line:  # 이미지 파일 이름 줄
            current_image = line
            image_path = os.path.join(images_dir, current_image)

            if not os.path.exists(image_path):
                print(f"Warning: Image not found - {image_path}")
                continue

            with Image.open(image_path) as img:
                image_width, image_height = img.size

            label_file_path = os.path.join(labels_dir, current_image.replace(".jpg", ".txt"))
            os.makedirs(os.path.dirname(label_file_path), exist_ok=True)
            with open(label_file_path, "w") as f_label:
                pass

        elif current_image and line.isdigit():  # 객체 개수 줄
            continue

        elif current_image:
            values = line.split()
            if len(values) >= 4:  # 앞 4개 값만 처리
                x_min, y_min, width, height = map(int, values[:4])

                # YOLO 형식으로 변환
                x_center = (x_min + width / 2) / image_width
                y_center = (y_min + height / 2) / image_height
                norm_width = width / image_width
                norm_height = height / image_height

                with open(label_file_path, "a") as f_label:
                    f_label.write(f"0 {x_center:.6f} {y_center:.6f} {norm_width:.6f} {norm_height:.6f}\n")
            else:
                print(f"Invalid bounding box line: {line}")

# 학습 데이터 처리
train_annotations_file = r"C:/Users/Administrator/Documents/GitHub/ComputerVision/yolov8/wider_face_train_bbx_gt.txt"
train_images_dir = r"C:/Users/Administrator/Documents/GitHub/ComputerVision/yolov8/images/train"
train_labels_dir = r"C:/Users/Administrator/Documents/GitHub/ComputerVision/yolov8/labels/train"
process_annotations(train_annotations_file, train_images_dir, train_labels_dir)

# 검증 데이터 처리
val_annotations_file = r"C:/Users/Administrator/Documents/GitHub/ComputerVision/yolov8/wider_face_val_bbx_gt.txt"
val_images_dir = r"C:/Users/Administrator/Documents/GitHub/ComputerVision/yolov8/images/val"
val_labels_dir = r"C:/Users/Administrator/Documents/GitHub/ComputerVision/yolov8/labels/val"
process_annotations(val_annotations_file, val_images_dir, val_labels_dir)
