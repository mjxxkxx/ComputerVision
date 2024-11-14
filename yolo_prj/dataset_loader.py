import os
import cv2
import numpy as np
import tensorflow as tf

def load_yolo_labels(label_path):
    labels = []
    with open(label_path, "r") as file:
        for line in file:
            parts = line.strip().split()
            if len(parts) < 5:
                continue  # 형식이 맞지 않는 경우 건너뜀
            class_id = int(parts[0])
            x_center, y_center, width, height = map(float, parts[1:])
            labels.append([class_id, x_center, y_center, width, height])

    return np.array(labels, dtype=np.float32) if labels else np.zeros((0, 5), dtype=np.float32)

def load_image(image_path):
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not load image at {image_path}")
    image = cv2.resize(image, (416, 416))
    image = image / 255.0
    return image.astype(np.float32)

def create_dataset(image_dir, label_dir):
    image_paths = sorted([os.path.join(image_dir, img) for img in os.listdir(image_dir) if img.endswith(".jpg")])
    label_paths = sorted([os.path.join(label_dir, lbl) for lbl in os.listdir(label_dir) if lbl.endswith(".txt")])

    images = []
    labels = []
    grid_size = 13  # YOLO의 grid size
    num_classes = 20  # 클래스 수
    max_boxes = 10  # 최대 바운딩 박스 수

    for img_path, lbl_path in zip(image_paths, label_paths):
        image = load_image(img_path)
        label = load_yolo_labels(lbl_path)

        # YOLO 형식 레이블 초기화
        yolo_label = np.zeros((grid_size, grid_size, 5 + num_classes), dtype=np.float32)

        for box in label[:max_boxes]:
            class_id, x_center, y_center, width, height = box

            # 그리드 셀 좌표 계산
            grid_x = int(x_center * grid_size)
            grid_y = int(y_center * grid_size)

            # 그리드 범위 내에 있는지 확인
            if grid_x >= grid_size or grid_y >= grid_size:
                continue  # 범위를 벗어나는 경우 건너뜀

            # 상대 좌표 계산
            x_offset = x_center * grid_size - grid_x
            y_offset = y_center * grid_size - grid_y
            
            # 그리드 셀에 바운딩 박스 정보 할당
            yolo_label[grid_y, grid_x, 0:4] = [x_offset, y_offset, width, height]
            yolo_label[grid_y, grid_x, 4] = 1.0  # 객체가 있음을 나타내는 값
            yolo_label[grid_y, grid_x, 5 + int(class_id)] = 1.0  # 클래스 원핫 인코딩

        images.append(image)
        labels.append(yolo_label)

    images = np.array(images, dtype=np.float32)
    labels = np.array(labels, dtype=np.float32)

    print(f"Final Images shape: {images.shape}, Final Labels shape: {labels.shape}")

    return tf.data.Dataset.from_tensor_slices((images, labels))
