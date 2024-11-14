import os

annotations_path = "C:/Users/Administrator/Documents/GitHub/ComputerVision/yolo_prj/_annotations.txt"
output_dir = "C:/Users/Administrator/Documents/GitHub/ComputerVision/yolo_prj/labels"
os.makedirs(output_dir, exist_ok=True)

IMAGE_WIDTH = 416
IMAGE_HEIGHT = 416

def convert_bbox_to_yolo(x_min, y_min, x_max, y_max, img_width, img_height):
    x_center = (x_min + x_max) / 2.0 / img_width
    y_center = (y_min + y_max) / 2.0 / img_height
    width = (x_max - x_min) / img_width
    height = (y_max - y_min) / img_height
    return x_center, y_center, width, height

with open(annotations_path, "r") as file:
    annotations = file.readlines()

for line in annotations:
    parts = line.strip().split()
    image_name = parts[0]
    objects = parts[1:]

    yolo_labels = []
    for obj in objects:
        x_min, y_min, x_max, y_max, class_id = map(int, obj.split(','))
        x_center, y_center, width, height = convert_bbox_to_yolo(x_min, y_min, x_max, y_max, IMAGE_WIDTH, IMAGE_HEIGHT)
        yolo_labels.append(f"{class_id} {x_center} {y_center} {width} {height}")

    txt_filename = os.path.splitext(image_name)[0] + ".txt"
    txt_path = os.path.join(output_dir, txt_filename)
    with open(txt_path, "w") as f:
        f.write("\n".join(yolo_labels))

print("YOLO 형식의 레이블 파일 생성이 완료되었습니다.")
