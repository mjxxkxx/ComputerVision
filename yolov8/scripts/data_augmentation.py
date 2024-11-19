import os
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2

# 데이터 증강 설정
augmentation = A.Compose([
    A.RandomBrightnessContrast(p=0.5),
    A.HorizontalFlip(p=0.5),
    A.Resize(640, 640, p=1.0),  # 이미지 크기 변경
    A.RandomScale(scale_limit=0.2, p=0.5),
    A.MotionBlur(p=0.2),
    ToTensorV2()
])

input_dir = "D:/yolov8/images/train"  # 원본 이미지 경로
output_dir = "D:/yolov8/images/train_augmented"  # 증강된 이미지 저장 경로

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# 데이터 증강 적용
for img_name in os.listdir(input_dir):
    img_path = os.path.join(input_dir, img_name)
    img = cv2.imread(img_path)
    augmented = augmentation(image=img)
    augmented_img = augmented['image']

    # 증강 이미지 저장
    cv2.imwrite(os.path.join(output_dir, img_name), augmented_img.numpy().transpose(1, 2, 0))

print("데이터 증강 완룟!")
