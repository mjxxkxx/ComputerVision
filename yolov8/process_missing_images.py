import os
from ultralytics import YOLO

def process_missing_images(model_path, images_dir, labels_dir, missing_files):
    # 모델 로드
    model = YOLO(model_path)

    # 누락된 이미지 처리
    for missing_file in missing_files:
        image_name = missing_file.replace(".txt", ".jpg")
        image_path = os.path.join(images_dir, image_name)

        # 이미지 존재 여부 확인
        if not os.path.exists(image_path):
            print(f"Image not found: {image_path}")
            continue

        print(f"Processing missing image: {image_name}")
        results = model(image_path)

        # 결과 저장
        label_file_path = os.path.join(labels_dir, missing_file)
        os.makedirs(os.path.dirname(label_file_path), exist_ok=True)

        # YOLO 포맷 저장
        with open(label_file_path, "w") as f:
            for box in results[0].boxes.xywhn.tolist():
                f.write(f"0 {box[0]:.6f} {box[1]:.6f} {box[2]:.6f} {box[3]:.6f}\n")

# 경로 설정
model_path = "D:/yolov8/YOLOv8_WIDERFace/train_results/weights/best.pt"
images_dir = "D:/yolov8/images/val"
labels_dir = "D:/yolov8/runs/detect/predict_adjusted/labels"

# Missing 라벨 파일 목록
missing_files = [
    "40_Gymnastics_Gymnastics_40_364.txt",
    "26_Soldier_Drilling_Soldiers_Drilling_26_710.txt",
    "24_Soldier_Firing_Soldier_Firing_24_95.txt",
    "47_Matador_Bullfighter_matadorbullfighting_47_354.txt",
    "14_Traffic_Traffic_14_361.txt",
    "54_Rescue_rescuepeople_54_840.txt",
    "41_Swimming_Swimming_41_535.txt",
    "35_Basketball_playingbasketball_35_366.txt",
    "40_Gymnastics_Gymnastics_40_361.txt",
    "44_Aerobics_Aerobics_44_707.txt",
    "27_Spa_Spa_27_225.txt",
    "26_Soldier_Drilling_Soldiers_Drilling_26_390.txt",
    "41_Swimming_Swimmer_41_56.txt",
    "40_Gymnastics_Gymnastics_40_24.txt",
    "24_Soldier_Firing_Soldier_Firing_24_281.txt",
    "18_Concerts_Concerts_18_657.txt",
    "41_Swimming_Swimming_41_52.txt",
    "47_Matador_Bullfighter_Matador_Bullfighter_47_171.txt",
    "39_Ice_Skating_Ice_Skating_39_458.txt",
    "41_Swimming_Swimming_41_730.txt",
    "55_Sports_Coach_Trainer_sportcoaching_55_414.txt",
    "27_Spa_Spa_27_109.txt",
    "15_Stock_Market_Stock_Market_15_483.txt",
    "39_Ice_Skating_iceskiing_39_869.txt",
    "52_Photographers_photographertakingphoto_52_653.txt",
    "14_Traffic_Traffic_14_55.txt",
    "44_Aerobics_Aerobics_44_578.txt",
    "35_Basketball_playingbasketball_35_283.txt",
    "45_Balloonist_Balloonist_45_685.txt",
    "57_Angler_peoplefishing_57_515.txt",
    "41_Swimming_Swimmer_41_68.txt",
    "25_Soldier_Patrol_Soldier_Patrol_25_614.txt",
    "39_Ice_Skating_iceskiing_39_1000.txt",
    "40_Gymnastics_Gymnastics_40_749.txt",
    "43_Row_Boat_Rowboat_43_301.txt",
    "2_Demonstration_Protesters_2_589.txt",
    "40_Gymnastics_Gymnastics_40_285.txt",
    "27_Spa_Spa_27_212.txt",
    "46_Jockey_Jockey_46_923.txt",
    "37_Soccer_soccer_ball_37_341.txt",
    "2_Demonstration_Demonstrators_2_306.txt",
    "52_Photographers_photographertakingphoto_52_84.txt",
    "14_Traffic_Traffic_14_722.txt",
    "0_Parade_Parade_0_246.txt",
    "40_Gymnastics_Gymnastics_40_638.txt",
    "27_Spa_Spa_27_420.txt",
    "41_Swimming_Swimming_41_26.txt",
    "41_Swimming_Swimming_41_412.txt",
    "2_Demonstration_Political_Rally_2_807.txt",
    "39_Ice_Skating_iceskiing_39_354.txt",
    "41_Swimming_Swimming_41_379.txt",
    "46_Jockey_Jockey_46_728.txt",
    "41_Swimming_Swimmer_41_148.txt",
    "26_Soldier_Drilling_Soldiers_Drilling_26_893.txt",
    "5_Car_Accident_Accident_5_576.txt",
    "0_Parade_marchingband_1_356.txt",
    "20_Family_Group_Family_Group_20_90.txt",
    "28_Sports_Fan_Sports_Fan_28_697.txt",
    "5_Car_Accident_Accident_5_243.txt",
    "55_Sports_Coach_Trainer_sportcoaching_55_640.txt",
    "41_Swimming_Swimmer_41_308.txt",
    "37_Soccer_soccer_ball_37_926.txt",
    "31_Waiter_Waitress_Waiter_Waitress_31_212.txt",
    "45_Balloonist_Balloonist_45_769.txt",
    "43_Row_Boat_Canoe_43_538.txt",
    "10_People_Marching_People_Marching_2_577.txt",
    "18_Concerts_Concerts_18_104.txt",
    "13_Interview_Interview_Sequences_13_15.txt",
    "47_Matador_Bullfighter_Matador_Bullfighter_47_636.txt",
    "26_Soldier_Drilling_Soldiers_Drilling_26_1022.txt",
    "26_Soldier_Drilling_Soldiers_Drilling_26_991.txt",
    "27_Spa_Spa_27_728.txt",
    "43_Row_Boat_Rowboat_43_688.txt",
    "8_Election_Campain_Election_Campaign_8_357.txt",
    "5_Car_Accident_Accident_5_641.txt",
    "52_Photographers_photographertakingphoto_52_755.txt",
    "26_Soldier_Drilling_Soldiers_Drilling_26_359.txt"
]

# 누락된 이미지 처리
process_missing_images(model_path, images_dir, labels_dir, missing_files)
