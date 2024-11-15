from ultralytics import YOLO
import multiprocessing as mp

if __name__ == '__main__':
    # Pre-trained YOLOv8 모델 로드
    model = YOLO("yolov8n.pt")  # 'yolov8n.pt'는 Nano 모델입니다. 필요한 경우 다른 모델로 변경 가능.

    # 학습 실행
    model.train(
        data="C:/Users/Administrator/Documents/GitHub/ComputerVision/yolov8/data.yaml",  # data.yaml 경로
        epochs=50,        # 학습 반복 횟수
        imgsz=416,         # 입력 이미지 크기
        batch=8,          # 배치 크기
        workers=2,         # 데이터 로더의 워커 수
        project="YOLOv8_WIDERFace",  # 프로젝트 이름
        name="train_results",        # 결과 저장 폴더 이름
        exist_ok=True                # 이미 폴더가 존재하면 덮어쓰기
    )
