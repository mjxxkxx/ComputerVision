import cv2
import os
from ultralytics import YOLO
from datetime import datetime

# 학습된 YOLO 모델 로드
model = YOLO("C:\\Users\\Administrator\\Desktop\\test\\best.pt")  # 학습된 모델 경로

# 저장 디렉토리 설정
output_dir = "C:\\Users\\Administrator\\Desktop\\test\\results"  # 탐지된 객체 저장 경로
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# 웹캠 열기
cap = cv2.VideoCapture(0)  # 0번 카메라 열기

if not cap.isOpened():
    print("웹캠을 열 수 없습니다.")
    exit()

frame_count = 0  # 저장할 프레임 카운터

while True:
    # 프레임 읽기
    ret, frame = cap.read()
    if not ret:
        print("프레임을 읽을 수 없습니다. 종료합니다.")
        break

    # YOLO 모델로 예측
    results = model.predict(source=frame, conf=0.25, device=0, show=False)

    # 탐지 결과 그리기
    annotated_frame = results[0].plot()

    # 전체 프레임 저장
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    frame_filename = f"{output_dir}/frame_{frame_count}_{timestamp}.jpg"
    cv2.imwrite(frame_filename, annotated_frame)  # 바운딩 박스가 그려진 전체 프레임 저장
    print(f"Saved annotated frame: {frame_filename}")

    frame_count += 1

    # 결과 프레임 출력
    cv2.imshow("YOLO Real-Time Detection", annotated_frame)

    # ESC 키로 종료
    if cv2.waitKey(1) & 0xFF == 27:  # 1ms 대기, ESC 키로 종료
        break

# 리소스 해제
cap.release()
cv2.destroyAllWindows()
