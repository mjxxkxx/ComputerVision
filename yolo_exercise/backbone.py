#! /usr/bin/env python  << python 인터프리터
# coding=utf-8

import tensorflow as tf
import core.common as common

# darknet53 함수 : 기본 Darknet53 네트워크 구조
def darknet53(input_data):
    '''
    1. 합성곱 레이어와 다운샘플링:
        - 각 합성곱 레이어에서는 3x3 커널을 사용해 특징을 추출하고, 점진적으로 채널을 확장하며 입력 데이터를 처리합니다.
        - downsample=True는 이미지 크기를 절반으로 줄이고, 채널 수를 늘려서 더 높은 수준의 특징을 추출합니다.
    2. Residual Block:
        - 각 단계에서 residual_block을 적용하여 입력 데이터를 처리합니다. 이 블럭은 skip connection을 이용하여 입력과 출력을 더해주는 방식으로, 학습을 안정화시키고 기울기 소실 문제를 해결합니다.
        - 반복 횟수는 각 단계에서 다르며, 첫 번째 단계에서는 1회, 두 번째 단계에서는 2회, 세 번째 단계에서는 8회, 마지막 단계에서는 4회 적용됩니다.
    3. 중간 출력 저장:
        - route_1과 route_2는 중간 출력을 저장하여 YOLO 헤드 또는 다른 네트워크에서 사용할 수 있도록 합니다.
        - route_1은 네트워크의 첫 번째 중간 출력으로, route_2는 두 번째 중간 출력으로 활용됩니다.
    4. 최종 출력:
        - 마지막으로, 네트워크의 최종 출력을 반환합니다. 이 출력은 route_1, route_2와 함께 반환되어 후속 네트워크에서 활용될 수 있습니다.
    '''
    # 1. 첫 번째 합성곱 레이어: 3x3 커널, 입력 채널 3, 출력 채널 32
    input_data = common.convolutional(input_data, (3, 3,  3,  32))

    # 2. 다운샘플링: 입력 채널 32 -> 출력 채널 64
    input_data = common.convolutional(input_data, (3, 3, 32,  64), downsample=True)

    # 3. Residual Block 1번 적용: 입력 채널 64 -> 출력 채널 64
    for i in range(1):
        input_data = common.residual_block(input_data,  64,  32, 64)

    # 4. 다운샘플링: 입력 채널 64 -> 출력 채널 128
    input_data = common.convolutional(input_data, (3, 3,  64, 128), downsample=True)

    # 5. Residual Block 2번 적용: 입력 채널 128 -> 출력 채널 128
    for i in range(2):
        input_data = common.residual_block(input_data, 128,  64, 128)

    # 6. 다운샘플링: 입력 채널 128 -> 출력 채널 256
    input_data = common.convolutional(input_data, (3, 3, 128, 256), downsample=True)

    # 7. Residual Block 8번 적용: 입력 채널 256 -> 출력 채널 256
    for i in range(8):
        input_data = common.residual_block(input_data, 256, 128, 256)

    # 8. 중간 출력 저장 (YOLO 헤드 연결용)
    route_1 = input_data

    # 9. 다운샘플링: 입력 채널 256 -> 출력 채널 512
    input_data = common.convolutional(input_data, (3, 3, 256, 512), downsample=True)

    # 10. Residual Block 8번 적용: 입력 채널 512 -> 출력 채널 512
    for i in range(8):
        input_data = common.residual_block(input_data, 512, 256, 512)

    # 11. 두 번째 중간 출력 저장
    route_2 = input_data

    # 12. 다운샘플링: 입력 채널 512 -> 출력 채널 1024
    input_data = common.convolutional(input_data, (3, 3, 512, 1024), downsample=True)

    # 13. Residual Block 4번 적용: 입력 채널 1024 -> 출력 채널 1024
    for i in range(4):
        input_data = common.residual_block(input_data, 1024, 512, 1024)

    return route_1, route_2, input_data # 중간 출력과 최종 출력을 반환

# cspdarknet53 함수 : Cross Stage Partial (CSP) 네트워크 구조 적용
def cspdarknet53(input_data):
    '''
    1. 합성곱 레이어와 다운샘플링:
        - 각 단계에서 합성곱 레이어를 사용해 특징을 추출하고, 다운샘플링을 통해 점진적으로 채널 수를 늘려가며 특징 맵을 압축합니다.
        - mish 활성화 함수를 사용하여 비선형성을 추가합니다.
    2. CSP 구조 (Cross Stage Partial):
        - 입력 데이터를 route와 input_data로 분리하고, 1x1 합성곱을 통해 특징을 강조한 후 결합하여 더 많은 정보를 처리합니다.
        - 각 단계에서 route를 따라가며 특징을 점차적으로 결합하고, 다운샘플링하여 네트워크를 심화시킵니다.
    3. Residual Block:
        - 각 단계에서 residual_block을 통해 입력 데이터와 출력 데이터를 더하여 학습 효과를 높입니다.
    4. 최종 출력:
        - MaxPooling을 여러 크기로 적용해 다양한 스케일의 특징을 추출하고, 이를 결합하여 최종 출력을 생성합니다.
    5. 출력:
        - 중간 결과인 route_1과 route_2와 최종 출력을 반환하여 후속 네트워크에서 활용할 수 있습니다.
    '''
    # 1. 첫 번째 합성곱 레이어: 3x3 커널, 입력 채널 3, 출력 채널 32, 활성화 함수 "mish"
    input_data = common.convolutional(input_data, (3, 3,  3,  32), activate_type="mish")

    # 2. 다운샘플링: 입력 채널 32 -> 출력 채널 64, 활성화 함수 "mish"
    input_data = common.convolutional(input_data, (3, 3, 32,  64), downsample=True, activate_type="mish")

    # 3. CSP 구조 (첫 번째 단계): 입력 데이터를 분리하고 결합
    route = input_data
    route = common.convolutional(route, (1, 1, 64, 64), activate_type="mish")
    input_data = common.convolutional(input_data, (1, 1, 64, 64), activate_type="mish")

    # 4. Residual Block 1번 적용
    for i in range(1):
        input_data = common.residual_block(input_data,  64,  32, 64, activate_type="mish")
    input_data = common.convolutional(input_data, (1, 1, 64, 64), activate_type="mish")

    # 5. 특징 결합 (Concatenation): route와 input_data를 결합
    input_data = tf.concat([input_data, route], axis=-1)

    # 6. 1x1 합성곱 후 3x3 다운샘플링: 채널 128로 변환
    input_data = common.convolutional(input_data, (1, 1, 128, 64), activate_type="mish")
    input_data = common.convolutional(input_data, (3, 3, 64, 128), downsample=True, activate_type="mish")

    # 7. CSP 구조 (두 번째 단계): 특징을 분리하고 결합
    route = input_data
    route = common.convolutional(route, (1, 1, 128, 64), activate_type="mish")
    input_data = common.convolutional(input_data, (1, 1, 128, 64), activate_type="mish")
    
    # 8. Residual Block 2번 적용
    for i in range(2):
        input_data = common.residual_block(input_data, 64,  64, 64, activate_type="mish")
    input_data = common.convolutional(input_data, (1, 1, 64, 64), activate_type="mish")
    
    # 9. 특징 결합 (Concatenation): route와 input_data를 결합
    input_data = tf.concat([input_data, route], axis=-1)

    # 10. 1x1 합성곱 후 3x3 다운샘플링: 채널 256로 변환
    input_data = common.convolutional(input_data, (1, 1, 128, 128), activate_type="mish")
    input_data = common.convolutional(input_data, (3, 3, 128, 256), downsample=True, activate_type="mish")
    
    # 11. CSP 구조 (세 번째 단계): 특징을 분리하고 결합
    route = input_data
    route = common.convolutional(route, (1, 1, 256, 128), activate_type="mish")
    input_data = common.convolutional(input_data, (1, 1, 256, 128), activate_type="mish")

    # 12. Residual Block 8번 적용
    for i in range(8):
        input_data = common.residual_block(input_data, 128, 128, 128, activate_type="mish")
    input_data = common.convolutional(input_data, (1, 1, 128, 128), activate_type="mish")
    
    # 13. 특징 결합 (Concatenation): route와 input_data를 결합
    input_data = tf.concat([input_data, route], axis=-1)

    # 14. 1x1 합성곱 후 3x3 다운샘플링: 채널 512로 변환
    input_data = common.convolutional(input_data, (1, 1, 256, 256), activate_type="mish")
    route_1 = input_data
    input_data = common.convolutional(input_data, (3, 3, 256, 512), downsample=True, activate_type="mish")
    
    # 15. CSP 구조 (네 번째 단계): 특징을 분리하고 결합
    route = input_data
    route = common.convolutional(route, (1, 1, 512, 256), activate_type="mish")
    input_data = common.convolutional(input_data, (1, 1, 512, 256), activate_type="mish")

    # 16. Residual Block 8번 적용
    for i in range(8):
        input_data = common.residual_block(input_data, 256, 256, 256, activate_type="mish")
    input_data = common.convolutional(input_data, (1, 1, 256, 256), activate_type="mish")
    
    # 17. 특징 결합 (Concatenation): route와 input_data를 결합
    input_data = tf.concat([input_data, route], axis=-1)
   
    # 18. 1x1 합성곱 후 3x3 다운샘플링: 채널 1024로 변환
    input_data = common.convolutional(input_data, (1, 1, 512, 512), activate_type="mish")
    route_2 = input_data
    input_data = common.convolutional(input_data, (3, 3, 512, 1024), downsample=True, activate_type="mish")
    
    # 19. CSP 구조 (다섯 번째 단계): 특징을 분리하고 결합
    route = input_data
    route = common.convolutional(route, (1, 1, 1024, 512), activate_type="mish")
    input_data = common.convolutional(input_data, (1, 1, 1024, 512), activate_type="mish")

    # 20. Residual Block 4번 적용
    for i in range(4):
        input_data = common.residual_block(input_data, 512, 512, 512, activate_type="mish")
    input_data = common.convolutional(input_data, (1, 1, 512, 512), activate_type="mish")
    
    # 21. 특징 결합 (Concatenation): route와 input_data를 결합
    input_data = tf.concat([input_data, route], axis=-1)

    # 22. 최종 합성곱: 최종 출력물 생성
    input_data = common.convolutional(input_data, (1, 1, 1024, 1024), activate_type="mish")
    input_data = common.convolutional(input_data, (1, 1, 1024, 512))
    input_data = common.convolutional(input_data, (3, 3, 512, 1024))
    input_data = common.convolutional(input_data, (1, 1, 1024, 512))

    # 23. 다양한 크기의 MaxPooling을 결합하여 다양한 스케일의 특징 추출
    input_data = tf.concat([tf.nn.max_pool(input_data, ksize=13, padding='SAME', strides=1), tf.nn.max_pool(input_data, ksize=9, padding='SAME', strides=1)
                            , tf.nn.max_pool(input_data, ksize=5, padding='SAME', strides=1), input_data], axis=-1)
    
    # 24. 최종 1x1 합성곱 후 3x3 합성곱
    input_data = common.convolutional(input_data, (1, 1, 2048, 512))
    input_data = common.convolutional(input_data, (3, 3, 512, 1024))

    # 25. 최종 출력물 생성
    input_data = common.convolutional(input_data, (1, 1, 1024, 512))

    return route_1, route_2, input_data

# darknet53_tiny 함수 : 경량화된 모델, 빠르지만 성능이 낮음
def darknet53_tiny(input_data):
    # 작은 크기의 합성곱과 MaxPooling 레이어로 빠르게 연산
    input_data = common.convolutional(input_data, (3, 3, 3, 16))
    input_data = tf.keras.layers.MaxPool2D(2, 2, 'same')(input_data)
    input_data = common.convolutional(input_data, (3, 3, 16, 32))
    input_data = tf.keras.layers.MaxPool2D(2, 2, 'same')(input_data)
    input_data = common.convolutional(input_data, (3, 3, 32, 64))
    input_data = tf.keras.layers.MaxPool2D(2, 2, 'same')(input_data)
    input_data = common.convolutional(input_data, (3, 3, 64, 128))
    input_data = tf.keras.layers.MaxPool2D(2, 2, 'same')(input_data)
    input_data = common.convolutional(input_data, (3, 3, 128, 256))
    
    # 중간 출력 (route_1)
    route_1 = input_data
    input_data = tf.keras.layers.MaxPool2D(2, 2, 'same')(input_data)

    # 다운샘플링 후 채널 512로 변환
    input_data = common.convolutional(input_data, (3, 3, 256, 512))
    input_data = tf.keras.layers.MaxPool2D(2, 1, 'same')(input_data)

    # 최종 출력
    input_data = common.convolutional(input_data, (3, 3, 512, 1024))

    return route_1, input_data # 중간 출력과 최종 출력을 반환