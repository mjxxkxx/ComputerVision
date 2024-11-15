#! /usr/bin/env python
# coding=utf-8

import numpy as np
import tensorflow as tf
import core.utils as utils
import core.common as common
import core.backbone as backbone
from core.config import cfg

# NUM_CLASS       = len(utils.read_class_names(cfg.YOLO.CLASSES))
# STRIDES         = np.array(cfg.YOLO.STRIDES)
# IOU_LOSS_THRESH = cfg.YOLO.IOU_LOSS_THRESH
# XYSCALE = cfg.YOLO.XYSCALE
# ANCHORS = utils.get_anchors(cfg.YOLO.ANCHORS)
'''
1. YOLOv3, YOLOv4, YOLOv3_tiny: 각각 YOLOv3, YOLOv4, Tiny YOLOv3 모델의 구조를 정의.
2. decode, decode_train: 모델의 예측 결과를 디코딩하여 바운딩 박스와 신뢰도, 확률을 구함.
3. bbox_iou, bbox_ciou, bbox_giou: 바운딩 박스 간 유사도를 계산하여 IOU, CIoU, GIoU를 구함.
4. compute_loss: 모델 학습에 필요한 손실(GIoU, 신뢰도 손실, 클래스 확률 손실)을 계산.
'''

# ------------------------------------------------------
# 1. YOLO 모델 구조
# ------------------------------------------------------
# YOLOv3 모델 정의
def YOLOv3(input_layer, NUM_CLASS):
    """
    YOLOv3 모델의 전체 네트워크 구조를 정의합니다.
    - input_layer: 입력 이미지 (텐서).
    - NUM_CLASS: 클래스의 수.
    """
    # 1. Darknet53 백본 네트워크에서 3개의 피처맵 출력
    # route_1: 작은 객체를 위한 피처맵
    # route_2: 중간 크기 객체를 위한 피처맵
    # conv: 큰 객체를 위한 피처맵
    route_1, route_2, conv = backbone.darknet53(input_layer)

    # 2. 큰 객체(Large Object)를 예측하기 위한 출력 레이어 구성
    conv = common.convolutional(conv, (1, 1, 1024, 512))  # 1x1 컨볼루션
    conv = common.convolutional(conv, (3, 3, 512, 1024))  # 3x3 컨볼루션
    conv = common.convolutional(conv, (1, 1, 1024, 512))  # 반복 수행
    conv = common.convolutional(conv, (3, 3, 512, 1024))
    conv = common.convolutional(conv, (1, 1, 1024, 512))
    
    # 3. 마지막 예측 분기 생성 (Large Object Prediction)
    conv_lobj_branch = common.convolutional(conv, (3, 3, 512, 1024))  # 3x3 컨볼루션
    conv_lbbox = common.convolutional(conv_lobj_branch, (1, 1, 1024, 3 * (NUM_CLASS + 5)), activate=False, bn=False)

    # 4. 중간 크기 객체(Medium Object)를 예측하기 위한 출력 레이어 구성
    conv = common.convolutional(conv, (1, 1, 512, 256))  # 1x1 컨볼루션 (다운샘플링)
    conv = common.upsample(conv)  # 업샘플링 수행
    conv = tf.concat([conv, route_2], axis=-1)  # 백본의 중간 피처맵과 결합

    conv = common.convolutional(conv, (1, 1, 768, 256))  # 추가 컨볼루션 레이어
    conv = common.convolutional(conv, (3, 3, 256, 512))
    conv = common.convolutional(conv, (1, 1, 512, 256))
    conv = common.convolutional(conv, (3, 3, 256, 512))
    conv = common.convolutional(conv, (1, 1, 512, 256))

    # 5. 중간 크기 객체 예측 분기 생성 (Medium Object Prediction)
    conv_mobj_branch = common.convolutional(conv, (3, 3, 256, 512))
    conv_mbbox = common.convolutional(conv_mobj_branch, (1, 1, 512, 3 * (NUM_CLASS + 5)), activate=False, bn=False)

    # 6. 작은 객체(Small Object)를 예측하기 위한 출력 레이어 구성
    conv = common.convolutional(conv, (1, 1, 256, 128))  # 다운샘플링
    conv = common.upsample(conv)  # 업샘플링 수행
    conv = tf.concat([conv, route_1], axis=-1)  # 백본의 작은 객체 피처맵과 결합

    conv = common.convolutional(conv, (1, 1, 384, 128))  # 추가 컨볼루션 레이어
    conv = common.convolutional(conv, (3, 3, 128, 256))
    conv = common.convolutional(conv, (1, 1, 256, 128))
    conv = common.convolutional(conv, (3, 3, 128, 256))
    conv = common.convolutional(conv, (1, 1, 256, 128))

    # 7. 작은 객체 예측 분기 생성 (Small Object Prediction)
    conv_sobj_branch = common.convolutional(conv, (3, 3, 128, 256))
    conv_sbbox = common.convolutional(conv_sobj_branch, (1, 1, 256, 3 * (NUM_CLASS + 5)), activate=False, bn=False)

    # 8. 최종 출력: 작은 객체, 중간 객체, 큰 객체 예측 결과를 반환
    return [conv_sbbox, conv_mbbox, conv_lbbox]

def YOLOv4(input_layer, NUM_CLASS):
    """
    YOLOv4 모델의 전체 네트워크 구조를 정의합니다.
    - input_layer: 입력 이미지 (텐서).
    - NUM_CLASS: 클래스의 수.
    """
    # 1. CSPDarknet53 백본 네트워크에서 피처맵 출력
    route_1, route_2, conv = backbone.cspdarknet53(input_layer)

    # 2. 큰 객체(Large Object) 예측 준비
    route = conv
    conv = common.convolutional(conv, (1, 1, 512, 256))  # 1x1 컨볼루션
    conv = common.upsample(conv)  # 업샘플링 수행
    route_2 = common.convolutional(route_2, (1, 1, 512, 256))  # 1x1 컨볼루션
    conv = tf.concat([route_2, conv], axis=-1)  # 결합

    # 3. 중간 크기 객체(Medium Object) 예측 준비
    conv = common.convolutional(conv, (1, 1, 512, 256))  # 컨볼루션 레이어 추가
    conv = common.convolutional(conv, (3, 3, 256, 512))
    conv = common.convolutional(conv, (1, 1, 512, 256))
    conv = common.convolutional(conv, (3, 3, 256, 512))
    conv = common.convolutional(conv, (1, 1, 512, 256))

    # 4. 작은 객체(Small Object) 예측 준비
    route_2 = conv
    conv = common.convolutional(conv, (1, 1, 256, 128))  # 다운샘플링
    conv = common.upsample(conv)  # 업샘플링
    route_1 = common.convolutional(route_1, (1, 1, 256, 128))
    conv = tf.concat([route_1, conv], axis=-1)  # 결합

    # 5. 작은 객체(Small Object) 예측
    conv = common.convolutional(conv, (1, 1, 256, 128))  # 컨볼루션 레이어
    conv = common.convolutional(conv, (3, 3, 128, 256))
    conv = common.convolutional(conv, (1, 1, 256, 128))
    conv = common.convolutional(conv, (3, 3, 128, 256))
    conv = common.convolutional(conv, (1, 1, 256, 128))
    route_1 = conv
    conv = common.convolutional(conv, (3, 3, 128, 256))  # 최종 분기
    conv_sbbox = common.convolutional(conv, (1, 1, 256, 3 * (NUM_CLASS + 5)), activate=False, bn=False)

    # 6. 중간 크기 객체(Medium Object) 예측
    conv = common.convolutional(route_1, (3, 3, 128, 256), downsample=True)  # 다운샘플링
    conv = tf.concat([conv, route_2], axis=-1)
    conv = common.convolutional(conv, (1, 1, 512, 256))
    conv = common.convolutional(conv, (3, 3, 256, 512))
    conv = common.convolutional(conv, (1, 1, 512, 256))
    conv = common.convolutional(conv, (3, 3, 256, 512))
    conv = common.convolutional(conv, (1, 1, 512, 256))
    route_2 = conv
    conv = common.convolutional(conv, (3, 3, 256, 512))  # 최종 분기
    conv_mbbox = common.convolutional(conv, (1, 1, 512, 3 * (NUM_CLASS + 5)), activate=False, bn=False)

    # 7. 큰 객체(Large Object) 예측
    conv = common.convolutional(route_2, (3, 3, 256, 512), downsample=True)  # 다운샘플링
    conv = tf.concat([conv, route], axis=-1)
    conv = common.convolutional(conv, (1, 1, 1024, 512))
    conv = common.convolutional(conv, (3, 3, 512, 1024))
    conv = common.convolutional(conv, (1, 1, 1024, 512))
    conv = common.convolutional(conv, (3, 3, 512, 1024))
    conv = common.convolutional(conv, (1, 1, 1024, 512))
    conv = common.convolutional(conv, (3, 3, 512, 1024))  # 최종 분기
    conv_lbbox = common.convolutional(conv, (1, 1, 1024, 3 * (NUM_CLASS + 5)), activate=False, bn=False)

    # 8. 결과 반환: 작은 객체, 중간 객체, 큰 객체 예측
    return [conv_sbbox, conv_mbbox, conv_lbbox]

# ------------------------------------------------------
# 2. 디코딩 함수
# ------------------------------------------------------

def decode(conv_output, NUM_CLASS, i=0):
    """
    YOLOv3/YOLOv4의 예측 결과를 디코딩하여 (x, y, w, h, 신뢰도, 클래스 확률) 반환.
    """
    # 1. 출력 형태 변환
    conv_shape = tf.shape(conv_output)
    batch_size = conv_shape[0]
    output_size = conv_shape[1]
    conv_output = tf.reshape(conv_output, (batch_size, output_size, output_size, 3, 5 + NUM_CLASS))

    # 2. 좌표 및 확률 분리
    conv_raw_xywh, conv_raw_conf, conv_raw_prob = tf.split(conv_output, (4, 1, NUM_CLASS), axis=-1)

    # 3. sigmoid 활성화 함수 적용
    pred_conf = tf.sigmoid(conv_raw_conf)
    pred_prob = tf.sigmoid(conv_raw_prob)

    # 4. 디코딩된 결과 반환
    return tf.concat([conv_raw_xywh, pred_conf, pred_prob], axis=-1)

def decode_train(conv_output, NUM_CLASS, STRIDES, ANCHORS, i=0, XYSCALE=[1, 1, 1]):
    """
    YOLO 모델의 학습용 디코딩 함수.
    - conv_output: 모델의 출력.
    - STRIDES, ANCHORS: 네트워크 특성에 맞는 설정 값.
    """
    # 1. 출력 형태 변환
    conv_shape = tf.shape(conv_output)
    batch_size = conv_shape[0]
    output_size = conv_shape[1]
    conv_output = tf.reshape(conv_output, (batch_size, output_size, output_size, 3, 5 + NUM_CLASS))

    # 2. 중심 좌표와 크기 계산
    conv_raw_dxdy, conv_raw_dwdh, conv_raw_conf, conv_raw_prob = tf.split(conv_output, (2, 2, 1, NUM_CLASS), axis=-1)

    # 3. 그리드 좌표 계산
    x = tf.tile(tf.expand_dims(tf.range(output_size, dtype=tf.int32), axis=0), [output_size, 1])
    y = tf.tile(tf.expand_dims(tf.range(output_size, dtype=tf.int32), axis=1), [1, output_size])
    xy_grid = tf.expand_dims(tf.stack([x, y], axis=-1), axis=2)
    xy_grid = tf.tile(tf.expand_dims(xy_grid, axis=0), [batch_size, 1, 1, 3, 1])
    xy_grid = tf.cast(xy_grid, tf.float32)

    # 4. 실제 좌표 변환 및 디코딩
    pred_xy = ((tf.sigmoid(conv_raw_dxdy) * XYSCALE[i]) - 0.5 * (XYSCALE[i] - 1) + xy_grid) * STRIDES[i]
    pred_wh = tf.exp(conv_raw_dwdh) * ANCHORS[i]
    pred_xywh = tf.concat([pred_xy, pred_wh], axis=-1)

    # 5. 확률 디코딩
    pred_conf = tf.sigmoid(conv_raw_conf)
    pred_prob = tf.sigmoid(conv_raw_prob)

    # 6. 최종 디코딩 결과 반환
    return tf.concat([pred_xywh, pred_conf, pred_prob], axis=-1)

# ------------------------------------------------------
# 3. 바운딩 박스 유사도 계산 함수들
# ------------------------------------------------------
def bbox_iou(boxes1, boxes2):
    """
    두 바운딩 박스 간의 IOU(Intersection Over Union) 계산.
    - boxes1, boxes2: 바운딩 박스 좌표 (x, y, w, h).
    """
    # 1. 각 바운딩 박스의 영역 계산
    boxes1_area = boxes1[..., 2] * boxes1[..., 3]
    boxes2_area = boxes2[..., 2] * boxes2[..., 3]

    # 2. 바운딩 박스 좌표를 [xmin, ymin, xmax, ymax] 형식으로 변환
    boxes1_coor = tf.concat([boxes1[..., :2] - boxes1[..., 2:] * 0.5,
                             boxes1[..., :2] + boxes1[..., 2:] * 0.5], axis=-1)
    boxes2_coor = tf.concat([boxes2[..., :2] - boxes2[..., 2:] * 0.5,
                             boxes2[..., :2] + boxes2[..., 2:] * 0.5], axis=-1)

    # 3. 교차 영역 계산
    left_up = tf.maximum(boxes1_coor[..., :2], boxes2_coor[..., :2])
    right_down = tf.minimum(boxes1_coor[..., 2:], boxes2_coor[..., 2:])
    inter_section = tf.maximum(right_down - left_up, 0.0)
    inter_area = inter_section[..., 0] * inter_section[..., 1]

    # 4. IOU 계산
    union_area = boxes1_area + boxes2_area - inter_area
    return 1.0 * inter_area / union_area

def bbox_ciou(boxes1, boxes2):
    """
    두 바운딩 박스 간의 CIoU(Complete Intersection Over Union) 계산.
    """
    # 1. [xmin, ymin, xmax, ymax] 형식으로 변환
    boxes1_coor = tf.concat([boxes1[..., :2] - boxes1[..., 2:] * 0.5,
                             boxes1[..., :2] + boxes1[..., 2:] * 0.5], axis=-1)
    boxes2_coor = tf.concat([boxes2[..., :2] - boxes2[..., 2:] * 0.5,
                             boxes2[..., :2] + boxes2[..., 2:] * 0.5], axis=-1)

    # 2. 둘러싸는 상자의 좌표 계산
    left = tf.minimum(boxes1_coor[..., 0], boxes2_coor[..., 0])
    up = tf.minimum(boxes1_coor[..., 1], boxes2_coor[..., 1])
    right = tf.maximum(boxes1_coor[..., 2], boxes2_coor[..., 2])
    down = tf.maximum(boxes1_coor[..., 3], boxes2_coor[..., 3])

    # 3. 둘러싸는 상자의 대각선 거리 계산
    c = (right - left) ** 2 + (up - down) ** 2

    # 4. IOU 계산
    iou = bbox_iou(boxes1, boxes2)

    # 5. 중심 좌표 거리 계산
    u = tf.reduce_sum((boxes1[..., :2] - boxes2[..., :2]) ** 2, axis=-1)

    # 6. 종합적인 CIoU 계산
    d = u / c
    ar_gt = boxes2[..., 2] / boxes2[..., 3]
    ar_pred = boxes1[..., 2] / boxes1[..., 3]
    ar_loss = 4 / (np.pi ** 2) * (tf.atan(ar_gt) - tf.atan(ar_pred)) ** 2
    alpha = ar_loss / (1 - iou + ar_loss + 1e-6)
    ciou = iou - d - alpha * ar_loss

    return ciou

def bbox_giou(boxes1, boxes2):
    """
    두 바운딩 박스 간의 GIoU(Generalized Intersection Over Union) 계산.
    """
    # 1. [xmin, ymin, xmax, ymax] 형식으로 변환
    boxes1_coor = tf.concat([boxes1[..., :2] - boxes1[..., 2:] * 0.5,
                             boxes1[..., :2] + boxes1[..., 2:] * 0.5], axis=-1)
    boxes2_coor = tf.concat([boxes2[..., :2] - boxes2[..., 2:] * 0.5,
                             boxes2[..., :2] + boxes2[..., 2:] * 0.5], axis=-1)

    # 2. 교차 영역 계산
    left_up = tf.maximum(boxes1_coor[..., :2], boxes2_coor[..., :2])
    right_down = tf.minimum(boxes1_coor[..., 2:], boxes2_coor[..., 2:])
    inter_section = tf.maximum(right_down - left_up, 0.0)
    inter_area = inter_section[..., 0] * inter_section[..., 1]

    # 3. IOU 계산
    boxes1_area = (boxes1_coor[..., 2] - boxes1_coor[..., 0]) * (boxes1_coor[..., 3] - boxes1_coor[..., 1])
    boxes2_area = (boxes2_coor[..., 2] - boxes2_coor[..., 0]) * (boxes2_coor[..., 3] - boxes2_coor[..., 1])
    union_area = boxes1_area + boxes2_area - inter_area
    iou = inter_area / union_area

    # 4. 둘러싸는 상자의 좌표 계산
    enclose_left_up = tf.minimum(boxes1_coor[..., :2], boxes2_coor[..., :2])
    enclose_right_down = tf.maximum(boxes1_coor[..., 2:], boxes2_coor[..., 2:])
    enclose = tf.maximum(enclose_right_down - enclose_left_up, 0.0)
    enclose_area = enclose[..., 0] * enclose[..., 1]

    # 5. GIoU 계산
    giou = iou - (enclose_area - union_area) / enclose_area
    return giou

# ------------------------------------------------------
# 4. 손실 함수
# ------------------------------------------------------
def compute_loss(pred, conv, label, bboxes, STRIDES, NUM_CLASS, IOU_LOSS_THRESH, i=0):
    """
    YOLO 모델의 손실 함수 계산.
    - GIoU 손실: 바운딩 박스의 위치/크기 불일치 반영.
    - 신뢰도 손실: 객체 존재 확률의 불일치 반영.
    - 클래스 확률 손실: 예측 확률과 실제 클래스 간의 불일치 반영.
    """
    # 1. 입력 데이터 크기 및 형태 변환
    conv_shape = tf.shape(conv)
    batch_size = conv_shape[0]
    output_size = conv_shape[1]
    input_size = STRIDES[i] * output_size
    conv = tf.reshape(conv, (batch_size, output_size, output_size, 3, 5 + NUM_CLASS))

    # 2. 예측값과 레이블 분리
    conv_raw_conf = conv[..., 4:5]  # 신뢰도
    conv_raw_prob = conv[..., 5:]  # 클래스 확률
    pred_xywh = pred[..., 0:4]  # 바운딩 박스
    pred_conf = pred[..., 4:5]

    # 3. 실제값과 레이블 분리
    label_xywh = label[..., 0:4]
    respond_bbox = label[..., 4:5]
    label_prob = label[..., 5:]

    # 4. GIoU 손실 계산
    giou = tf.expand_dims(bbox_giou(pred_xywh, label_xywh), axis=-1)
    bbox_loss_scale = 2.0 - 1.0 * label_xywh[..., 2:3] * label_xywh[..., 3:4] / (input_size ** 2)
    giou_loss = respond_bbox * bbox_loss_scale * (1 - giou)

    # 5. 신뢰도 손실 계산
    iou = bbox_iou(pred_xywh[..., np.newaxis, :], bboxes[np.newaxis, np.newaxis, np.newaxis, :, :])
    max_iou = tf.reduce_max(iou, axis=-1, keepdims=True)
    respond_bgd = (1.0 - respond_bbox) * tf.cast(max_iou < IOU_LOSS_THRESH, tf.float32)
    conf_focal = tf.pow(respond_bbox - pred_conf, 2)
    conf_loss = conf_focal * (
        respond_bbox * tf.nn.sigmoid_cross_entropy_with_logits(labels=respond_bbox, logits=conv_raw_conf) +
        respond_bgd * tf.nn.sigmoid_cross_entropy_with_logits(labels=respond_bbox, logits=conv_raw_conf)
    )

    # 6. 클래스 확률 손실 계산
    prob_loss = respond_bbox * tf.nn.sigmoid_cross_entropy_with_logits(labels=label_prob, logits=conv_raw_prob)

    # 7. 손실값 평균 계산 및 반환
    giou_loss = tf.reduce_mean(tf.reduce_sum(giou_loss, axis=[1, 2, 3, 4]))
    conf_loss = tf.reduce_mean(tf.reduce_sum(conf_loss, axis=[1, 2, 3, 4]))
    prob_loss = tf.reduce_mean(tf.reduce_sum(prob_loss, axis=[1, 2, 3, 4]))

    return giou_loss, conf_loss, prob_loss
