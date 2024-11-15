#! /usr/bin/env python
# coding=utf-8

import os
import cv2
import random
import numpy as np
import tensorflow as tf
import core.utils as utils
from core.config import cfg
'''
1. 초기화 (__init__)
    - dataset_type: train 또는 test로 데이터셋을 구분.
    - annot_path: 주석 파일 경로(훈련/테스트 데이터셋).
    - input_sizes, batch_size, data_aug: 입력 크기, 배치 크기, 데이터 증강 여부를 설정.
    - strides, anchors, anchor_per_scale: YOLO 모델의 앵커 크기 및 처리에 필요한 값들.
    - annotations: 주석 파일을 로드하여 샘플 데이터를 읽고 셔플합니다.

2. 주석 파일 로딩 및 전처리 (load_annotations)
    - 주석 파일을 읽고, 비어 있지 않은 주석만 필터링하여 무작위로 섞은 후 반환.

3. 배치 데이터 생성 (__next__)
    - 배치 크기만큼 데이터를 생성하는 메서드.
    - 각 배치는 이미지와 그에 해당하는 바운딩 박스 라벨을 포함.
    - 데이터를 순차적으로 처리하며, 매 배치마다 이미지와 라벨을 반환.

4. 데이터 증강 관련 함수들
    - random_horizontal_flip: 이미지를 수평으로 뒤집고, 바운딩 박스의 좌우를 반전시킴.
    - random_crop: 이미지의 일부를 자르고, 바운딩 박스를 조정.
    - random_translate: 이미지를 수평 및 수직으로 이동시키고, 바운딩 박스 위치를 조정.

5. 주석 파싱 및 전처리 (parse_annotation)
    - 이미지 경로와 바운딩 박스 정보를 읽고, 주석에 대해 데이터 증강을 수행.
    - 이미지 크기를 train_input_size에 맞게 조정하고, 바운딩 박스를 YOLO 형식에 맞게 전처리.

6. 교차영역 비율 계산 (bbox_iou)
    - 두 바운딩 박스 간의 IOU(Intersection over Union)를 계산하여, 앵커 박스와 실제 바운딩 박스의 겹침 정도를 평가.

7. 실제 바운딩 박스 전처리 (preprocess_true_boxes)
    - 실제 바운딩 박스를 앵커 박스와 비교하여 YOLO 형식에 맞게 라벨을 생성.
    - 작은, 중간, 큰 객체에 대해 별도로 라벨을 생성하고, 각 객체에 대한 바운딩 박스를 처리.

8. __len__
    - 데이터셋의 배치 수를 반환. 데이터셋의 크기(샘플 수)와 배치 크기를 바탕으로 계산된 배치 수.
'''

class Dataset(object):
    """Dataset 클래스: 훈련 및 테스트 데이터셋을 처리하는 클래스"""
    def __init__(self, dataset_type):
        # 훈련과 테스트 데이터셋의 경로 및 설정을 선택적으로 불러옴
        self.annot_path  = cfg.TRAIN.ANNOT_PATH if dataset_type == 'train' else cfg.TEST.ANNOT_PATH
        self.input_sizes = cfg.TRAIN.INPUT_SIZE if dataset_type == 'train' else cfg.TEST.INPUT_SIZE
        self.batch_size  = cfg.TRAIN.BATCH_SIZE if dataset_type == 'train' else cfg.TEST.BATCH_SIZE
        self.data_aug    = cfg.TRAIN.DATA_AUG   if dataset_type == 'train' else cfg.TEST.DATA_AUG

        # 훈련용 설정값 및 YOLO 관련 설정들
        self.train_input_sizes = cfg.TRAIN.INPUT_SIZE
        self.strides = np.array(cfg.YOLO.STRIDES)
        self.classes = utils.read_class_names(cfg.YOLO.CLASSES)
        self.num_classes = len(self.classes)
        self.anchors = np.array(utils.get_anchors(cfg.YOLO.ANCHORS))
        self.anchor_per_scale = cfg.YOLO.ANCHOR_PER_SCALE
        self.max_bbox_per_scale = 150

        # 주석 파일 로드 및 데이터 개수 계산
        self.annotations = self.load_annotations(dataset_type)
        self.num_samples = len(self.annotations)
        self.num_batchs = int(np.ceil(self.num_samples / self.batch_size))
        self.batch_count = 0 # 현재 배치 번호

    # 주석 파일에서 이미지 경로와 바운딩 박스 정보 읽어옴. 
    # 빈주석은 제외하고 데이터를 섞어 순서를 무작위로 섞음.
    def load_annotations(self, dataset_type):
        with open(self.annot_path, 'r') as f:
            txt = f.readlines()
            annotations = [line.strip() for line in txt if len(line.strip().split()[1:]) != 0]
        np.random.shuffle(annotations) # 데이터 섞기
        return annotations

    def __iter__(self):
        return self

    def __next__(self):
        """각 배치마다 이미지를 로드하고 처리하는 함수"""

        with tf.device('/cpu:0'):
            # 훈련 시 사용할 입력 크기 및 출력 크기 설정
            self.train_input_size = cfg.TRAIN.INPUT_SIZE
            self.train_output_sizes = self.train_input_size // self.strides
            
            # 배치별 데이터 배열 초기화
            batch_image = np.zeros((self.batch_size, self.train_input_size, self.train_input_size, 3), dtype=np.float32)

            batch_label_sbbox = np.zeros((self.batch_size, self.train_output_sizes[0], self.train_output_sizes[0],
                                          self.anchor_per_scale, 5 + self.num_classes), dtype=np.float32)
            batch_label_mbbox = np.zeros((self.batch_size, self.train_output_sizes[1], self.train_output_sizes[1],
                                          self.anchor_per_scale, 5 + self.num_classes), dtype=np.float32)
            batch_label_lbbox = np.zeros((self.batch_size, self.train_output_sizes[2], self.train_output_sizes[2],
                                          self.anchor_per_scale, 5 + self.num_classes), dtype=np.float32)

            batch_sbboxes = np.zeros((self.batch_size, self.max_bbox_per_scale, 4), dtype=np.float32)
            batch_mbboxes = np.zeros((self.batch_size, self.max_bbox_per_scale, 4), dtype=np.float32)
            batch_lbboxes = np.zeros((self.batch_size, self.max_bbox_per_scale, 4), dtype=np.float32)

            num = 0
            # 배치 크기만큼 이미지와 라벨을 로드하여 배치에 채움
            if self.batch_count < self.num_batchs:
                while num < self.batch_size:
                    index = self.batch_count * self.batch_size + num
                    if index >= self.num_samples: index -= self.num_samples # 순환적으로 데이터를 가져옴
                    annotation = self.annotations[index]
                    image, bboxes = self.parse_annotation(annotation) # 주석 파일을 파싱하여 이미지와 바운딩박스 정보 추출
                    label_sbbox, label_mbbox, label_lbbox, sbboxes, mbboxes, lbboxes = self.preprocess_true_boxes(bboxes) # 실제 바운딩박스를 YOLO 형식으로 전처리

                    # 배치에 이미지와 라벨 데이터를 채움
                    batch_image[num, :, :, :] = image
                    batch_label_sbbox[num, :, :, :, :] = label_sbbox
                    batch_label_mbbox[num, :, :, :, :] = label_mbbox
                    batch_label_lbbox[num, :, :, :, :] = label_lbbox
                    batch_sbboxes[num, :, :] = sbboxes
                    batch_mbboxes[num, :, :] = mbboxes
                    batch_lbboxes[num, :, :] = lbboxes
                    num += 1
                self.batch_count += 1
                batch_smaller_target = batch_label_sbbox, batch_sbboxes
                batch_medium_target  = batch_label_mbbox, batch_mbboxes
                batch_larger_target  = batch_label_lbbox, batch_lbboxes

                return batch_image, (batch_smaller_target, batch_medium_target, batch_larger_target)
            else:
                self.batch_count = 0
                np.random.shuffle(self.annotations) # 배치 끝날 때마다 데이터 섞음
                raise StopIteration # 마지막 배치에서 반복 종료

    # 수평 뒤집기: 이미지와 바운딩박스를 랜덤으로 수평 반전
    def random_horizontal_flip(self, image, bboxes):
        if random.random() < 0.5:
            _, w, _ = image.shape
            image = image[:, ::-1, :]
            bboxes[:, [0,2]] = w - bboxes[:, [2,0]] # 바운딩 박스 좌우 반전
        return image, bboxes

    # 이미지 자르기: 랜덤으로 이미지를 잘라내고 바운딩 박스를 그에 맞게 조정
    def random_crop(self, image, bboxes):

        if random.random() < 0.5:
            h, w, _ = image.shape
            max_bbox = np.concatenate([np.min(bboxes[:, 0:2], axis=0), np.max(bboxes[:, 2:4], axis=0)], axis=-1)

            # 잘라낼 영역 계산
            max_l_trans = max_bbox[0]
            max_u_trans = max_bbox[1]
            max_r_trans = w - max_bbox[2]
            max_d_trans = h - max_bbox[3]

            crop_xmin = max(0, int(max_bbox[0] - random.uniform(0, max_l_trans)))
            crop_ymin = max(0, int(max_bbox[1] - random.uniform(0, max_u_trans)))
            crop_xmax = max(w, int(max_bbox[2] + random.uniform(0, max_r_trans)))
            crop_ymax = max(h, int(max_bbox[3] + random.uniform(0, max_d_trans)))

            image = image[crop_ymin : crop_ymax, crop_xmin : crop_xmax]

            # 바운딩 박스 좌표 조정
            bboxes[:, [0, 2]] = bboxes[:, [0, 2]] - crop_xmin
            bboxes[:, [1, 3]] = bboxes[:, [1, 3]] - crop_ymin

        return image, bboxes

    # 이미지 평행 이동: 이미지를 평행으로 이동하고 바운딩 박스를 조정
    def random_translate(self, image, bboxes):

        if random.random() < 0.5:
            h, w, _ = image.shape
            max_bbox = np.concatenate([np.min(bboxes[:, 0:2], axis=0), np.max(bboxes[:, 2:4], axis=0)], axis=-1)

            # 이동 가능한 범위 계산
            max_l_trans = max_bbox[0]
            max_u_trans = max_bbox[1]
            max_r_trans = w - max_bbox[2]
            max_d_trans = h - max_bbox[3]

            tx = random.uniform(-(max_l_trans - 1), (max_r_trans - 1))
            ty = random.uniform(-(max_u_trans - 1), (max_d_trans - 1))

            # 이미지를 이동
            M = np.array([[1, 0, tx], [0, 1, ty]])
            image = cv2.warpAffine(image, M, (w, h))

            # 바운딩 박스 이동
            bboxes[:, [0, 2]] = bboxes[:, [0, 2]] + tx
            bboxes[:, [1, 3]] = bboxes[:, [1, 3]] + ty

        return image, bboxes

    # 주석 파싱 및 전처리: 이미지 경로와 바운딩 박스를 읽고 전처리하는 함수
    def parse_annotation(self, annotation):
        line = annotation.split()
        image_path = line[0]
        if not os.path.exists(image_path):
            raise KeyError("%s does not exist ... " %image_path)
        image = cv2.imread(image_path)
        bboxes = np.array([list(map(int, box.split(','))) for box in line[1:]])

        if self.data_aug:
            image, bboxes = self.random_horizontal_flip(np.copy(image), np.copy(bboxes))
            image, bboxes = self.random_crop(np.copy(image), np.copy(bboxes))
            image, bboxes = self.random_translate(np.copy(image), np.copy(bboxes))

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image, bboxes = utils.image_preporcess(np.copy(image), [self.train_input_size, self.train_input_size], np.copy(bboxes))
        return image, bboxes

    # 교차영역 비율 계산: 두 바운딩 박스의 IOU(Intersection over Union) 계산
    def bbox_iou(self, boxes1, boxes2):
        boxes1 = np.array(boxes1)
        boxes2 = np.array(boxes2)

        boxes1_area = boxes1[..., 2] * boxes1[..., 3]
        boxes2_area = boxes2[..., 2] * boxes2[..., 3]

        boxes1 = np.concatenate([boxes1[..., :2] - boxes1[..., 2:] * 0.5,
                                boxes1[..., :2] + boxes1[..., 2:] * 0.5], axis=-1)
        boxes2 = np.concatenate([boxes2[..., :2] - boxes2[..., 2:] * 0.5,
                                boxes2[..., :2] + boxes2[..., 2:] * 0.5], axis=-1)

        left_up = np.maximum(boxes1[..., :2], boxes2[..., :2])
        right_down = np.minimum(boxes1[..., 2:], boxes2[..., 2:])

        inter_section = np.maximum(right_down - left_up, 0.0)
        inter_area = inter_section[..., 0] * inter_section[..., 1]
        union_area = boxes1_area + boxes2_area - inter_area

        return inter_area / union_area

    # 실제 바운딩 박스 전처리: 앵커 박스와 실제 바운딩 박스를 비교하여 YOLO 형식에 맞게 전처리
    def preprocess_true_boxes(self, bboxes):

        label = [np.zeros((self.train_output_sizes[i], self.train_output_sizes[i], self.anchor_per_scale,
                           5 + self.num_classes)) for i in range(3)]
        bboxes_xywh = [np.zeros((self.max_bbox_per_scale, 4)) for _ in range(3)]
        bbox_count = np.zeros((3,))

        for bbox in bboxes:
            bbox_coor = bbox[:4]
            bbox_class_ind = bbox[4]

            onehot = np.zeros(self.num_classes, dtype=np.float)
            onehot[bbox_class_ind] = 1.0
            uniform_distribution = np.full(self.num_classes, 1.0 / self.num_classes)
            deta = 0.01
            smooth_onehot = onehot * (1 - deta) + deta * uniform_distribution

            bbox_xywh = np.concatenate([(bbox_coor[2:] + bbox_coor[:2]) * 0.5, bbox_coor[2:] - bbox_coor[:2]], axis=-1)
            bbox_xywh_scaled = 1.0 * bbox_xywh[np.newaxis, :] / self.strides[:, np.newaxis]

            iou = []
            exist_positive = False
            for i in range(3):
                anchors_xywh = np.zeros((self.anchor_per_scale, 4))
                anchors_xywh[:, 0:2] = np.floor(bbox_xywh_scaled[i, 0:2]).astype(np.int32) + 0.5
                anchors_xywh[:, 2:4] = self.anchors[i]

                iou_scale = self.bbox_iou(bbox_xywh_scaled[i][np.newaxis, :], anchors_xywh)
                iou.append(iou_scale)
                iou_mask = iou_scale > 0.3 # IOU가 0.3 이상인 앵커는 긍정적인 예시로 간주

                if np.any(iou_mask):
                    xind, yind = np.floor(bbox_xywh_scaled[i, 0:2]).astype(np.int32)

                    label[i][yind, xind, iou_mask, :] = 0
                    label[i][yind, xind, iou_mask, 0:4] = bbox_xywh
                    label[i][yind, xind, iou_mask, 4:5] = 1.0
                    label[i][yind, xind, iou_mask, 5:] = smooth_onehot

                    bbox_ind = int(bbox_count[i] % self.max_bbox_per_scale)
                    bboxes_xywh[i][bbox_ind, :4] = bbox_xywh
                    bbox_count[i] += 1

                    exist_positive = True

            # 긍정적인 예시가 없는 경우 가장 잘 맞는 앵커로 설정
            if not exist_positive:
                best_anchor_ind = np.argmax(np.array(iou).reshape(-1), axis=-1)
                best_detect = int(best_anchor_ind / self.anchor_per_scale)
                best_anchor = int(best_anchor_ind % self.anchor_per_scale)
                xind, yind = np.floor(bbox_xywh_scaled[best_detect, 0:2]).astype(np.int32)

                label[best_detect][yind, xind, best_anchor, :] = 0
                label[best_detect][yind, xind, best_anchor, 0:4] = bbox_xywh
                label[best_detect][yind, xind, best_anchor, 4:5] = 1.0
                label[best_detect][yind, xind, best_anchor, 5:] = smooth_onehot

                bbox_ind = int(bbox_count[best_detect] % self.max_bbox_per_scale)
                bboxes_xywh[best_detect][bbox_ind, :4] = bbox_xywh
                bbox_count[best_detect] += 1

        # 작은, 중간, 큰 객체에 대한 라벨과 바운딩 박스 반환
        label_sbbox, label_mbbox, label_lbbox = label
        sbboxes, mbboxes, lbboxes = bboxes_xywh
        return label_sbbox, label_mbbox, label_lbbox, sbboxes, mbboxes, lbboxes

    def __len__(self):
        return self.num_batchs