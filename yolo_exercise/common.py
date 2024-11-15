#! /usr/bin/env python
# coding=utf-8

import tensorflow as tf
# import tensorflow_addons as tfa

'''
1. BatchNormalization.call: 커스텀 배치 정규화 레이어 호출, 학습/추론 모드 설정.
2. convolutional: YOLOv4에서 사용되는 합성곱 레이어 정의.
3. softplus: 입력 크기에 따라 다르게 동작하는 softplus 함수. (안정적인 계산)
4. mish: mish 활성화 함수. (성능 개선)
5. residual_block: 잔차 블록을 추가하여 정보 손실 방지.
6. upsample: 이미지 업샘플링. 해상도 2배.
'''

# 커스텀 BatchNormalization 레이어 정의
class BatchNormalization(tf.keras.layers.BatchNormalization):
    """
    BatchNormalization을 확장하여 'Frozen state'와 'inference mode'에서 다르게 동작하도록 설정.
    `layer.trainable = False`는 해당 레이어를 '동결'시켜 학습 시 'gama'와 'beta' 값을 업데이트하지 않도록 함.
    """
    def call(self, x, training=False):
        if not training:
            training = tf.constant(False) # 추론 모드로 설정
        training = tf.logical_and(training, self.trainable) # 훈련 상태와 동결 여부를 결합
        return super().call(x, training)

# YOLOv4에서 사용하는 합성곱 레이어 정의
def convolutional(input_layer, filters_shape, downsample=False, activate=True, bn=True, activate_type='leaky'):
    '''
    합성곱 레이어를 정의. 입력, 출력 채널, 필터 크기, 다운샘플링 여부, 활성화 함수 설정 등을 처리.

    input_layer: 입력 데이터.
    filters_shape: 필터의 크기 및 입력과 출력 채널 수.
    downsample: 다운샘플링 여부.
    activate: 활성화 함수를 적용할지 여부.
    bn: 배치 정규화 여부.
    activate_type: 활성화 함수의 유형.
    '''
    # 다운샘플링을 위해 strides=2와 ZeroPadding을 적용하여 크기를 절반으로 줄임
    if downsample:
        input_layer = tf.keras.layers.ZeroPadding2D(((1, 0), (1, 0)))(input_layer)
        padding = 'valid' # 유효한 패딩 (zero padding)
        strides = 2 # 스트라이드 2로 다운샘플링
    else: 
        strides = 1
        padding = 'same' # 출력 크기와 입력 크기가 동일하도록 패딩

    # 합성곱 연산. 배치 정규화가 있을 경우 bias는 생략하고, kernel_regularizer로 L2 정규화 적용.
    conv = tf.keras.layers.Conv2D(filters=filters_shape[-1], kernel_size = filters_shape[0], strides=strides, padding=padding,
                                  use_bias=not bn, kernel_regularizer=tf.keras.regularizers.l2(0.0005),
                                  kernel_initializer=tf.random_normal_initializer(stddev=0.01),
                                  bias_initializer=tf.constant_initializer(0.))(input_layer)
    
    # 배치 정규화가 활성화 되어 있으면 커스텀 BatchNormalization 레이어 적용
    if bn: conv = BatchNormalization()(conv)

    # 활성화 함수 적용
    if activate == True:
        # 활성화 함수 종류: leaky_relu 또는 mish
        if activate_type == "leaky":
            conv = tf.nn.leaky_relu(conv, alpha=0.1)
        elif activate_type == "mish":
            conv = mish(conv)
            # conv = softplus(conv)
            # conv = conv * tf.math.tanh(tf.math.softplus(conv))
            # conv = conv * tf.tanh(softplus(conv))
            # conv = tf.nn.leaky_relu(conv, alpha=0.1)
            # conv = tfa.activations.mish(conv)
            # conv = conv * tf.nn.tanh(tf.keras.activations.relu(tf.nn.softplus(conv), max_value=20))
            # conv = tf.nn.softplus(conv)
            # conv = tf.keras.activations.relu(tf.nn.softplus(conv), max_value=20)

    return conv

# softplus 활성화 함수의 변형 정의
def softplus(x, threshold = 20.):
    '''
    Softplus 활성화 함수의 안정적인 구현. 
    특정 threshold를 넘어가는 값에 대해 계산을 안정적으로 처리.
    '''
    def f1():
        return x # x가 크면 그대로 반환
    def f2():
        return tf.exp(x) # x가 매우 작으면 exp(x) 계산
    def f3():
        return tf.math.log(1 + tf.exp(x)) # 일반적인 softplus 계산
    # mask = tf.greater(x, threshold)
    # x = tf.exp(x[mask])
    # return tf.exp(x)

    # x 값에 따라 적절한 연산을 수행
    return tf.case([(tf.greater(x, tf.constant(threshold)), lambda:f1()), (tf.less(x, tf.constant(-threshold)), lambda:f2())], default=lambda:f3()) # threshold를 넘지 않으면 기본 softplus 계산
    # return tf.case([(tf.greater(x, threshold), lambda:f1())])

# Mish 함수 : 딥러닝 모델의 성능을 개선하는 활성화 함수
def mish(x):
    '''
    Mish 활성화 함수 : `x * tanh(softplus(x))` 형태로 구현된 활성화 함수로, 학습 성능을 개선하는 효과가 있음.
    '''
    return tf.keras.layers.Lambda(lambda x: x*tf.tanh(tf.math.log(1+tf.exp(x))))(x)
    # return tf.keras.layers.Lambda(lambda x: softplus(x))(x)
    # return tf.keras.layers.Lambda(lambda x: x * tf.tanh(softplus(x)))(x)

# residual_block 함수 : 잔차 블록 정의 (Residual Connection)
def residual_block(input_layer, input_channel, filter_num1, filter_num2, activate_type='leaky'):
    '''
    잔차 블록을 정의하여 깊은 네트워크에서도 정보가 손실되지 않도록 함.
    
    input_layer: 입력 데이터 (Tensor).
    input_channel: 입력 채널 수.
    filter_num1: 첫 번째 합성곱 레이어의 출력 채널 수.
    filter_num2: 두 번째 합성곱 레이어의 출력 채널 수.
    activate_type: 활성화 함수 종류 ('leaky' 또는 'mish').
    '''
    short_cut = input_layer # 입력을 그대로 저장하여 나중에 더할 수 있게 함

    # 1x1 합성곱 (채널 수를 줄이기 위한 연산)
    conv = convolutional(input_layer, filters_shape=(1, 1, input_channel, filter_num1), activate_type=activate_type)

    # 3x3 합성곱 (특징을 추출)
    conv = convolutional(conv, filters_shape=(3, 3, filter_num1,   filter_num2), activate_type=activate_type)

    # 잔차 연결 (short_cut + conv)
    residual_output = short_cut + conv
    return residual_output

# upsample 함수 : 해상도 2배 늘림
def upsample(input_layer):
    '''
    이미지의 해상도를 2배로 늘려줌 (nearest neighbor 방식).
    
    input_layer: 입력 데이터 (Tensor).
    '''
    return tf.image.resize(input_layer, (input_layer.shape[1] * 2, input_layer.shape[2] * 2), method='nearest')

