import tensorflow as tf
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Input, Reshape

def simple_cnn(input_shape=(416, 416, 3), grid_size=13, num_classes=20):
    inputs = Input(shape=input_shape)

    # Convolution and MaxPooling layers
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
    x = MaxPooling2D((2, 2))(x)
    x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D((2, 2))(x)
    x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D((2, 2))(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D((2, 2))(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D((2, 2))(x)
    x = Conv2D(1024, (3, 3), activation='relu', padding='same')(x)
    
    # 마지막 Conv 레이어에서 13x13 크기를 만들도록 설정
    x = Conv2D(5 + num_classes, (1, 1), activation='linear', padding='same')(x)  # 5 + num_classes: YOLO 형식
    
    # 출력 형태를 (13, 13, 5 + num_classes)로 조정
    outputs = Reshape((grid_size, grid_size, 5 + num_classes))(x)
    
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    return model
