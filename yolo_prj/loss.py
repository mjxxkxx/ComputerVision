import tensorflow as tf

def yolo_loss(y_true, y_pred):
    # 위치 손실 (x, y 좌표)
    position_loss = tf.reduce_sum(tf.square(y_true[..., :2] - y_pred[..., :2]))

    # 크기 손실 (width, height)
    size_loss = tf.reduce_sum(tf.square(tf.sqrt(y_true[..., 2:4]) - tf.sqrt(y_pred[..., 2:4])))

    # 객체 존재 여부 손실
    object_loss = tf.reduce_sum(tf.square(y_true[..., 4] - y_pred[..., 4]))

    # 클래스 손실
    class_loss = tf.reduce_sum(
        tf.keras.losses.categorical_crossentropy(y_true[..., 5:], y_pred[..., 5:])
    )

    # 총 손실 계산
    total_loss = position_loss + size_loss + object_loss + class_loss
    return total_loss
