import os
import cv2
import numpy as np
from backbone import simple_cnn

def load_image_for_prediction(image_path):
    image = cv2.imread(image_path)
    original_image = image.copy()
    image = cv2.resize(image, (416, 416))
    image = image / 255.0
    return image.astype(np.float32), original_image

def draw_bounding_boxes(original_image, predictions, confidence_threshold=0.5):
    height, width, _ = original_image.shape
    for pred in predictions:
        class_id, x_center, y_center, box_width, box_height, confidence = pred
        if confidence < confidence_threshold:
            continue

        x_min = int((x_center - box_width / 2) * width)
        y_min = int((y_center - box_height / 2) * height)
        x_max = int((x_center + box_width / 2) * width)
        y_max = int((y_center + box_height / 2) * height)

        cv2.rectangle(original_image, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
        label = f"Class {int(class_id)}: {confidence:.2f}"
        cv2.putText(original_image, label, (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    return original_image

def predict_and_save(image_path, model_weights="yolo_prj/saved_models/model_checkpoint.h5", output_dir="output"):
    model = simple_cnn()
    model.load_weights(model_weights)

    image, original_image = load_image_for_prediction(image_path)
    image = np.expand_dims(image, axis=0).astype(np.float32)

    predictions = model.predict(image)

    predicted_boxes = []
    for pred in predictions[0]:
        class_id = np.argmax(pred[:20])
        confidence = pred[20]
        x_center, y_center, width, height = pred[21:25]
        predicted_boxes.append([class_id, x_center, y_center, width, height, confidence])

    output_image = draw_bounding_boxes(original_image, predicted_boxes)
    os.makedirs(output_dir, exist_ok=True)

    output_filename = os.path.join(output_dir, f"output_{os.path.basename(image_path)}")
    cv2.imwrite(output_filename, output_image)
    print(f"Output image saved at {output_filename}")
