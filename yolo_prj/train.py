from backbone import simple_cnn
from loss import yolo_loss
from dataset_loader import create_dataset

def train_model(image_dir, label_dir):
    model = simple_cnn(grid_size=13, num_classes=20)
    model.compile(optimizer="adam", loss=yolo_loss, metrics=["accuracy"])

    dataset = create_dataset(image_dir, label_dir)
    model.fit(dataset.batch(2), epochs=10)
