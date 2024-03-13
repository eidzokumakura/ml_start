from ultralytics import YOLO

yaml_file = 'dataset/dataset.yaml'

# Load a model
model = YOLO('yolov8n.pt')  # load a pretrained model (recommended for training)


if __name__ == '__main__':
    # Train the model
    results = model.train(data=yaml_file, epochs=100, imgsz=640, batch=8)