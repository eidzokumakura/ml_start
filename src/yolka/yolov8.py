import cv2
from ultralytics import YOLO

# Load a model
model = YOLO('C:\\Users\\anlok\Desktop\Zadanie\\runs\detect\\train\weights\\best.pt')  # load a pretrained model (recommended for training)


if __name__ == '__main__':
    # Train the model
    results = model.train(data='coco128.yaml', epochs=50, imgsz=640, batch=8)
