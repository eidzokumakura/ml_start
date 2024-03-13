import torch
import cv2
import numpy as np
from ultralytics import YOLO

# Load a model
model = YOLO('best.pt')  # pretrained YOLOv8n model

image = 'image.jpg'
# Run batched inference on a list of images
results = model([image])  # return a list of Results objects

# Process results list
for result in results:
    boxes = result.boxes  # Boxes object for bounding box outputs
    print(boxes)
    masks = result.masks  # Masks object for segmentation masks outputs
    keypoints = result.keypoints  # Keypoints object for pose outputs
    probs = result.probs  # Probs object for classification outputs
    result.show()  # display to screen
    result.save(filename='result.jpg')

    x1, y1, x2, y2 = boxes

    image = cv2.rectangle(image, (x1, y1), (x2, y2), (255, 0, 0), 2)

    cv2.imshow('Image with bounding box', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()