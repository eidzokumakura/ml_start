import torch
from torch import nn
import torchvision.transforms as transforms
import torchvision.models as models
import cv2
import sys
import os

# Загрузка обученного классификатора
def load_classifier(classifier_name, weights_path, device):
    if classifier_name == 'resnet18':
        state_dict = torch.load(weights_path, map_location=device).state_dict()
        model = models.resnet18()
    elif classifier_name == 'resnet34':
        state_dict = torch.load(weights_path, map_location=device).state_dict()
        model = models.resnet34()
    elif classifier_name == 'resnet50':
        state_dict = torch.load(weights_path, map_location=device).state_dict()
        model = models.resnet50()
    else:
        print("Неподдерживаемый классификатор")
        sys.exit()
    model.fc = nn.Sequential(
        nn.Linear(model.fc.in_features, 500),
        nn.ReLU(),
        nn.Dropout(),
        nn.Linear(500, 2)
    )
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model

# Предобработка изображения
def preprocess_image(image_path):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    image = transform(image)
    image = image.unsqueeze(0)
    return image

# Загрузка классификатора
classifier_name = sys.argv[1]
weights_path = sys.argv[2]
device = torch.device(sys.argv[3])
model = load_classifier(classifier_name, weights_path, device)

# Обработка изображения и предсказание
if len(sys.argv) == 5:
    image_path = sys.argv[4]
    if os.path.isdir(image_path):
        for file_name in os.listdir(image_path):
            if file_name.endswith(".jpg") or file_name.endswith(".png"):
                file_path = os.path.join(image_path, file_name)
                image = preprocess_image(file_path).to(device)
                with torch.no_grad():
                    output = model(image)
                    _, predicted = torch.max(output, 1)
                class_label = "aircraft" if predicted.item() == 0 else "ship"
                print(f"Предсказанный класс для {file_name}:", class_label)
                img = cv2.imread(file_path)
                cv2.putText(img, class_label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.imshow("Фото", img)
                cv2.waitKey(0)
                cv2.destroyAllWindows()
    else:
        image = preprocess_image(image_path).to(device)
        with torch.no_grad():
            output = model(image)
            _, predicted = torch.max(output, 1)
        class_label = "aircraft" if predicted.item() == 0 else "ship"
        print("Предсказанный класс:", class_label)
        img = cv2.imread(image_path)
        cv2.putText(img, class_label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow("Фото", img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
else:
    print("Поддерживается только предсказание одного изображения или папки с изображениями")