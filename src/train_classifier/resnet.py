# необходимые импорты
import torch
import torchvision.datasets as dataset
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models

# создаем объект tranfrorm для трансформации изображений
transform = transforms.Compose([
      transforms.Resize(256),
      transforms.CenterCrop(224),
      transforms.ToTensor(),
      transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

train_path = "../../aircrafts_and_ships/train"
test_path = "../../aircrafts_and_ships/test"

train_data = dataset.ImageFolder(train_path, transform)
test_data = dataset.ImageFolder(test_path, transform)

print(type(train_data))
print(type(test_data))

print(train_data.classes)
print(test_data.classes)

train_loader = DataLoader(train_data, batch_size=16, shuffle=True)
test_loader  = DataLoader(train_data, batch_size=16, shuffle=True)

# построение сверточной нейронной сети на PyTorch

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        # стек сверточных слоев
        self.conv_layers = nn.Sequential(
            # здесь определяются сверточные слои
            # можно явно вычислить размер выходной карты признаков каждого
            # сверточного слоя по следующей формуле:
            # [(shape + 2*padding - kernel_size) / stride] + 1
            nn.Conv2d(in_channels=1, out_channels=12, kernel_size=3, padding=1, stride=1), # (N, 1, 28, 28)
            nn.ReLU(),
            # после первого сверточного слоя размер выходной карты признаков равен:
            # [(28 + 2*1 - 3)/1] + 1 = 28.
            nn.MaxPool2d(kernel_size=2),
            # при прохождении слоя MaxPooling с размером окна 2
            # карты признаков сжимаются вдвое
            # 28 / 2 = 14
            nn.Conv2d(in_channels=12, out_channels=24, kernel_size=3, padding=1, stride=1),
            nn.ReLU(),
            # после второго сверточного слоя размер выходной карты признаков равен:
            #[(14 + 2*1 - 3)/1] + 1 = 14
            nn.MaxPool2d(kernel_size=2),
            # после второго слоя MaxPooling2D выходнае карты признаков имеют размерность
            # 14 / 2 = 7
        )
        # стек полносвязных слоев
        self.linear_layers = nn.Sequential(
            # после второго сверточного слоя имеем количество выходных карт признаков
            # равное 24 размером 7х7
            # эти данные и будут входными признаками в первом линейном слое
            nn.Linear(in_features=24*7*7, out_features=64),
            nn.ReLU(),
            nn.Dropout(0.2), # обнуляем 20% входного тензора для предотвращения переобучения
            nn.Linear(in_features=64, out_features=10) # количество выходных признаков равно количеству классов

        )

    # определение метода для прчмого распространения сигналов по сети
    def forward(self, x):
        x = self.conv_layers(x)
        # перед отправкой в блок полносвязных слоев признаки необходимо сделать одномерными
        x = x.view(x.size(0), -1)
        x = self.linear_layers(x)
        return x

# вывод структуры модели
cnn = CNN()
print(cnn)

# определим функцию, которая будет вычислять точность модели на итерации
def calculate_accuracy(y_pred, y):

    # находим количество верных совпадений лейбла и выходного класса по каждому примеру в батче
    top_pred = y_pred.argmax(1, keepdim = True)
    correct = top_pred.eq(y.view_as(top_pred)).sum()

    # посчитаем точность, которая равна отношению количества верных совпадений к общему числу примеров в батче
    acc = correct.float() / y.shape[0]
    return acc

# функция, отвечающая за обучение сети на одной эпохе
def train(model, dataloader, optimizer, loss_function, device):
    # определим значения точности и потерь на старте эпохи
    epoch_acc = 0
    epoch_loss = 0

    # переведем модель в режим тренировки
    model.train()

    # для каждого батча в даталоадере
    for (images, labels) in dataloader:

        # отправляем изображения и метки на устройство
        images = images.to(device)
        labels = labels.to(device)

        # обнуляем градиенты
        optimizer.zero_grad()

        # вычислим выходы сети на данном батче
        predicts = model(images)

        # вычислим величину потерь на данном батче
        loss    = loss_function(predicts, labels)

        # вычислим точность на данном батче
        acc     = calculate_accuracy(predicts, labels)

        # вычислим значения градиентов на батче
        loss.backward()

        # корректируем веса
        optimizer.step()

        # прибавим значения потерь и точности на батче
        epoch_loss += loss.item()
        epoch_acc  += acc.item()

    # возвращаем величину потерь и точность на эпохе
    return epoch_loss / len(dataloader),  epoch_acc / len(dataloader)

# функция, отвечающая за проверку модели на одной эпохе
def evaluate(model, dataloader, loss_function, device):

    # определим начальные величины потерь и точности
    epoch_acc = 0
    epoch_loss = 0

    # переведем модель в режим валидации
    model.eval()

    # указываем, что градиенты вычислять не нужно
    with torch.no_grad():

        # для каждого батча в даталоадере
        for images, labels in dataloader:

            # переносим изображения и лейблы на устройство
            images = images.to(device)
            labels = labels.to(device)

            # вычислим выходы сети на батче
            predicts = model(images)

            # вычислим величину потерь на батче
            loss = loss_function(predicts, labels)

            # вычислим точность на батче
            acc  = calculate_accuracy(predicts, labels)


            # прибавим значения потерь и точности на батче к общему
            epoch_loss += loss.item()
            epoch_acc  += acc.item()


    # возвращаем величину потерь и точность на эпохе
    return epoch_loss / len(dataloader),  epoch_acc / len(dataloader)

def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs

# определим функцию оптимизации
optimizer = optim.Adam(cnn.parameters(), lr=0.001)

# определим функцию потерь
loss_function = nn.CrossEntropyLoss()

# определим устройство, на котором будет идти обучение
device = None
if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

print(device)

# перемещение модели на устройство
cnn.to(device)

# инициализируем предобученную модель ResNet50
pretrained_resnet50 = models.resnet50(pretrained=True)

# замораживаем слои, используя метод requires_grad()
# в этом случае не вычисляются градиенты для слоев
# сделать это надо для всех параметеров сети
for name, param in pretrained_resnet50.named_parameters():
    param.requires_grad = False


# к различным блокам модели в PyTorch легко получить доступ
# заменим блок классификатора на свой, подходящий для решения
# задачи классификации кошек и собак
pretrained_resnet50.fc = nn.Sequential(
    nn.Linear(pretrained_resnet50.fc.in_features, 500),
    nn.ReLU(),
    nn.Dropout(),
    nn.Linear(500, 2)
)

# перенесем модель на устройство
pretrained_resnet50.to(device)

# выведем модель
print(pretrained_resnet50)

# попробуем обучить!

epochs = 5
optimizer = optim.Adam(pretrained_resnet50.parameters(), lr=0.001)
loss_function = nn.CrossEntropyLoss()

best_loss = 1000000
best_acc = 0
for epoch in range(epochs):

    train_loss, train_acc = train(pretrained_resnet50, train_loader, optimizer, loss_function, device)

    test_loss, test_acc = evaluate(pretrained_resnet50, test_loader, loss_function, device)

    print(f'Epoch: {epoch+1:02}')
    print(f'\tTrain Loss: {train_loss:.3f} | Train Acc: {train_acc*100:.2f}%')
    print(f'\t Val. Loss: {test_loss:.3f} |  Val. Acc: {test_acc*100:.2f}%')

    if test_loss < best_loss:
        torch.save(pretrained_resnet50, "../run_classifier/resnet50_best_loss.pth")
