import torch
from torch import nn
import torch.onnx
import onnx
import onnxruntime
import numpy as np
import torchvision.models as models

device = None
if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

state_dict = torch.load("../run_classifier/resnet18_best_loss.pth", map_location=device).state_dict()

model = models.resnet50()

model.fc = nn.Sequential(
    nn.Linear(model.fc.in_features, 500),
    nn.ReLU(),
    nn.Dropout(),
    nn.Linear(500, 2)
)

model.load_state_dict(state_dict)
model.to(device)
model.eval()

# создаем входной тензор модели
x = torch.randn(1, 3, 224, 224, requires_grad=True).to(device)

# сгенерируем выходы модели
out = model(x)

# экспортируем модель в onnx
torch.onnx.export(model, x, "resnet18_best_loss.onnx")

onnx_model = onnx.load("resnet18_best_loss.onnx")
onnx.checker.check_model(onnx_model)

# убедимся, что выходы PTH-модели и ONNX-модели совпадают


# инициализируем сессию  ONNXRuntime
ort_session = onnxruntime.InferenceSession("resnet18_best_loss.onnx", providers=["CPUExecutionProvider" if device == "cpu" else "CUDAExecutionProvider"])

# определим функцию перевода тензора в numpy-массив
def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

# определим входы и выходы
ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(x)}
ort_outs = ort_session.run(None, ort_inputs)
print(ort_outs)
print(out)

# сравним результаты
np.testing.assert_allclose(to_numpy(out[0]), ort_outs[0][0], rtol=2e-02, atol=1e-03)
print("Все хорошо!")