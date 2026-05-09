import torch.nn as nn
from torchvision import models


class ResNetFurniture(nn.Module):
    def __init__(self, num_classes=6):
        super(ResNetFurniture, self).__init__()
        # Загружаем предобученную модель
        self.resnet = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)

        # 1. ЗАМОРОЗКА: отключаем обновление градиентов для всех слоев
        for param in self.resnet.parameters():
            param.requires_grad = False

        # 2. ЗАМЕНА СЛОЯ: у ResNet-18 последний слой называется 'fc' (fully connected)
        # Узнаем количество входных признаков (их 512)
        num_ftrs = self.resnet.fc.in_features

        # Заменяем его на новый линейный слой под твои 6 классов
        # У этого нового слоя requires_grad по умолчанию True
        self.resnet.fc = nn.Linear(num_ftrs, num_classes)

    def forward(self, x):
        return self.resnet(x)