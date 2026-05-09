import torch
import torch.nn as nn
import torch.optim as optim
from model import BaseCNN
from data_loader import get_loaders
import config

# 1. Инициализация
train_loader, valid_loader = get_loaders()
# Явно вытащим device из конфига для удобства
device = torch.device(config.DEVICE)
print(device)

model = BaseCNN().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE)

history = {'train_loss': [], 'val_acc': []}

# 2. Основной цикл
for epoch in range(config.EPOCHS):  # Используем значение из конфига
    model.train()
    total_loss = 0
    for images, labels in train_loader:
        # Перенос на GPU (RTX 5060 Ti)
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)  # Прогноз
        loss = criterion(outputs, labels)
        loss.backward()  # Считаем градиенты
        optimizer.step()  # Обновляем веса

        total_loss += loss.item()

    # 3. Валидация
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for images, labels in valid_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    history['train_loss'].append(total_loss / len(train_loader))
    history['val_acc'].append(accuracy)

    print(f"Эпоха {epoch + 1}/{config.EPOCHS}: Loss {total_loss / len(train_loader):.4f}, Val Acc {accuracy:.2f}%")

# 4. Сохранение весов (чтобы не потерять результат)
torch.save(model.state_dict(), 'base_cnn_furniture.pth')
print("Модель сохранена!")