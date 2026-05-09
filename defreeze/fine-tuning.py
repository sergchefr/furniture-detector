import torch
import torch.nn as nn
import torch.optim as optim
import time
import os

# Твои локальные файлы
from model import ResNetFurniture
from data_loader import get_loaders
from utils import plot_history, run_full_evaluation
import config


def run_fine_tuning():
    # 1. Настройка устройства
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"--- Запуск FINE-TUNING на {device} ---")

    # 2. Загрузка данных
    train_loader, valid_loader, test_loader = get_loaders()
    classes = train_loader.dataset.classes

    # 3. Инициализация модели и ЗАГРУЗКА ВЕСОВ
    model = ResNetFurniture(num_classes=len(classes)).to(device)

    # Путь к весам из прошлого этапа (убедись, что файл лежит в этой же папке)
    weights_path = 'resnet18_feature_extraction.pth'

    if os.path.exists(weights_path):
        model.load_state_dict(torch.load(weights_path))
        print(f"Успешно загружены веса: {weights_path}")
    else:
        print(f"ВНИМАНИЕ: Файл {weights_path} не найден! Начинаем с нуля (ImageNet).")

    # 4. РАЗМОРАЖИВАЕМ ВСЕ СЛОИ
    for param in model.parameters():
        param.requires_grad = True

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Обучаемых параметров: {trainable_params:,} (вся сеть разморожена)")

    # 5. Оптимизатор с ОЧЕНЬ маленьким LR
    # 1e-5 — золотой стандарт для дообучения ResNet
    optimizer = optim.Adam(model.parameters(), lr=1e-5)
    criterion = nn.CrossEntropyLoss()

    history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}

    # Для Fine-tuning обычно хватает 5-7 эпох
    num_epochs = 10
    print(f"Начинаю дообучение ({num_epochs} эпох)...")

    for epoch in range(num_epochs):
        epoch_start = time.time()

        # --- TRAINING ---
        model.train()
        running_loss, correct, total = 0.0, 0, 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        # --- VALIDATION ---
        model.eval()
        v_loss, v_corr, v_tot = 0.0, 0, 0
        with torch.no_grad():
            for images, labels in valid_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                v_loss += criterion(outputs, labels).item()
                _, predicted = torch.max(outputs.data, 1)
                v_tot += labels.size(0)
                v_corr += (predicted == labels).sum().item()

        # Метрики
        history['train_loss'].append(running_loss / len(train_loader))
        history['val_loss'].append(v_loss / len(valid_loader))
        history['train_acc'].append(100 * correct / total)
        history['val_acc'].append(100 * v_corr / v_tot)

        print(f"Epoch [{epoch + 1}/{num_epochs}] - Time: {time.time() - epoch_start:.1f}s - "
              f"Loss: {history['train_loss'][-1]:.4f} - Val Acc: {history['val_acc'][-1]:.2f}%")

    # 6. Финал
    print("\nFine-tuning завершен!")
    plot_history(history)
    run_full_evaluation(model, test_loader, device, classes)

    torch.save(model.state_dict(), 'resnet18_fine_tuned.pth')
    print("Финальная модель сохранена как resnet18_fine_tuned.pth")


if __name__ == '__main__':
    run_fine_tuning()