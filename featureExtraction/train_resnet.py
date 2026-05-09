import torch
import torch.nn as nn
import torch.optim as optim
import time
from model import ResNetFurniture
from data_loader import get_loaders
from utils import plot_history, run_full_evaluation, save_error_samples
import config


def train_feature_extraction():
    # 1. Настройка устройства
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"--- Запуск Feature Extraction на {device} ---")

    # 2. Загрузка данных (используем наш сбалансированный лоадер)
    # Напоминаю: num_workers=2 в data_loader.py спасет твою оперативку
    train_loader, valid_loader, test_loader = get_loaders()
    classes = train_loader.dataset.classes

    # 3. Инициализация предобученной ResNet-18
    model = ResNetFurniture(num_classes=len(classes)).to(device)

    # Вычисляем параметры для отчета (пункт 3.3.3)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"Общее число параметров: {total_params:,}")
    print(f"Обучаемых параметров: {trainable_params:,}")
    # Должно быть около 3,078 — это только веса слоя fc

    # 4. Оптимизатор и Функция потерь
    # filter(lambda p: p.requires_grad, ...) — передаем только размороженный слой
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}
    start_train_time = time.time()

    print("Начинаю обучение (10 эпох)...")

    # 5. Цикл обучения
    for epoch in range(10):
        epoch_start = time.time()
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

        # Валидация
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

        # Сохранение метрик
        history['train_loss'].append(running_loss / len(train_loader))
        history['val_loss'].append(v_loss / len(valid_loader))
        history['train_acc'].append(100 * correct / total)
        history['val_acc'].append(100 * v_corr / v_tot)

        epoch_end = time.time()
        print(f"Эпоха [{epoch + 1}/10] - Время: {epoch_end - epoch_start:.1f}s - "
              f"Loss: {history['train_loss'][-1]:.4f} - Val Acc: {history['val_acc'][-1]:.2f}%")

    print(f"\nОбучение завершено за {(time.time() - start_train_time) / 60:.2f} минут")

    # 6. Автоматическая аналитика (сохранит все графики и матрицу ошибок)
    plot_history(history)
    run_full_evaluation(model, test_loader, device, classes)
    save_error_samples(model, test_loader, device, classes)

    torch.save(model.state_dict(), 'resnet18_feature_extraction.pth')
    print("Модель сохранена.")


if __name__ == '__main__':
    train_feature_extraction()