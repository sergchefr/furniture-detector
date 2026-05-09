import torch
import torch.nn as nn
import torch.optim as optim
import time
from model import BaseCNN
from data_loader import get_loaders
import config
from utils import plot_history, run_full_evaluation, save_error_samples


def train_model():
    # Настройка устройства
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Загрузка данных
    # ВАЖНО: В data_loader.py поставь num_workers=2, чтобы не забить 16ГБ ОЗУ!
    train_loader, valid_loader, test_loader = get_loaders()
    classes = train_loader.dataset.classes

    model = BaseCNN().to(device)

    # Считаем параметры для пункта 3.3.3

    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total trainable parameters: {params:,}")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE)

    history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}

    start_train_time = time.time()

    for epoch in range(20):
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
        val_loss, val_correct, val_total = 0.0, 0, 0
        with torch.no_grad():
            for images, labels in valid_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()

        # Сохраняем метрики
        history['train_loss'].append(running_loss / len(train_loader))
        history['val_loss'].append(val_loss / len(valid_loader))
        history['train_acc'].append(100 * correct / total)
        history['val_acc'].append(100 * val_correct / val_total)

        epoch_end = time.time()
        print(f"Epoch [{epoch + 1}/20] - Time: {epoch_end - epoch_start:.1f}s - "
              f"Loss: {history['train_loss'][-1]:.4f} - Val Acc: {history['val_acc'][-1]:.2f}%")

    total_time = time.time() - start_train_time
    print(f"\nTraining complete in {total_time / 60:.1f} minutes")

    # --- АВТОМАТИЧЕСКИЙ АНАЛИЗ ПОСЛЕ ОБУЧЕНИЯ ---

    # 1. Графики
    plot_history(history)

    # 2. Матрица и F1 на ТЕСТОВЫХ данных
    run_full_evaluation(model, test_loader, device, classes)

    # 3. Картинки с ошибками
    save_error_samples(model, test_loader, device, classes)

    torch.save(model.state_dict(), 'base_cnn_final.pth')


if __name__ == '__main__':
    train_model()