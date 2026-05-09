import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, f1_score

from model import ResNetFurniture
from data_loader import get_loaders
import config


def run_full_test():
    device = torch.device(config.DEVICE)

    # 1. Загрузка данных (нам нужен только test_loader)
    # Убедись, что get_loaders() возвращает (train, valid, test)
    _, _, test_loader = get_loaders()
    classes = test_loader.dataset.classes

    # 2. Инициализация модели и загрузка весов
    model = ResNetFurniture(num_classes=len(classes)).to(device)
    model.load_state_dict(torch.load('resnet18_feature_extraction.pth'))
    model.eval()

    all_preds = []
    all_labels = []
    incorrect_examples = []  # Для визуального анализа ошибок

    print("Запуск финального тестирования...")

    with torch.no_grad():
        for images, labels in test_loader:
            images_gpu, labels_gpu = images.to(device), labels.to(device)
            outputs = model(images_gpu)
            _, preds = torch.max(outputs, 1)

            # Сохраняем для метрик
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

            # Ищем ошибки для пункта 3.3.4
            for i in range(len(preds)):
                if preds[i] != labels_gpu[i] and len(incorrect_examples) < 5:
                    # Сохраняем: картинку, предсказание, реальный класс
                    incorrect_examples.append((images[i], preds[i].cpu().item(), labels[i].item()))

    # --- ОТЧЕТ ПО ПУНКТУ 3.3.2 ---
    print("\n" + "=" * 30)
    print("ИТОГОВЫЕ МЕТРИКИ (TEST SET)")
    print("=" * 30)

    # Точность и Macro-F1
    report = classification_report(all_labels, all_preds, target_names=classes)
    macro_f1 = f1_score(all_labels, all_preds, average='macro')
    print(report)
    print(f"Итоговый Macro-F1: {macro_f1:.4f}")

    # Построение Confusion Matrix
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.title('Матрица ошибок — Base CNN')
    plt.ylabel('Реальные классы')
    plt.xlabel('Предсказания модели')
    plt.savefig('confusion_matrix.png')
    plt.show()

    # --- ВИЗУАЛЬНЫЙ АНАЛИЗ ОШИБОК (ПУНКТ 3.3.4) ---
    print("\nГенерация изображений с ошибками...")
    plt.figure(figsize=(18, 5))
    for i, (img, pred, label) in enumerate(incorrect_examples):
        plt.subplot(1, 5, i + 1)

        # Денормализация картинки для корректного отображения
        img_vis = img.permute(1, 2, 0).numpy()
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        img_vis = std * img_vis + mean
        img_vis = np.clip(img_vis, 0, 1)

        plt.imshow(img_vis)
        plt.title(f"Предсказано: {classes[pred]}\nРеально: {classes[label]}", color='red', fontsize=10)
        plt.axis('off')

    plt.suptitle("Примеры ошибок модели (для визуального анализа)", fontsize=16)
    plt.savefig('error_analysis.png')
    plt.show()

    # --- ПАРАМЕТРЫ МОДЕЛИ (ПУНКТ 3.3.3) ---
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nОбщее число параметров модели: {total_params:,}")


if __name__ == "__main__":
    run_full_test()