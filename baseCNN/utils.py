import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import numpy as np


def plot_history(history):
    plt.figure(figsize=(12, 5))

    # График Loss
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Val Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    # График Accuracy
    plt.subplot(1, 2, 2)
    plt.plot(history['train_acc'], label='Train Acc')
    plt.plot(history['val_acc'], label='Val Acc')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()

    plt.tight_layout()
    plt.savefig('learning_curves.png')
    plt.show()


def run_full_evaluation(model, loader, device, classes):
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # 1. Матрица ошибок
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=classes, yticklabels=classes)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.savefig('confusion_matrix.png')
    plt.show()

    # 2. Текстовый отчет
    print("\n" + "=" * 30)
    print("ИТОГОВЫЕ МЕТРИКИ (TEST SET)")
    print("=" * 30)
    print(classification_report(all_labels, all_preds, target_names=classes))


def save_error_samples(model, loader, device, classes):
    model.eval()
    errors = []

    with torch.no_grad():
        for images, labels in loader:
            images_dev = images.to(device)
            outputs = model(images_dev)
            _, preds = torch.max(outputs, 1)

            for i in range(len(preds)):
                if preds[i] != labels[i]:
                    # Сохраняем картинку, правильный класс и то, что предсказала сеть
                    errors.append((images[i], labels[i], preds[i]))
            if len(errors) > 20: break  # Нам хватит для выбора

    # Рисуем 5 примеров (по одному на разные случаи)
    plt.figure(figsize=(15, 5))
    for i in range(min(5, len(errors))):
        img, true_lab, pred_lab = errors[i]
        img = img.permute(1, 2, 0).numpy()
        # Денормировка для корректного цвета
        img = img * [0.229, 0.224, 0.225] + [0.485, 0.456, 0.406]
        img = np.clip(img, 0, 1)

        plt.subplot(1, 5, i + 1)
        plt.imshow(img)
        plt.title(f"True: {classes[true_lab]}\nPred: {classes[pred_lab]}", color='red')
        plt.axis('off')

    plt.tight_layout()
    plt.savefig('error_analysis.png')
    plt.show()