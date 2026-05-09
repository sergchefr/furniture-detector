import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import os
import shutil
from model import ResNetFurniture  # Убедись, что файл model.py в этой же папке


def run_inference():
    # 1. Настройки
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_path = 'resnet18_fine_tuned.pth'
    source_folder = 'predict_me'  # Папка с твоими новыми картинками
    classes = ['backpack', 'book', 'chair', 'sofa', 'table', 'vase']

    # 2. Подготовка модели
    model = ResNetFurniture(num_classes=len(classes)).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # 3. Трансформы (в точности как при валидации)
    inference_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # Создаем папки для классов, если их нет
    for cls in classes:
        os.makedirs(os.path.join(source_folder, cls), exist_ok=True)

    # 4. Процесс распознавания
    print(f"Начинаю распознавание в папке: {source_folder}...")

    valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp')
    files = [f for f in os.listdir(source_folder) if f.lower().endswith(valid_extensions)]

    if not files:
        print("Картинки не найдены. Положи файлы в папку 'predict_me'.")
        return

    with torch.no_grad():
        for filename in files:
            img_path = os.path.join(source_folder, filename)

            try:
                # Загрузка и предобработка
                img = Image.open(img_path).convert('RGB')
                tensor = inference_transforms(img).unsqueeze(0).to(device)

                # Предсказание
                outputs = model(tensor)
                _, predicted = torch.max(outputs, 1)
                class_name = classes[predicted.item()]

                # Перемещение
                dest_path = os.path.join(source_folder, class_name, filename)
                shutil.move(img_path, dest_path)

                print(f"[{filename}] -> Распознано как: {class_name}")

            except Exception as e:
                print(f"Ошибка при обработке {filename}: {e}")

    print("\nГотово! Все объекты отсортированы по подпапкам.")


if __name__ == '__main__':
    run_inference()