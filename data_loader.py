import os
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import config


def get_loaders():
    """
    Создает загрузчики данных для обучения, валидации и финального тестирования.
    """

    # 1. Трансформации для тренировочной выборки (с аугментацией)
    train_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomRotation(15),  # Случайный поворот
        transforms.RandomHorizontalFlip(),  # Отражение по горизонтали
        transforms.ToTensor(),
        # Стандартная нормализация для ImageNet (подходит и для ResNet)
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # 2. Трансформации для валидации и теста (только изменение размера и нормализация)
    val_test_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # 3. Создание объектов Dataset
    # Пути берутся из твоего config.py (например, DATA_PATH = './dataset/cropped')
    train_dataset = datasets.ImageFolder(
        os.path.join(config.DATA_PATH, 'train'),
        transform=train_transforms
    )
    valid_dataset = datasets.ImageFolder(
        os.path.join(config.DATA_PATH, 'valid'),
        transform=val_test_transforms
    )
    test_dataset = datasets.ImageFolder(
        os.path.join(config.DATA_PATH, 'test'),
        transform=val_test_transforms
    )

    # 4. Параметры для DataLoader
    # num_workers=4 задействует 4 потока процессора для подготовки фото
    # pin_memory=True ускоряет перенос данных в видеопамять
    loader_args = {
        'batch_size': config.BATCH_SIZE,
        'num_workers': 4,
        'pin_memory': True,
        'persistent_workers': True  # Оставляет воркеры активными между эпохами
    }

    # 5. Создание DataLoader
    train_loader = DataLoader(train_dataset, shuffle=True, **loader_args)
    valid_loader = DataLoader(valid_dataset, shuffle=False, **loader_args)
    test_loader = DataLoader(test_dataset, shuffle=False, **loader_args)

    # Вывод информации о классах для проверки
    print(f"Найдено классов: {len(train_dataset.classes)} ({train_dataset.classes})")
    print(f"Размер выборок: Train={len(train_dataset)}, Valid={len(valid_dataset)}, Test={len(test_dataset)}")

    return train_loader, valid_loader, test_loader


if __name__ == "__main__":
    # Короткий тест: проверяем, что всё грузится
    tr, val, ts = get_loaders()
    images, labels = next(iter(tr))
    print(f"Формат батча: {images.shape}")  # Должно быть [batch_size, 3, 224, 224]