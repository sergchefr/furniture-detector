import os
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, WeightedRandomSampler
import config


def get_loaders():
    # 1. Аугментация (как мы обсуждали ранее)
    train_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomRotation(20),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    val_test_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # 2. Загрузка датасетов
    train_dataset = datasets.ImageFolder(os.path.join(config.DATA_PATH, 'train'), train_transforms)
    valid_dataset = datasets.ImageFolder(os.path.join(config.DATA_PATH, 'valid'), val_test_transforms)
    test_dataset = datasets.ImageFolder(os.path.join(config.DATA_PATH, 'test'), val_test_transforms)

    # --- ЛОГИКА БАЛАНСИРОВКИ (SAMPLER) ---

    # Получаем метки всех картинок в трейне [0, 0, 1, 3, 0, ...]
    targets = torch.tensor(train_dataset.targets)

    # Считаем количество примеров в каждом классе
    class_sample_count = torch.tensor(
        [(targets == t).sum() for t in torch.unique(targets, sorted=True)]
    )

    # Вес класса = 1 / количество_примеров
    weight = 1. / class_sample_count.float()

    # Присваиваем вес каждой конкретной картинке в датасете
    samples_weight = torch.tensor([weight[t] for t in targets])

    # Создаем самплер. num_samples равен длине датасета,
    # чтобы за одну эпоху мы прошли столько же итераций, сколько и раньше.
    sampler = WeightedRandomSampler(
        weights=samples_weight,
        num_samples=len(samples_weight),
        replacement=True
    )

    # 3. DataLoader
    loader_args = {
        'batch_size': config.BATCH_SIZE,
        'num_workers': 2,
        'pin_memory': True,
        'persistent_workers': True
    }

    # ВАЖНО: shuffle=True НЕЛЬЗЯ использовать вместе с sampler
    train_l = DataLoader(train_dataset, sampler=sampler, **loader_args)

    valid_l = DataLoader(valid_dataset, shuffle=False, **loader_args)
    test_l = DataLoader(test_dataset, shuffle=False, **loader_args)

    print(f"Классы: {train_dataset.classes}")
    print(f"Распределение в трейне: {class_sample_count.tolist()}")

    return train_l, valid_l, test_l