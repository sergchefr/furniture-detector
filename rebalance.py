import os
import shutil
import random
from glob import glob


def rebalance_raw_yolo(source_dir, target_dir, train_ratio=0.7, val_ratio=0.15):
    # 1. Ищем все картинки во всех подпапках исходника
    all_images = glob(os.path.join(source_dir, '**', '*.jpg'), recursive=True)

    # Оставляем только уникальные имена (без путей), чтобы найти пары
    file_names = [os.path.splitext(os.path.basename(x))[0] for x in all_images]
    random.shuffle(file_names)

    # 2. Считаем лимиты
    n_total = len(file_names)
    n_train = int(n_total * train_ratio)
    n_val = int(n_total * val_ratio)

    splits = {
        'train': file_names[:n_train],
        'valid': file_names[n_train:n_train + n_val],
        'test': file_names[n_train + n_val:]
    }

    # 3. Создаем структуру и копируем
    for split_name, names in splits.items():
        img_dest = os.path.join(target_dir, split_name, 'images')
        lbl_dest = os.path.join(target_dir, split_name, 'labels')
        os.makedirs(img_dest, exist_ok=True)
        os.makedirs(lbl_dest, exist_ok=True)

        for name in names:
            # Ищем, где файл лежал изначально (в train, valid или test)
            found = False
            for original_split in ['train', 'valid', 'test']:
                src_img = os.path.join(source_dir, original_split, 'images', f"{name}.jpg")
                src_lbl = os.path.join(source_dir, original_split, 'labels', f"{name}.txt")

                if os.path.exists(src_img) and os.path.exists(src_lbl):
                    shutil.copy2(src_img, os.path.join(img_dest, f"{name}.jpg"))
                    shutil.copy2(src_lbl, os.path.join(lbl_dest, f"{name}.txt"))
                    found = True
                    break
            if not found:
                print(f"Предупреждение: Не найдена пара для {name}")

    print(f"Переразбивка завершена! Новые данные в: {target_dir}")
    for s in splits:
        print(f"{s}: {len(splits[s])} изображений")

rebalance_raw_yolo(source_dir='./dataset/raw', target_dir='dataset/rebalanced')