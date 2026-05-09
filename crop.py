import os
import cv2
import numpy as np


def crop_to_folders(source_split_path, save_path, padding_percent=10):
    """
    source_split_path: путь к папке выборки (где лежат images и labels)
    save_path: куда сохранять нарезанные папки (например, 'cropped_train')
    padding_percent: сколько процентов добавить к рамке
    """
    # Твой реальный список классов
    real_classes = ['book', 'backpack', 'chair', 'table', 'sofa', 'vase']

    img_dir = os.path.join(source_split_path, 'images')
    lbl_dir = os.path.join(source_split_path, 'labels')

    if not os.path.exists(lbl_dir):
        print(f"Ошибка: Папка с метками не найдена по пути {lbl_dir}")
        return

    for label_file in os.listdir(lbl_dir):
        if not label_file.endswith('.txt'): continue

        img_name = label_file.replace('.txt', '.jpg')
        img = cv2.imread(os.path.join(img_dir, img_name))
        if img is None: continue

        h_img, w_img, _ = img.shape

        with open(os.path.join(lbl_dir, label_file), 'r') as f:
            for i, line in enumerate(f.readlines()):
                data = line.split()
                if not data: continue

                cls_idx = int(data[0])
                x_c, y_c, w_n, h_n = map(float, data[1:])

                # 1. Переводим нормализованные координаты в пиксели
                w_box = w_n * w_img
                h_box = h_n * h_img
                x_center = x_c * w_img
                y_center = y_c * h_img

                # 2. Добавляем Padding
                pad_w = w_box * (padding_percent / 100)
                pad_h = h_box * (padding_percent / 100)

                w_pad = w_box + pad_w
                h_pad = h_box + pad_h

                # 3. Приводим к квадрату (берем сторону по максимуму)
                side = max(w_pad, h_pad)

                # Координаты углов квадрата
                x1 = int(max(0, x_center - side / 2))
                y1 = int(max(0, y_center - side / 2))
                x2 = int(min(w_img, x_center + side / 2))
                y2 = int(min(h_img, y_center + side / 2))

                crop = img[y1:y2, x1:x2]

                if crop.size > 0:
                    class_name = real_classes[cls_idx]
                    target_dir = os.path.join(save_path, class_name)
                    os.makedirs(target_path := target_dir, exist_ok=True)

                    # Сохраняем результат
                    cv2.imwrite(os.path.join(target_path, f"{label_file[:-4]}_{i}.jpg"), crop)

    print(f"Нарезка завершена. Результаты в папке: {save_path}")


# ПРИМЕР ЗАПУСКА:
# Предположим, твоя переразбитая папка обучения лежит в './rebalanced/train'
crop_to_folders(
    source_split_path='dataset/rebalanced/valid',
    save_path='dataset/cropped_valid',
    padding_percent=15  # Добавит по 15% с каждой стороны
)