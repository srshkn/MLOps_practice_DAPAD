# data_preprocessing.py
import cv2
from pathlib import Path
import albumentations as A
import numpy as np

# Пути к папкам
DATASET_DIR = Path(__file__).resolve().parent / "dataset"
train_dir = DATASET_DIR / "train"

# Создаём последовательность аугментаций
seq = A.Compose([
    A.HorizontalFlip(p=0.3),
    A.Affine(scale=(0.95, 1.05), rotate=(-10, 10), translate_percent=(-0.05, 0.05)),
    A.ColorJitter(brightness=0.05, contrast=0.05, saturation=0.05, hue=0.02, p=0.5),
    A.GaussNoise(var_limit=(2.0, 10.0), p=0.2),
    A.GaussianBlur(blur_limit=(3,3), p=0.2),
    A.RandomRain(slant_lower=-5, slant_upper=5, drop_length=8, drop_width=1, p=0.1),
    A.RandomSnow(snow_point_lower=0.05, snow_point_upper=0.1, p=0.1),
    A.RandomFog(fog_coef_lower=0.05, fog_coef_upper=0.1, p=0.1)
])

def preprocess_and_save(folder: Path):
    print(f"\n📂 Обработка папки: {folder.name}")
    image_paths = list(folder.rglob("*.*"))
    count = 0

    for img_path in image_paths:
        if not img_path.is_file():
            continue

        # Читаем изображение
        img = cv2.imread(str(img_path))
        if img is None:
            continue

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Применяем аугментации
        augmented = seq(image=img)['image']

        # Сохраняем с припиской "_augmented"
        new_name = img_path.stem + "_augmented" + img_path.suffix
        new_path = img_path.parent / new_name
        augmented_bgr = cv2.cvtColor(augmented, cv2.COLOR_RGB2BGR)
        cv2.imwrite(str(new_path), augmented_bgr)
        count += 1

    print(f"🎨 Создано аугментированных изображений: {count}")

if __name__ == "__main__":
    print("🎉 Начало предобработки...")

    # Тестовый набор не аугментируем, чтобы метрика оставалась честной.
    preprocess_and_save(train_dir)

    print("\n🎉 Предобработка завершена!")