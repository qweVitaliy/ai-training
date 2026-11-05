import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

# 1. Визначення параметрів та класів датасету
DATASET_DIR = "dataset"  # Шлях до кореневої папки з даними
CLASSES = ["cats", "dogs", "birds"]
IMG_SIZE = (224, 224)
RANDOM_SEED = 42

np.random.seed(RANDOM_SEED)


# 2. АНАЛІЗ ДАТАСЕТУ

def analyze_dataset(base_dir, classes):
    print("--- АНАЛІЗ ДАТАСЕТУ ---")
    total_images = 0
    class_counts = {}
    corrupted_files = []
    image_sizes = []

    for cls in classes:
        cls_path = os.path.join(base_dir, cls)
        if not os.path.exists(cls_path):
            print(f"Попередження: Папка {cls_path} не знайдена!")
            class_counts[cls] = 0
            continue
            
        files = os.listdir(cls_path)
        valid_count = 0
        
        for file in files:
            file_path = os.path.join(cls_path, file)
            img = cv2.imread(file_path)
            if img is None:
                corrupted_files.append(file_path)
            else:
                valid_count += 1
                image_sizes.append(img.shape)
                
        class_counts[cls] = valid_count
        total_images += valid_count

    print(f"Загальна кількість валідних зображень: {total_images}")
    print(f"Кількість прикладів по класах: {class_counts}")
    print(f"Пошкоджені або нечитабельні файли: {len(corrupted_files)}")
    if image_sizes:
        print(f"Типові розміри (перший файл): {image_sizes[0]}")
    print("-" * 30)
    return class_counts


# 3. ПРЕПОЦЕСИНГ

def preprocess_image(image_path, target_size=IMG_SIZE):
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Неможливо прочитати файл: {image_path}")
    
    # Конвертація BGR -> RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # Зміна розміру
    img = cv2.resize(img, target_size)
    # Нормалізація у діапазон [0, 1]
    img = img.astype(np.float32) / 255.0
    
    return img


# 4. АУГМЕНТАЦІЯ ДАНИХ
def augment_image(img):
    # 1. Горизонтальне відображення
    aug_flip = cv2.flip(img, 1)
    
    # 2. Поворот на 15 градусів
    h, w = img.shape[:2]
    matrix = cv2.getRotationMatrix2D((w // 2, h // 2), 15, 1.0)
    aug_rotate = cv2.warpAffine(img, matrix, (w, h))
    
    # 3. Зміна яскравості
    bright_factor = 1.2
    aug_brightness = np.clip(img.astype(np.float32) * bright_factor, 0, 255).astype(np.uint8)
    
    return aug_flip, aug_rotate, aug_brightness


# 5. РОЗБИТТЯ TRAIN / VAL / TEST

def create_splits(file_paths, labels, test_size=0.2, val_size=0.2):
    file_paths = np.array(file_paths)
    labels = np.array(labels)
    
    # Перемішування індексів
    indices = np.arange(len(file_paths))
    np.random.shuffle(indices)
    
    file_paths = file_paths[indices]
    labels = labels[indices]
    
    n_total = len(file_paths)
    n_test = int(n_total * test_size)
    n_val = int(n_total * val_size)
    
    # Розбиття на вибірки
    test_files, test_labels = file_paths[:n_test], labels[:n_test]
    val_files, val_labels = file_paths[n_test:n_test+n_val], labels[n_test:n_test+n_val]
    train_files, train_labels = file_paths[n_test+n_val:], labels[n_test+n_val:]
    
    print(f"Розмір Train: {len(train_files)}, Val: {len(val_files)}, Test: {len(test_files)}")
    return (train_files, train_labels), (val_files, val_labels), (test_files, test_labels)

# ==========================================
# 6. ВІЗУАЛІЗАЦІЯ РОЗПОДІЛУ ТА ПРИКЛАДІВ
# ==========================================
def visualize_data(class_counts, sample_images_dict):
    # 1. Гістограма розподілу класів
    plt.figure(figsize=(6, 4))
    plt.bar(class_counts.keys(), class_counts.values(), color=['skyblue', 'salmon', 'lightgreen'])
    plt.title("Розподіл класів у датасеті")
    plt.xlabel("Класи")
    plt.ylabel("Кількість зображень")
    plt.show()

    # 2. Візуалізація прикладів
    plt.figure(figsize=(12, 6))
    row_idx = 1
    for cls, images in sample_images_dict.items():
        for i, img in enumerate(images[:4]):
            plt.subplot(len(CLASSES), 4, row_idx)
            img_disp = (img * 255).astype(np.uint8) if img.max() <= 1.0 else img
            plt.imshow(img_disp)
            plt.title(f"{cls} ({i+1})")
            plt.axis('off')
            row_idx += 1
    plt.tight_layout()
    plt.show()

# MAIN

if __name__ == "__main__":
    class_counts = analyze_dataset(DATASET_DIR, CLASSES)
    
    if sum(class_counts.values()) > 0:
        file_paths = []
        labels = []
        sample_images = {}
        
        for cls in CLASSES:
            cls_path = os.path.join(DATASET_DIR, cls)
            if not os.path.exists(cls_path):
                continue
            files = os.listdir(cls_path)
            cls_images = []
            
            for file in files:
                f_path = os.path.join(cls_path, file)
                test_img = cv2.imread(f_path)
                if test_img is not None:
                    file_paths.append(f_path)
                    labels.append(cls)
                    
                    if len(cls_images) < 4:
                        processed = preprocess_image(f_path, IMG_SIZE)
                        cls_images.append(processed)
            
            sample_images[cls] = cls_images

        if len(file_paths) >= 3:
            train_data, val_data, test_data = create_splits(file_paths, labels)
            
            visualize_data(class_counts, sample_images)
            
            if sample_images[CLASSES[0]]:
                sample = sample_images[CLASSES[0]][0]
                img_uint8 = (sample * 255).astype(np.uint8)
                
                flip, rotate, bright = augment_image(img_uint8)
                
                fig, axes = plt.subplots(1, 4, figsize=(12, 3))
                axes[0].imshow(img_uint8); axes[0].set_title("Оригінал"); axes[0].axis('off')
                axes[1].imshow(flip); axes[1].set_title("Flip (Дзеркало)"); axes[1].axis('off')
                axes[2].imshow(rotate); axes[2].set_title("Rotate (Поворот)"); axes[2].axis('off')
                axes[3].imshow(bright); axes[3].set_title("Brightness (Яскравість)"); axes[3].axis('off')
                plt.tight_layout()
                plt.show()
        else:
            print("Занадто мало валідних зображень для розбиття.")
    else:
        print("Датасет порожній або папки не знайдені.")