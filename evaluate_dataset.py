"""
Перевірка детектора на датасеті: скільки зображень кожного класу знайдено правильно.

Запуск з папки lab3:  python evaluate_dataset.py
Вхід:  ../lab2/dataset (cats, dogs, birds)
Вихід: results/detection_by_class.csv і таблиця в терміналі
Зображення, де знайдено правильний клас, рахуються як "виявлено"; решта — "не виявлено".
Окремо рахуються власні фото (великої роздільності) і зображення CIFAR-10 (32x32).
"""
import csv
import os

import cv2

from detect import TARGET_CLASSES, YoloDetector

DATA_DIR = os.path.join("..", "lab2", "dataset")
CLASSES = {"cats": "cat", "dogs": "dog", "birds": "bird"}
OUT = os.path.join("results", "detection_by_class.csv")


def is_own_photo(fname):
    # власні фото мають імена без префікса cifar_
    return not fname.startswith("cifar_")


def main():
    detector = YoloDetector("yolov8n.pt", imgsz=640)
    rows = []
    for folder, true_label in CLASSES.items():
        for fname in sorted(os.listdir(os.path.join(DATA_DIR, folder))):
            img = cv2.imread(os.path.join(DATA_DIR, folder, fname))
            if img is None:
                continue
            dets, _ = detector.detect(img)
            labels = [d[2] for d in dets]
            if true_label in labels:
                status = "правильний клас знайдено"
            elif labels:
                status = "знайдено інший клас"
            else:
                status = "не знайдено"
            rows.append({
                "source": "власне фото" if is_own_photo(fname) else "CIFAR-10",
                "class": true_label,
                "file": fname,
                "result": status,
                "detections": len(dets),
            })

    os.makedirs("results", exist_ok=True)
    with open(OUT, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    print("{:<12} {:<6} {:>8} {:>12} {:>14} {:>12}".format(
        "джерело", "клас", "всього", "знайдено", "інший клас", "не знайдено"))
    for source in ["власне фото", "CIFAR-10"]:
        for cls in ["cat", "dog", "bird"]:
            sub = [r for r in rows if r["source"] == source and r["class"] == cls]
            if not sub:
                continue
            found = sum(r["result"].startswith("правильний") for r in sub)
            other = sum(r["result"].startswith("знайдено інший") for r in sub)
            none = sum(r["result"] == "не знайдено" for r in sub)
            print("{:<12} {:<6} {:>8} {:>12} {:>14} {:>12}".format(
                source, cls, len(sub), found, other, none))
    print(f"\nРезультати збережено: {OUT}")


if __name__ == "__main__":
    main()
