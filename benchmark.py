"""
Вимірювання швидкості (latency та FPS) для трьох конфігурацій моделі.

Запуск з папки lab3:  python benchmark.py
Вхід:  ../lab2/dataset (159 зображень, що більше за вимогу 100)
Вихід: results/benchmark.csv і таблиця в терміналі

Прогрів: WARMUP запусків перед вимірюванням.
Час inference і повний час обробки (preprocess + inference + postprocess) вимірюються окремо.
Читання файлів з диска не входить у час.
"""
import csv
import os
import statistics

import cv2
import numpy as np
import torch

from detect import YoloDetector

DATA_DIR = os.path.join("..", "lab2", "dataset")
CLASSES = ["cats", "dogs", "birds"]
WARMUP = 10
CONFIGS = [
    ("yolov8n", 320),
    ("yolov8n", 640),
    ("yolov8s", 640),
]
OUT_CSV = os.path.join("results", "benchmark.csv")


def load_images():
    paths = []
    for cls in CLASSES:
        folder = os.path.join(DATA_DIR, cls)
        paths += [os.path.join(folder, f) for f in sorted(os.listdir(folder))]
    images = [cv2.imread(p) for p in paths]
    return [img for img in images if img is not None]


def benchmark(weights, imgsz, images):
    detector = YoloDetector(f"{weights}.pt", imgsz=imgsz)
    # прогрів: перші виклики містять ініціалізацію, тому їх не враховуємо
    for img in images[:WARMUP]:
        detector.detect(img)

    timings, counts = [], []
    for img in images:
        dets, t = detector.detect(img)
        timings.append(t)
        counts.append(len(dets))

    def mean(key):
        return statistics.mean(t[key] for t in timings)

    total = mean("total_ms")
    return {
        "model": weights,
        "imgsz": imgsz,
        "n_images": len(images),
        "pre_ms": round(mean("pre_ms"), 2),
        "infer_ms": round(mean("infer_ms"), 2),
        "post_ms": round(mean("post_ms"), 2),
        "total_ms": round(total, 2),
        "fps": round(1000.0 / total, 2),
        "avg_detections": round(statistics.mean(counts), 2),
    }


def main():
    torch.set_num_threads(os.cpu_count() or 1)
    images = load_images()
    print(f"Зображень для тесту: {len(images)}; прогрів: {WARMUP}")
    rows = []
    for weights, imgsz in CONFIGS:
        print(f"Вимірюємо {weights}, imgsz={imgsz} ...")
        rows.append(benchmark(weights, imgsz, images))

    os.makedirs("results", exist_ok=True)
    with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print("\n{:<10} {:>6} {:>9} {:>10} {:>9} {:>10} {:>8} {:>10}".format(
        "модель", "вхід", "pre, мс", "infer, мс", "post, мс", "total, мс", "FPS", "детекцій"))
    for r in rows:
        print("{:<10} {:>6} {:>9.1f} {:>10.1f} {:>9.1f} {:>10.1f} {:>8.2f} {:>10.2f}".format(
            r["model"], r["imgsz"], r["pre_ms"], r["infer_ms"], r["post_ms"],
            r["total_ms"], r["fps"], r["avg_detections"]))
    print(f"\nРезультати збережено: {OUT_CSV}")


if __name__ == "__main__":
    main()
