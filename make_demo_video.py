"""
Створює тестове відео з зображень датасету (щоб перевірити конвеєр на відеофайлі).

Запуск з папки lab3:  python make_demo_video.py
Вхід:  ../lab2/dataset  (папки cats, dogs, birds)
Вихід: data/demo.mp4
"""
import os

import cv2
import numpy as np

DATA_DIR = os.path.join("..", "lab2", "dataset")
OUT = os.path.join("data", "demo.mp4")
W, H, FPS = 640, 480, 5
CLASSES = ["cats", "dogs", "birds"]


def letterbox_frame(img):
    h, w = img.shape[:2]
    r = min(W / w, H / h)
    nw, nh = int(w * r), int(h * r)
    resized = cv2.resize(img, (nw, nh))
    canvas = np.zeros((H, W, 3), dtype=np.uint8)
    x, y = (W - nw) // 2, (H - nh) // 2
    canvas[y:y + nh, x:x + nw] = resized
    return canvas


def main():
    os.makedirs("data", exist_ok=True)
    writer = cv2.VideoWriter(OUT, cv2.VideoWriter_fourcc(*"mp4v"), FPS, (W, H))
    n = 0
    for cls in CLASSES:
        folder = os.path.join(DATA_DIR, cls)
        for fname in sorted(os.listdir(folder)):
            img = cv2.imread(os.path.join(folder, fname))
            if img is None:
                continue
            writer.write(letterbox_frame(img))
            n += 1
    writer.release()
    print(f"Збережено {OUT}: {n} кадрів, {FPS} FPS")


if __name__ == "__main__":
    main()
