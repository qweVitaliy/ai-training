"""
Розпізнавання одного фото найкращою моделлю (для демонстрації на захисті).

Запуск з папки lab2:  python predict.py шлях\до\фото.jpg
Потрібен файл results/best_model.pt (з'являється після train_cnn.py).
"""
import os
import sys

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

CLASSES = ["cats", "dogs", "birds"]
NAMES_UA = {"cats": "кіт", "dogs": "собака", "birds": "птах"}
IMG_SIZE = (224, 224)
WEIGHTS = os.path.join("results", "best_model.pt")


# Та сама архітектура, що й у train_cnn.py: без неї ваги не завантажаться
class SimpleCNN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        return self.classifier(self.pool(self.features(x)))


def main():
    if len(sys.argv) < 2:
        print("Використання: python predict.py шлях_до_фото.jpg")
        return

    img_bgr = cv2.imread(sys.argv[1])
    if img_bgr is None:
        print("Не вдалося прочитати файл. Перевірте шлях і формат (jpg або png).")
        return
    if not os.path.exists(WEIGHTS):
        print(f"Немає файлу {WEIGHTS}. Спочатку запустіть train_cnn.py")
        return

    img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, IMG_SIZE)
    x = torch.from_numpy(img).permute(2, 0, 1).float().div(255.0).unsqueeze(0)

    model = SimpleCNN(num_classes=len(CLASSES))
    model.load_state_dict(torch.load(WEIGHTS, map_location="cpu"))
    model.eval()

    with torch.no_grad():
        probs = torch.softmax(model(x), dim=1)[0].numpy()
    pred = int(np.argmax(probs))

    print("Ймовірності по класах:")
    for i, cls in enumerate(CLASSES):
        print(f"  {NAMES_UA[cls]:>7}: {probs[i]:.1%}")
    print(f"\nМодель вважає, що це: {NAMES_UA[CLASSES[pred]]} ({probs[pred]:.1%})")

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.imshow(img)
    ax.axis("off")
    ax.set_title(f"Прогноз: {NAMES_UA[CLASSES[pred]]} ({probs[pred]:.0%})")
    plt.show()


if __name__ == "__main__":
    main()
