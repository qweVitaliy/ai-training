"""
Показ результатів лабораторної №2 для захисту.

Запуск з папки lab2:  python show_results.py
Виводить метрики в термінал (перераховані з results/predictions_test.csv)
і відкриває вікна з графіками: loss/accuracy, confusion matrix, приклади прогнозів.
"""
import csv
import os

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_score,
    recall_score,
)

RESULTS_DIR = "results"
CLASSES = ["cats", "dogs", "birds"]
PICTURES = [
    ("curves.png", "Loss та accuracy за епохами"),
    ("confusion_matrix.png", "Confusion matrix на test-вибірці"),
    ("examples.png", "Правильні та помилкові прогнози"),
]


def load_predictions(path):
    y_true, y_pred = [], []
    with open(path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            y_true.append(CLASSES.index(row["true"]))
            y_pred.append(CLASSES.index(row["pred"]))
    return np.array(y_true), np.array(y_pred)


def print_metrics(y_true, y_pred):
    labels = list(range(len(CLASSES)))
    prec = precision_score(y_true, y_pred, labels=labels, average=None, zero_division=0)
    rec = recall_score(y_true, y_pred, labels=labels, average=None, zero_division=0)

    print("--- РЕЗУЛЬТАТИ НА TEST ---")
    print(f"Розмір test-вибірки: {len(y_true)}")
    print(f"Accuracy: {accuracy_score(y_true, y_pred):.3f}")
    for i, cls in enumerate(CLASSES):
        print(f"{cls:>6}: precision {prec[i]:.3f}, recall {rec[i]:.3f}")
    print("\nConfusion matrix (рядки — істинні, стовпці — передбачені):")
    print(f"classes: {CLASSES}")
    print(confusion_matrix(y_true, y_pred, labels=labels))
    print("\n" + classification_report(y_true, y_pred, labels=labels,
                                       target_names=CLASSES, zero_division=0))


def show_pictures():
    opened = 0
    for fname, title in PICTURES:
        path = os.path.join(RESULTS_DIR, fname)
        if not os.path.exists(path):
            print(f"Немає файлу {path}. Спочатку запустіть train_cnn.py")
            continue
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.imshow(mpimg.imread(path))
        ax.axis("off")
        ax.set_title(title)
        fig.tight_layout()
        opened += 1
    if opened:
        plt.show()


def main():
    pred_path = os.path.join(RESULTS_DIR, "predictions_test.csv")
    if os.path.exists(pred_path):
        y_true, y_pred = load_predictions(pred_path)
        print_metrics(y_true, y_pred)
    else:
        print(f"Немає файлу {pred_path}. Спочатку запустіть train_cnn.py")
    show_pictures()


if __name__ == "__main__":
    main()
