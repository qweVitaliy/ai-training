"""
Лабораторна робота №2. Розробка та навчання згорткової нейронної мережі (PyTorch).

Запуск з папки lab2:  python train_cnn.py
Результати зберігаються в папку results/.

Датасет і розбиття train/val/test беруться з лабораторної роботи №1 (script.py).
"""
import os
import random

import cv2
import matplotlib

matplotlib.use("Agg")  # графіки зберігаються у файли, без спливаючих вікон
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
import torchvision.transforms as T  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_score,
    recall_score,
)
from torch.utils.data import DataLoader, Dataset  # noqa: E402

from script import CLASSES, DATASET_DIR, IMG_SIZE, RANDOM_SEED, create_splits  # noqa: E402

# ==========================================
# 0. ГІПЕРПАРАМЕТРИ
# ==========================================
EPOCHS = 30
BATCH_SIZE = 4
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
RESULTS_DIR = "results"
BEST_WEIGHTS = os.path.join(RESULTS_DIR, "best_model.pt")


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


# ==========================================
# 1. DATASET / DATALOADER
# ==========================================
class ImageDataset(Dataset):
    """Читає зображення з диска, перетворює їх у тензор C x H x W у діапазоні [0, 1]."""

    def __init__(self, paths, labels, transform=None):
        self.paths = list(paths)
        self.labels = [CLASSES.index(str(lbl)) for lbl in labels]
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = cv2.imread(str(self.paths[idx]))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, IMG_SIZE)
        img = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
        if self.transform is not None:
            img = self.transform(img)
        return img, self.labels[idx]


# ==========================================
# 2. МОДЕЛЬ: БАЗОВА CNN З ТРЬОМА ЗГОРТКОВИМИ БЛОКАМИ
# ==========================================
class SimpleCNN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.features = nn.Sequential(
            # блок 1: 3 -> 32 канали, 224 -> 112
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            # блок 2: 32 -> 64 канали, 112 -> 56
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            # блок 3: 64 -> 128 канали, 56 -> 28
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
        x = self.features(x)
        x = self.pool(x)
        return self.classifier(x)


# ==========================================
# 3. ЦИКЛ TRAIN / VALIDATION
# ==========================================
def run_epoch(model, loader, criterion, optimizer, device, train):
    """Одна епоха: при train=True навчає модель, при train=False лише оцінює."""
    model.train(train)  # train() вмикає Dropout/BatchNorm у режимі навчання, eval() — у режимі оцінки
    total_loss, correct, total = 0.0, 0, 0

    with torch.enable_grad() if train else torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)

            if train:
                optimizer.zero_grad()

            logits = model(x)
            loss = criterion(logits, y)

            if train:
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * x.size(0)
            correct += (logits.argmax(dim=1) == y).sum().item()
            total += x.size(0)

    return total_loss / total, correct / total


def plot_curves(history, path):
    epochs = range(1, len(history["train_loss"]) + 1)
    fig, (ax_loss, ax_acc) = plt.subplots(1, 2, figsize=(12, 4))

    ax_loss.plot(epochs, history["train_loss"], label="train")
    ax_loss.plot(epochs, history["val_loss"], label="validation")
    ax_loss.set_title("Loss за епохами")
    ax_loss.set_xlabel("Епоха")
    ax_loss.set_ylabel("Cross-entropy loss")
    ax_loss.legend()
    ax_loss.grid(alpha=0.3)

    ax_acc.plot(epochs, history["train_acc"], label="train")
    ax_acc.plot(epochs, history["val_acc"], label="validation")
    ax_acc.set_title("Accuracy за епохами")
    ax_acc.set_xlabel("Епоха")
    ax_acc.set_ylabel("Accuracy")
    ax_acc.set_ylim(0, 1.05)
    ax_acc.legend()
    ax_acc.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


# ==========================================
# 4. ОЦІНКА НА TEST
# ==========================================
@torch.no_grad()
def predict(model, loader, device):
    """Повертає істинні мітки, передбачені мітки та ймовірності (порядок збігається з loader без shuffle)."""
    model.eval()
    y_true, y_pred, probs_all = [], [], []
    for x, y in loader:
        logits = model(x.to(device))
        probs = torch.softmax(logits, dim=1).cpu().numpy()
        probs_all.append(probs)
        y_pred.extend(probs.argmax(axis=1).tolist())
        y_true.extend(y.tolist())
    return np.array(y_true), np.array(y_pred), np.concatenate(probs_all)


def plot_confusion_matrix(cm, path):
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(CLASSES)), CLASSES)
    ax.set_yticks(range(len(CLASSES)), CLASSES)
    ax.set_xlabel("Передбачений клас")
    ax.set_ylabel("Істинний клас")
    ax.set_title("Confusion matrix (test)")
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, cm[i, j], ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def plot_examples(paths, y_true, y_pred, probs, path, max_per_group=5):
    """Показує до 5 правильних і до 5 помилкових прогнозів з test-вибірки."""
    correct_idx = np.where(y_true == y_pred)[0][:max_per_group]
    wrong_idx = np.where(y_true != y_pred)[0][:max_per_group]
    rows = [("Правильно", correct_idx), ("Помилка", wrong_idx)]

    n_cols = max_per_group
    fig, axes = plt.subplots(2, n_cols, figsize=(3 * n_cols, 7), squeeze=False)
    for r, (title, idx) in enumerate(rows):
        for c in range(n_cols):
            ax = axes[r][c]
            ax.axis("off")
            if c >= len(idx):
                continue
            i = idx[c]
            img = cv2.cvtColor(cv2.imread(paths[i]), cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, IMG_SIZE)
            ax.imshow(img)
            ax.set_title(
                f"{title}\nіст.: {CLASSES[y_true[i]]}\nпрогноз: {CLASSES[y_pred[i]]} "
                f"({probs[i][y_pred[i]]:.2f})",
                fontsize=9,
            )
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return len(correct_idx), len(wrong_idx)


# ==========================================
# MAIN
# ==========================================
def main():
    set_seed(RANDOM_SEED)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Пристрій: {device}")

    # 1. Збираємо коректні файли та розбиваємо їх так само, як у лабораторній №1
    file_paths, labels = [], []
    for cls in CLASSES:
        cls_dir = os.path.join(DATASET_DIR, cls)
        for fname in sorted(os.listdir(cls_dir)):
            fpath = os.path.join(cls_dir, fname)
            if cv2.imread(fpath) is not None:
                file_paths.append(fpath)
                labels.append(cls)

    (tr_f, tr_l), (va_f, va_l), (te_f, te_l) = create_splits(file_paths, labels)
    if min(len(tr_f), len(va_f), len(te_f)) == 0:
        raise SystemExit("Замало зображень: у якійсь вибірці (train/val/test) немає жодного прикладу.")

    # 2. Dataset і DataLoader; аугментація лише для train
    train_tf = T.Compose([T.RandomHorizontalFlip(p=0.5), T.RandomRotation(15)])
    train_ds = ImageDataset(tr_f, tr_l, transform=train_tf)
    val_ds = ImageDataset(va_f, va_l)
    test_ds = ImageDataset(te_f, te_l)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)

    # 3. Модель, функція втрат, оптимізатор
    model = SimpleCNN(num_classes=len(CLASSES)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    print(f"Параметрів у моделі: {sum(p.numel() for p in model.parameters()):,}")

    # 4. Навчання; найкращі ваги зберігаємо за validation accuracy
    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
    best_val_acc, best_epoch = -1.0, -1

    for epoch in range(1, EPOCHS + 1):
        tr_loss, tr_acc = run_epoch(model, train_loader, criterion, optimizer, device, train=True)
        va_loss, va_acc = run_epoch(model, val_loader, criterion, optimizer, device, train=False)

        history["train_loss"].append(tr_loss)
        history["val_loss"].append(va_loss)
        history["train_acc"].append(tr_acc)
        history["val_acc"].append(va_acc)

        print(f"Епоха {epoch:02d} | train loss {tr_loss:.4f}, acc {tr_acc:.3f} | "
              f"val loss {va_loss:.4f}, acc {va_acc:.3f}")

        if va_acc > best_val_acc:
            best_val_acc, best_epoch = va_acc, epoch
            torch.save(model.state_dict(), BEST_WEIGHTS)

    print(f"\nНайкраща модель: епоха {best_epoch}, validation accuracy {best_val_acc:.3f}")
    plot_curves(history, os.path.join(RESULTS_DIR, "curves.png"))

    # 5. Оцінка найкращої моделі на test-наборі
    model.load_state_dict(torch.load(BEST_WEIGHTS, map_location=device))
    y_true, y_pred, probs = predict(model, test_loader, device)
    n = len(CLASSES)
    labels_idx = list(range(n))

    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, labels=labels_idx, average=None, zero_division=0)
    rec = recall_score(y_true, y_pred, labels=labels_idx, average=None, zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=labels_idx)

    print("\n--- ОЦІНКА НА TEST ---")
    print(f"Розмір test-вибірки: {len(y_true)}")
    print(f"Accuracy: {acc:.3f}")
    for i, cls in enumerate(CLASSES):
        print(f"{cls:>6}: precision {prec[i]:.3f}, recall {rec[i]:.3f}")
    print("\nConfusion matrix (рядки — істинні, стовпці — передбачені):")
    print(f"classes: {CLASSES}")
    print(cm)
    print("\n" + classification_report(y_true, y_pred, labels=labels_idx,
                                       target_names=CLASSES, zero_division=0))

    plot_confusion_matrix(cm, os.path.join(RESULTS_DIR, "confusion_matrix.png"))

    n_ok, n_err = plot_examples(te_f, y_true, y_pred, probs,
                                os.path.join(RESULTS_DIR, "examples.png"))
    print(f"Прикладів на графіку: правильних {n_ok}, помилкових {n_err}")
    if n_ok < 5 or n_err < 5:
        print("УВАГА: завдання вимагає не менше 5 правильних і 5 помилкових прикладів. "
              "Для цього в test-вибірці має бути більше зображень; додайте дані до датасету.")

    # 6. Таблиця прогнозів для звіту
    with open(os.path.join(RESULTS_DIR, "predictions_test.csv"), "w", encoding="utf-8") as f:
        f.write("file,true,pred,correct\n")
        for p, t, pr in zip(te_f, y_true, y_pred):
            f.write(f"{p},{CLASSES[t]},{CLASSES[pr]},{int(t == pr)}\n")

    print(f"\nГотово. Результати збережено в папку: {os.path.abspath(RESULTS_DIR)}")


if __name__ == "__main__":
    main()
