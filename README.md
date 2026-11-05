# Лабораторна робота №2. Згорткова нейронна мережа (PyTorch)

## Структура
- `script.py` — датасет, розбиття train/val/test (з лабораторної №1)
- `train_cnn.py` — Dataset/DataLoader, CNN, цикл навчання, оцінка на test
- `dataset/` — зображення по класах: `cats`, `dogs`, `birds`
- `results/` — результати запуску (створюється автоматично)

## Встановлення
Рекомендовано створити віртуальне середовище в цій папці:

```
python -m venv .venv
.venv\Scripts\activate        # Windows
pip install -r requirements.txt
```

## Запуск
Запускати з папки `lab2`:

```
python train_cnn.py
```

## Результати
- `results/best_model.pt` — ваги найкращої моделі (за validation accuracy)
- `results/curves.png` — графіки loss та accuracy за епохами
- `results/confusion_matrix.png` — confusion matrix на test
- `results/examples.png` — правильні та помилкові прогнози
- `results/predictions_test.csv` — прогнози по кожному test-зображенню

Датасет і ваги моделі до репозиторію не додаються.
