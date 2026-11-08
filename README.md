# Лабораторна робота №3. Попередньо навчена модель і прикладний комп'ютерний зір

Конвеєр детекції об'єктів: OpenCV + попередньо навчена YOLOv8 (COCO).
Модель знаходить класи bird, cat, dog.

## Структура
- `detect.py` — конвеєр: preprocessing, inference, postprocessing (NMS), малювання; вхід: зображення, відео або камера
- `benchmark.py` — вимірювання latency та FPS для трьох конфігурацій (прогрів 10 запусків, 159 зображень)
- `evaluate_dataset.py` — перевірка детектора на датасеті за класами
- `make_demo_video.py` — створює тестове відео з датасету
- `results/` — результати запусків (зображення, відео, таблиці, графік швидкості)

Датасет береться з папки `lab2` (`../lab2/dataset`), тому обидві папки мають бути поруч.

## Встановлення
У папці `lab3` створіть окреме віртуальне середовище (версії numpy відрізняються від lab2):
```
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```
Ваги YOLOv8 Ultralytics завантажує автоматично з GitHub під час першого запуску (файли `yolov8n.pt`, `yolov8s.pt`).

## Запуск
```
python detect.py --source ..\lab2\dataset\dogs\images.jpg
python make_demo_video.py
python detect.py --source data\demo.mp4 --save results\demo_out.mp4 --no-show
python detect.py --source 0          # веб-камера (Esc — вихід)
python benchmark.py
python evaluate_dataset.py
```
