"""
Лабораторна робота №3. Конвеєр детекції об'єктів: OpenCV + попередньо навчена YOLOv8 (COCO).

Модель виявляє класи bird, cat, dog (вони є серед 80 класів COCO).
Preprocessing і postprocessing реалізовано власноруч; Ultralytics використовується лише
для завантаження ваг, а forward-прохід виконує torch-модуль моделі.

Приклади запуску (з папки lab3):
  python detect.py --source data/photos/dog.jpg
  python detect.py --source data/demo.mp4 --save results/demo_out.mp4 --no-show
  python detect.py --source 0                     # веб-камера
"""
import argparse
import os
import time

import cv2
import torch
import torchvision
from ultralytics import YOLO

# Індекси класів COCO, які нас цікавлять
TARGET_CLASSES = {14: "bird", 15: "cat", 16: "dog"}
COLORS = {"bird": (0, 200, 255), "cat": (255, 128, 0), "dog": (0, 220, 0)}
PAD_VALUE = 114  # сірий колір для доповнення letterbox


class YoloDetector:
    def __init__(self, weights="yolov8n.pt", imgsz=640, conf=0.25, iou=0.45):
        self.model = YOLO(weights).model.eval().float()  # torch-модуль, без обгортки Ultralytics
        self.imgsz = imgsz
        self.conf = conf
        self.iou = iou

    def preprocess(self, img_bgr):
        """Letterbox: зберігаємо пропорції, доповнюємо до квадрата imgsz x imgsz."""
        h, w = img_bgr.shape[:2]
        r = min(self.imgsz / h, self.imgsz / w)
        new_w, new_h = int(round(w * r)), int(round(h * r))
        resized = cv2.resize(img_bgr, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        dw, dh = self.imgsz - new_w, self.imgsz - new_h
        left, top = dw // 2, dh // 2
        canvas = cv2.copyMakeBorder(resized, top, dh - top, left, dw - left,
                                    cv2.BORDER_CONSTANT, value=(PAD_VALUE,) * 3)
        rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)                 # BGR -> RGB
        x = torch.from_numpy(rgb).permute(2, 0, 1).float().div(255.0)  # HWC -> CHW, [0, 1]
        return x.unsqueeze(0), (r, left, top)

    @torch.no_grad()
    def infer(self, x):
        out = self.model(x)
        out = out[0] if isinstance(out, (list, tuple)) else out
        return out  # форма (1, 84, N): 4 координати xywh + 80 класових оцінок

    def postprocess(self, pred, meta, orig_shape):
        """Фільтр за класами і confidence, NMS, перехід до координат вихідного зображення."""
        r, left, top = meta
        p = pred[0].T                                   # (N, 84)
        boxes_xywh = p[:, :4]
        class_ids = torch.tensor(list(TARGET_CLASSES.keys()))
        scores = p[:, 4:][:, class_ids]                 # лише цільові класи
        best_score, best_j = scores.max(dim=1)
        keep = best_score > self.conf
        boxes_xywh, best_score = boxes_xywh[keep], best_score[keep]
        cls = class_ids[best_j[keep]]
        if boxes_xywh.shape[0] == 0:
            return []

        xyxy = torch.cat([boxes_xywh[:, :2] - boxes_xywh[:, 2:] / 2,
                          boxes_xywh[:, :2] + boxes_xywh[:, 2:] / 2], dim=1)
        keep_idx = torchvision.ops.batched_nms(xyxy, best_score, cls, self.iou)
        xyxy, best_score, cls = xyxy[keep_idx], best_score[keep_idx], cls[keep_idx]

        # з координат letterbox у координати оригінального зображення
        xyxy[:, [0, 2]] = (xyxy[:, [0, 2]] - left) / r
        xyxy[:, [1, 3]] = (xyxy[:, [1, 3]] - top) / r
        h, w = orig_shape[:2]
        xyxy[:, [0, 2]] = xyxy[:, [0, 2]].clamp(0, w)
        xyxy[:, [1, 3]] = xyxy[:, [1, 3]].clamp(0, h)

        return [(box.int().tolist(), float(s), TARGET_CLASSES[int(c)])
                for box, s, c in zip(xyxy, best_score, cls)]

    def detect(self, img_bgr):
        """Повний конвеєр для одного кадру. Повертає детекції та час етапів у мс."""
        t0 = time.perf_counter()
        x, meta = self.preprocess(img_bgr)
        t1 = time.perf_counter()
        pred = self.infer(x)
        t2 = time.perf_counter()
        dets = self.postprocess(pred, meta, img_bgr.shape)
        t3 = time.perf_counter()
        timing = {"pre_ms": (t1 - t0) * 1e3, "infer_ms": (t2 - t1) * 1e3,
                  "post_ms": (t3 - t2) * 1e3, "total_ms": (t3 - t0) * 1e3}
        return dets, timing


def draw(img, dets):
    for (x1, y1, x2, y2), score, label in dets:
        color = COLORS[label]
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        text = f"{label} {score:.2f}"
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(img, (x1, max(0, y1 - th - 6)), (x1 + tw + 4, y1), color, -1)
        cv2.putText(img, text, (x1 + 2, max(th, y1 - 4)), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (0, 0, 0), 2)
    return img


def run(args):
    detector = YoloDetector(args.weights, args.imgsz, args.conf, args.iou)
    src = args.source
    is_camera = src.isdigit()
    ext = os.path.splitext(src)[1].lower()

    if not is_camera and ext in (".jpg", ".jpeg", ".png", ".bmp"):
        img = cv2.imread(src)
        if img is None:
            raise SystemExit(f"Не вдалося прочитати зображення: {src}")
        dets, timing = detector.detect(img)
        print(f"Знайдено об'єктів: {len(dets)}")
        for box, score, label in dets:
            print(f"  {label:>5}  confidence {score:.2f}  box {box}")
        print(f"Час: inference {timing['infer_ms']:.1f} мс, повний {timing['total_ms']:.1f} мс")
        out = draw(img, dets)
        if args.save:
            os.makedirs(os.path.dirname(args.save) or ".", exist_ok=True)
            cv2.imwrite(args.save, out)
            print(f"Збережено: {args.save}")
        if not args.no_show:
            cv2.imshow("YOLOv8 + OpenCV", out)
            cv2.waitKey(0)
        cv2.destroyAllWindows()
        return

    # Відео або камера
    cap = cv2.VideoCapture(int(src) if is_camera else src)
    if not cap.isOpened():
        raise SystemExit(f"Не вдалося відкрити джерело: {src}")
    writer = None
    frames, total_t0 = 0, time.perf_counter()
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        dets, _ = detector.detect(frame)
        out = draw(frame, dets)
        frames += 1
        if args.save:
            if writer is None:
                os.makedirs(os.path.dirname(args.save) or ".", exist_ok=True)
                fps_src = cap.get(cv2.CAP_PROP_FPS) or 10
                h, w = out.shape[:2]
                writer = cv2.VideoWriter(args.save, cv2.VideoWriter_fourcc(*"mp4v"),
                                         fps_src, (w, h))
            writer.write(out)
        if not args.no_show:
            cv2.imshow("YOLOv8 + OpenCV", out)
            if cv2.waitKey(1) == 27:  # Esc — вихід
                break
    total = time.perf_counter() - total_t0
    cap.release()
    if writer is not None:
        writer.release()
    cv2.destroyAllWindows()
    if frames:
        print(f"Оброблено кадрів: {frames}, середній FPS (повний конвеєр): {frames / total:.1f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="YOLOv8 + OpenCV конвеєр")
    parser.add_argument("--source", required=True, help="зображення, відео або 0 для камери")
    parser.add_argument("--weights", default="yolov8n.pt")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.45)
    parser.add_argument("--save", default="", help="шлях для збереження результату")
    parser.add_argument("--no-show", action="store_true", help="не відкривати вікно")
    run(parser.parse_args())
