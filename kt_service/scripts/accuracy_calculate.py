import os
import numpy as np
import cv2
from ultralytics import YOLO
from tqdm import tqdm
from collections import defaultdict


class PixelLevelEvaluator:
    def __init__(self, model_path, images_dir, labels_dir, img_size=512):
        self.model = YOLO(model_path, task='segment')
        self.images_dir = images_dir
        self.labels_dir = labels_dir
        self.img_size = img_size
        self.class_names = {
            0: 'bone',
            1: 'muscles',
            2: 'lung',
            3: 'adipose'
        }

    def create_mask_from_yolo(self, label_path, img_width, img_height):
        """Создаем маску из YOLO разметки, где значение пикселя = class_id + 1

        Args:
            label_path (str): Путь к файлу с YOLO разметкой
            img_width (int): Ширина изображения
            img_height (int): Высота изображения

        Returns:
            numpy.ndarray: Одноканальная маска (H, W), где:
                - 0: фон (нет объекта)
                - 1: объект класса 0
                - 2: объект класса 1
                - и т.д.
        """
        mask = np.zeros((img_height, img_width), dtype=np.uint8)

        if not os.path.exists(label_path):
            return mask

        with open(label_path, 'r') as f:
            lines = f.readlines()

        for line in lines:
            parts = line.strip().split()
            if not parts:
                continue

            class_id = int(parts[0])
            coords = list(map(float, parts[1:]))

            # Конвертируем нормализованные координаты в абсолютные
            polygon = []
            for i in range(0, len(coords), 2):
                x = int(round(coords[i] * img_width))
                y = int(round(coords[i + 1] * img_height))
                polygon.append([x, y])

            # Рисуем полигон на маске с значением = class_id + 1
            if len(polygon) >= 3:
                cv2.fillPoly(
                    mask,
                    [np.array(polygon, dtype=np.int32)],
                    color=class_id + 1  # Используем class_id + 1 как значение пикселя
                )

        return mask

    def predict_mask(self, image_path):
        """Получаем предсказанную маску от модели с гарантированной поддержкой всех классов

        Args:
            image_path (str): Путь к входному изображению

        Returns:
            tuple: (pred_mask, img_width, img_height) где:
                pred_mask - объединенная маска всех объектов (numpy array)
                img_width - ширина изображения (int)
                img_height - высота изображения (int)
        """
        # Читаем изображение
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Не удалось загрузить изображение по пути: {image_path}")

        img_height, img_width = img.shape[:2]

        # Получаем предсказание модели
        result = self.model.predict(img, imgsz=self.img_size, conf=0.1, device=0, verbose=False, batch=16)[0]

        # Создаем пустую маску
        pred_mask = np.zeros((img_height, img_width), dtype=np.uint8)

        if result.masks is not None:
            # Обрабатываем каждую маску и класс
            for mask_tensor, cls in zip(result.masks.data, result.boxes.cls.cpu().numpy()):
                # Преобразуем тензор маски в numpy array и масштабируем к исходному размеру изображения
                mask = mask_tensor.cpu().numpy()
                mask = cv2.resize(mask, (img_width, img_height))

                # Преобразуем маску в бинарную (0 или 1)
                binary_mask = (mask > 0.5).astype(np.uint8)

                # Накладываем маску на итоговое изображение с учетом класса
                # Здесь можно добавить логику для разных классов, если нужно
                pred_mask = np.maximum(pred_mask, binary_mask * (int(cls) + 1))

        return pred_mask.astype(np.uint8), img_width, img_height

    def calculate_pixel_metrics(self, gt_mask, pred_mask):
        """Вычисляем попиксельные метрики для масок в формате class_id + 1

        Args:
            gt_mask (numpy.ndarray): Ground Truth маска (H, W), значения:
                                     - 0: фон
                                     - 1: класс 0
                                     - 2: класс 1
                                     - и т.д.
            pred_mask (numpy.ndarray): Предсказанная маска в том же формате

        Returns:
            dict: Метрики для каждого класса
        """
        metrics = {}
        total_pixels = gt_mask.shape[0] * gt_mask.shape[1]

        # Визуализация для отладки (можно раскомментировать)
        # cv2.imshow("GT Mask", (gt_mask * (255//len(self.class_names))).astype(np.uint8))
        # cv2.imshow("Pred Mask", (pred_mask * (255//len(self.class_names))).astype(np.uint8))
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

        for class_id, class_name in self.class_names.items():
            # Создаем бинарные маски для текущего класса
            gt_class_mask = (gt_mask == (class_id + 1)).astype(np.uint8)
            pred_class_mask = (pred_mask == (class_id + 1)).astype(np.uint8)

            # Вычисляем метрики
            tp = np.sum((gt_class_mask == 1) & (pred_class_mask == 1))  # True Positives
            fp = np.sum((gt_class_mask == 0) & (pred_class_mask == 1))  # False Positives
            fn = np.sum((gt_class_mask == 1) & (pred_class_mask == 0))  # False Negatives
            tn = np.sum((gt_class_mask == 0) & (pred_class_mask == 0))  # True Negatives

            # Дополнительные метрики
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            accuracy = (tp + tn) / total_pixels if total_pixels > 0 else 0
            iou = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0

            metrics[class_id] = {
                'class_name': class_name,
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1_score': f1_score,
                'iou': iou,
                'tp': tp,
                'fp': fp,
                'fn': fn,
                'tn': tn,
                'total_pixels': np.sum(gt_class_mask)  # Общее количество пикселей класса в GT
            }

        return metrics

    def evaluate(self):
        """Оценка модели на всем датасете"""
        class_metrics = defaultdict(lambda: {
            'accuracy': 0,
            'tp': 0,
            'fp': 0,
            'fn': 0,
            'tn': 0,
            'total_pixels': 0,
            'count': 0
        })

        image_files = [f for f in os.listdir(self.images_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

        for img_file in tqdm(image_files, desc="Evaluating images"):
            base_name = os.path.splitext(img_file)[0]
            image_path = os.path.join(self.images_dir, img_file)
            label_path = os.path.join(self.labels_dir, f"{base_name}.txt")

            # Получаем маски
            pred_mask, img_width, img_height = self.predict_mask(image_path)
            gt_mask = self.create_mask_from_yolo(label_path, img_width, img_height)

            # Вычисляем метрики
            metrics = self.calculate_pixel_metrics(gt_mask, pred_mask)

            # Агрегируем результаты
            for class_id in metrics:
                for key in ['accuracy', 'tp', 'fp', 'fn', 'tn', 'total_pixels']:
                    class_metrics[class_id][key] += metrics[class_id][key]
                class_metrics[class_id]['count'] += 1

        # Вычисляем средние значения
        results = {}
        for class_id in class_metrics:
            m = class_metrics[class_id]
            count = m['count']

            if count == 0:
                continue

            results[class_id] = {
                'accuracy': m['accuracy'] / count,
                'tp_rate': m['tp'] / m['total_pixels'] if m['total_pixels'] > 0 else 0,
                'fn_rate': m['fn'] / m['total_pixels'] if m['total_pixels'] > 0 else 0,
                'fp_rate': m['fp'] / (self.img_size * self.img_size * count)  # Относительно всех пикселей
            }

        return results

    def print_results(self, results):
        """Вывод результатов в понятном формате"""
        print("\n=== Pixel-Level Evaluation Results ===")
        print(f"{'Class':<10} {'Accuracy':<10} {'TP Rate':<10} {'FN Rate':<10} {'FP Rate':<10}")
        print("-" * 50)

        for class_id in sorted(results.keys()):
            r = results[class_id]
            print(f"{self.class_names[class_id]:<10} "
                  f"{r['accuracy']:.2%}      "
                  f"{r['tp_rate']:.2%}      "
                  f"{r['fn_rate']:.2%}      "
                  f"{r['fp_rate']:.2%}")

        # Вычисляем общие средние
        if results:
            avg_accuracy = np.mean([r['accuracy'] for r in results.values()])
            avg_tp = np.mean([r['tp_rate'] for r in results.values()])
            avg_fn = np.mean([r['fn_rate'] for r in results.values()])

            print("\n=== Summary ===")
            print(f"Average Accuracy: {avg_accuracy:.2%}")
            print(f"Average True Positive Rate: {avg_tp:.2%}")
            print(f"Average False Negative Rate: {avg_fn:.2%}")


if __name__ == "__main__":
    # Конфигурация
    MODEL_PATH = "yolov11s_axial_16_04_100ep_16batch_512_best.pt"
    IMAGES_DIR = "test/images"
    LABELS_DIR = "test/labels"
    IMG_SIZE = 512

    # Инициализация и запуск оценки
    evaluator = PixelLevelEvaluator(MODEL_PATH, IMAGES_DIR, LABELS_DIR, IMG_SIZE)
    results = evaluator.evaluate()
    evaluator.print_results(results)
