import os
import numpy as np
from shapely.geometry import Polygon, box
from collections import defaultdict


def read_yolo_polygons(label_path, img_width=1024, img_height=1024):
    """Чтение YOLO полигонов из файла и преобразование в координаты изображения"""
    polygons = []
    if not os.path.exists(label_path):
        return polygons

    with open(label_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            class_id = int(parts[0])
            points = list(map(float, parts[1:]))
            # Преобразование YOLO формата в абсолютные координаты
            absolute_points = []
            for i in range(0, len(points), 2):
                x = points[i] * img_width
                y = points[i + 1] * img_height
                absolute_points.append((x, y))
            polygons.append((class_id, absolute_points))
    return polygons


def calculate_iou(poly1, poly2):
    """Вычисление Intersection over Union для двух полигонов"""
    try:
        shapely_poly1 = Polygon(poly1)
        shapely_poly2 = Polygon(poly2)

        if not shapely_poly1.is_valid or not shapely_poly2.is_valid:
            return 0.0

        intersection = shapely_poly1.intersection(shapely_poly2).area
        union = shapely_poly1.union(shapely_poly2).area
        return intersection / union if union > 0 else 0.0
    except:
        return 0.0


def evaluate_segmentation(gt_dir, pred_dir, iou_threshold=0.5):
    """Оценка точности сегментации"""
    results = defaultdict(lambda: {'tp': 0, 'fp': 0, 'fn': 0})

    for gt_file in os.listdir(gt_dir):
        if not gt_file.endswith('.txt'):
            continue

        base_name = os.path.splitext(gt_file)[0]
        pred_file = os.path.join(pred_dir, gt_file)

        # Чтение ground truth и предсказанных полигонов
        gt_polygons = read_yolo_polygons(os.path.join(gt_dir, gt_file))
        pred_polygons = read_yolo_polygons(pred_file)

        # Для каждого класса
        for class_id in set([p[0] for p in gt_polygons] + [p[0] for p in pred_polygons]):
            gt_class = [p for p in gt_polygons if p[0] == class_id]
            pred_class = [p for p in pred_polygons if p[0] == class_id]

            # Сопоставление предсказаний с ground truth
            matched_gt = set()
            matched_pred = set()

            # Ищем наилучшие соответствия
            for i, gt in enumerate(gt_class):
                for j, pred in enumerate(pred_class):
                    if j in matched_pred:
                        continue
                    iou = calculate_iou(gt[1], pred[1])
                    if iou >= iou_threshold:
                        matched_gt.add(i)
                        matched_pred.add(j)
                        results[class_id]['tp'] += 1
                        break

            # Несопоставленные ground truth - это False Negatives
            results[class_id]['fn'] += len(gt_class) - len(matched_gt)

            # Несопоставленные предсказания - это False Positives
            results[class_id]['fp'] += len(pred_class) - len(matched_pred)

    # Вычисление метрик для каждого класса
    metrics = {}
    for class_id in results:
        tp = results[class_id]['tp']
        fp = results[class_id]['fp']
        fn = results[class_id]['fn']

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        accuracy = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0

        metrics[class_id] = {
            'TP': tp,
            'FP': fp,
            'FN': fn,
            'Precision': precision,
            'Recall': recall,
            'Accuracy': accuracy
        }

    return metrics


if __name__ == "__main__":
    # Пути к директориям с ground truth и предсказаниями
    gt_labels_dir = 'labels'  # Папка с размеченными данными
    pred_labels_dir = 'predictions'  # Папка с предсказаниями модели

    # Оценка точности
    metrics = evaluate_segmentation(gt_labels_dir, pred_labels_dir)

    # Вывод результатов
    print("{:<10} {:<10} {:<10} {:<10} {:<12} {:<12} {:<12}".format(
        'Class', 'TP', 'FP', 'FN', 'Precision', 'Recall', 'Accuracy'))

    for class_id in sorted(metrics.keys()):
        m = metrics[class_id]
        print("{:<10} {:<10} {:<10} {:<10} {:<12.4f} {:<12.4f} {:<12.4f}".format(
            class_id, m['TP'], m['FP'], m['FN'],
            m['Precision'], m['Recall'], m['Accuracy']))
