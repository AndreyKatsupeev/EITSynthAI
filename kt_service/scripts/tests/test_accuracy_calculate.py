import unittest
import os
import shutil
from kt_service.scripts.accuracy_calculate import read_yolo_polygons, calculate_iou, evaluate_segmentation


class TestAccuracyCalculate(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Создаем тестовые директории и файлы"""
        cls.gt_dir = 'test_gt_labels'
        cls.pred_dir = 'test_pred_labels'
        os.makedirs(cls.gt_dir, exist_ok=True)
        os.makedirs(cls.pred_dir, exist_ok=True)

        # Создаем тестовые YOLO файлы
        with open(os.path.join(cls.gt_dir, 'test1.txt'), 'w') as f:
            f.write('0 0.1 0.1 0.2 0.1 0.2 0.2 0.1 0.2\n')
            f.write('1 0.5 0.5 0.6 0.5 0.6 0.6 0.5 0.6\n')

        with open(os.path.join(cls.pred_dir, 'test1.txt'), 'w') as f:
            f.write('0 0.1 0.1 0.2 0.1 0.2 0.2 0.1 0.2\n')  # полное совпадение с GT
            f.write('1 0.55 0.55 0.65 0.55 0.65 0.65 0.55 0.65\n')  # частичное совпадение

    @classmethod
    def tearDownClass(cls):
        """Удаляем тестовые директории"""
        shutil.rmtree(cls.gt_dir)
        shutil.rmtree(cls.pred_dir)

    def test_read_yolo_polygons(self):
        """Тест чтения YOLO полигонов"""
        polygons = read_yolo_polygons(os.path.join(self.gt_dir, 'test1.txt'))
        self.assertEqual(len(polygons), 2)
        self.assertEqual(polygons[0][0], 0)  # class_id
        self.assertEqual(len(polygons[0][1]), 4)  # 4 точки

    def test_calculate_iou(self):
        """Тест расчета IoU"""
        poly1 = [(10, 10), (20, 10), (20, 20), (10, 20)]
        poly2 = [(15, 15), (25, 15), (25, 25), (15, 25)]
        iou = calculate_iou(poly1, poly2)
        self.assertGreaterEqual(iou, 0)
        self.assertLessEqual(iou, 1)

        # Проверка полного совпадения
        iou_same = calculate_iou(poly1, poly1)
        self.assertAlmostEqual(iou_same, 1.0)

    def test_evaluate_segmentation(self):
        """Тест оценки сегментации"""
        metrics = evaluate_segmentation(self.gt_dir, self.pred_dir, iou_threshold=0.5)

        # Проверяем, что метрики рассчитаны для обоих классов
        self.assertIn(0, metrics)
        self.assertIn(1, metrics)

        # Класс 0 должен иметь TP=1, FP=0, FN=0 (полное совпадение)
        self.assertEqual(metrics[0]['TP'], 1)
        self.assertEqual(metrics[0]['FP'], 0)
        self.assertEqual(metrics[0]['FN'], 0)

        # Проверяем, что precision, recall, accuracy для класса 0 равны 1
        self.assertAlmostEqual(metrics[0]['Precision'], 1.0)
        self.assertAlmostEqual(metrics[0]['Recall'], 1.0)
        self.assertAlmostEqual(metrics[0]['Accuracy'], 1.0)


if __name__ == '__main__':
    unittest.main()
