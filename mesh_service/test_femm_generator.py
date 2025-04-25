import unittest
import os
import tempfile
import numpy as np
from unittest.mock import patch, MagicMock
from femm_generator import (
    divide_triangles_into_groups,
    export_mesh_for_femm,
    get_image,
    largest_segment_area_index,
    merge_collinear_segments,
    point_line_distance
)


class TestFemmGenerator(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Создаем тестовые данные"""
        cls.temp_dir = tempfile.mkdtemp()
        cls.test_output_path = os.path.join(cls.temp_dir, 'test_mesh.txt')

        # Тестовые контуры
        cls.test_contours = [
            [0, 0, 0, 10, 0, 10, 10, 0, 10],  # Класс 0
            [1, 2, 2, 8, 2, 8, 8, 2, 8]  # Класс 1
        ]

    def test_largest_segment_area_index(self):
        """Тест поиска контура с максимальной площадью"""
        polygons = [
            "0 0 0 10 0 10 10 0 10",  # Площадь 100
            "1 0 0 5 0 5 5 0 5"  # Площадь 25
        ]
        result = largest_segment_area_index(polygons)
        self.assertEqual(result, 0)

    def test_merge_collinear_segments(self):
        """Тест объединения коллинеарных сегментов"""
        contour = [0, 0, 0, 5, 0, 10, 0, 10, 5, 10, 10]  # Точка (5,0) коллинеарна
        result = merge_collinear_segments(contour, distance_threshold=1.0)
        self.assertEqual(len(result), 8)  # Должна удалиться одна точка

    def test_point_line_distance(self):
        """Тест расчета расстояния от точки до линии"""
        distance = point_line_distance(5, 5, 0, 0, 10, 10)
        self.assertAlmostEqual(distance, 0.0, places=6)

    @patch('femm_generator.gmsh.model.mesh.getElements')
    @patch('femm_generator.gmsh.model.mesh.getNodes')
    def test_divide_triangles_into_groups(self, mock_nodes, mock_elements):
        """Тест разделения треугольников по группам"""
        # Настраиваем моки для Gmsh
        mock_nodes.return_value = (
            [1, 2, 3],
            [0, 0, 0, 10, 0, 0, 10, 10, 0, 0, 10, 0],
            []
        )
        mock_elements.return_value = (
            [2],  # Тип элемента (треугольник)
            [[1]],  # Теги элементов
            [[1, 2, 3]]  # Теги узлов
        )

        # Запускаем тестируемую функцию
        groups = divide_triangles_into_groups(self.test_contours, outer_contour_class=0)

        # Проверяем результаты
        self.assertIn(0, groups)
        self.assertIn(1, groups)
        self.assertEqual(len(groups[0]) + len(groups[1]), 1)

    @patch('femm_generator.gmsh.model.mesh.getNodes')
    @patch('femm_generator.gmsh.model.mesh.getElements')
    def test_export_mesh_for_femm(self, mock_elements, mock_nodes):
        """Тест экспорта mesh для FEMM"""
        # Настраиваем моки
        mock_nodes.return_value = (
            [1, 2, 3],
            [0, 0, 0, 10, 0, 0, 10, 10, 0],
            []
        )
        mock_elements.return_value = (
            [2],  # Тип элемента (треугольник)
            [[1]],  # Теги элементов
            [[1, 2, 3]]  # Теги узлов
        )

        # Тестовые группы
        class_groups = {0: [1], 1: []}

        # Запускаем тестируемую функцию
        export_mesh_for_femm(self.test_output_path, class_groups)

        # Проверяем что файл создан
        self.assertTrue(os.path.exists(self.test_output_path))

    @patch('femm_generator.gmsh.model.mesh.getNodes')
    @patch('femm_generator.gmsh.model.mesh.getElement')
    def test_get_image(self, mock_element, mock_nodes):
        """Тест генерации изображения mesh"""
        # Настраиваем моки
        mock_nodes.return_value = (
            [1, 2, 3],
            [0, 0, 0, 10, 0, 0, 10, 10, 0],
            []
        )
        mock_element.return_value = (
            2,  # Тип элемента
            [1, 2, 3],  # Теги узлов
            [], []
        )

        # Тестовые группы
        class_groups = {0: [1], 1: []}

        # Запускаем тестируемую функцию
        img = get_image(class_groups, image_size=(500, 500))

        # Проверяем результаты
        self.assertEqual(img.shape, (500, 500, 3))
        self.assertEqual(img.dtype, np.uint8)

    @classmethod
    def tearDownClass(cls):
        """Удаляем временную директорию"""
        import shutil
        shutil.rmtree(cls.temp_dir)


if __name__ == '__main__':
    unittest.main()