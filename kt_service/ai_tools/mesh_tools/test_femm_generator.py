import unittest
import numpy as np
import tempfile
import os
from EITSynthAI.kt_service.ai_tools.mesh_tools.femm_generator import (
    largest_segment_area_index,
    merge_collinear_segments,
    point_line_distance,
    create_mesh
)


class TestFemmGenerator(unittest.TestCase):
    def setUp(self):
        self.test_polygons = [
            '0 0 0 0 10 10 10 10 0',  # квадрат
            '1 2 2 2 8 8 8 8 2',  # внутренний квадрат
            '2 20 20 20 30 30 30 30 20'  # другой квадрат
        ]

    def test_largest_segment_area_index(self):
        # Первый полигон (квадрат 10x10) должен иметь наибольшую площадь
        result = largest_segment_area_index(self.test_polygons)
        self.assertEqual(result, 0)

        # Добавляем полигон с большей площадью
        large_poly = ['3 0 0 0 20 20 20 20 0']
        result = largest_segment_area_index(large_poly + self.test_polygons)
        self.assertEqual(result, 0)

    def test_merge_collinear_segments(self):
        # Почти коллинеарные точки
        contour = [0, 0, 1, 0.1, 2, 0.2, 3, 0, 0, 0]
        merged = merge_collinear_segments(contour, distance_threshold=0.5)
        # Должны остаться только угловые точки
        self.assertEqual(len(merged), 6)  # [0,0, 3,0, 0,0]

    def test_point_line_distance(self):
        # Точка на линии
        dist = point_line_distance(5, 5, 0, 0, 10, 10)
        self.assertAlmostEqual(dist, 0)

        # Точка на расстоянии от линии
        dist = point_line_distance(0, 5, 0, 0, 10, 0)
        self.assertAlmostEqual(dist, 5)

    def test_create_mesh(self):
        # Проверяем что функция выполняется без ошибок
        img = create_mesh(
            pixel_spacing=[0.682, 0.682],
            polygons=self.test_polygons,
            show_meshing_result_method="opencv",
            # is_exporting_to_femm=False
        )
        self.assertIsInstance(img, np.ndarray)
        self.assertEqual(img.ndim, 3)  # RGB изображение

        # Проверяем экспорт в файл
        with tempfile.NamedTemporaryFile(suffix='.txt', delete=False) as tmp:
            try:
                create_mesh(
                    pixel_spacing=[0.682, 0.682],
                    polygons=self.test_polygons,
                    show_meshing_result_method="no",
                    is_exporting_to_femm=True,
                    export_filename=tmp.name
                )
                self.assertTrue(os.path.exists(tmp.name))
                self.assertGreater(os.path.getsize(tmp.name), 0)
            finally:
                os.unlink(tmp.name)


if __name__ == '__main__':
    unittest.main()
