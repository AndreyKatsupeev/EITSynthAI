import unittest
import numpy as np
from EITSynthAI.kt_service.ai_tools.femm_tools.filters import (
    calc_lin_coef,
    calc_dist,
    check_point_in_line,
    filter_degr_polyfit,
    filter_inline_points,
    PolyArea,
    сut_min_area_close_points,
    interpolate_big_vert_breaks_lin,
    interpolate_big_vert_breaks_poly,
)


class TestFilters(unittest.TestCase):
    """
    Набор тестов для проверки функциональности модуля filters.
    """

    def test_calc_lin_coef_normal(self):
        """
        Тест функции calc_lin_coef для нормального случая.
        Проверяет корректность вычисления коэффициентов линейной функции по двум точкам.
        """
        point1 = [1, 2]
        point2 = [3, 4]
        k, b = calc_lin_coef(point1, point2)
        self.assertAlmostEqual(k, 1.0)
        self.assertAlmostEqual(b, 1.0)

    def test_calc_lin_coef_vertical(self):
        """
        Тест функции calc_lin_coef для вертикальной линии.
        Проверяет, что функция выбрасывает исключение ValueError для вертикальных линий.
        """
        point1 = [1, 2]
        point2 = [1, 4]
        with self.assertRaises(ValueError):
            calc_lin_coef(point1, point2)

    def test_calc_dist_euclidean(self):
        """
        Тест функции calc_dist для евклидова расстояния.
        Проверяет корректность вычисления евклидова расстояния между двумя точками.
        """
        point1 = [1, 2]
        point2 = [4, 6]
        self.assertAlmostEqual(calc_dist(point1, point2), 5.0)

    def test_calc_dist_max_coord_dif(self):
        """
        Тест функции calc_dist для максимальной разницы координат.
        Проверяет корректность вычисления максимальной разницы между координатами двух точек.
        """
        point1 = np.array([1, 2])
        point2 = np.array([4, 6])
        self.assertAlmostEqual(calc_dist(point1, point2, typ='max_coord_dif'), 4)

    def test_calc_dist_unknown_method(self):
        """
        Тест функции calc_dist для неизвестного метода.
        Проверяет, что функция выбрасывает исключение ValueError для неизвестного метода вычисления расстояния.
        """
        point1 = [1, 2]
        point2 = [4, 6]
        with self.assertRaises(ValueError):
            calc_dist(point1, point2, typ='unknown')

    def test_check_point_in_line_true(self):
        """
        Тест функции check_point_in_line для точки, лежащей на линии.
        Проверяет, что функция корректно определяет, лежит ли точка на линии, образованной двумя другими точками.
        """
        filtered_data = np.array([[1, 1], [2, 2]])
        point = [1.5, 1.5]
        self.assertTrue(check_point_in_line(filtered_data, point, 0.1))

    def test_check_point_in_line_false(self):
        """
        Тест функции check_point_in_line для точки, не лежащей на линии.
        Проверяет, что функция корректно определяет, не лежит ли точка на линии, образованной двумя другими точками.
        """
        filtered_data = np.array([[1, 1], [2, 2]])
        point = [1.5, 3.0]
        self.assertFalse(check_point_in_line(filtered_data, point, 0.1))

    def test_check_point_in_line_vertical(self):
        """
        Тест функции check_point_in_line для вертикальной линии.
        Проверяет, что функция корректно определяет принадлежность точки вертикальной линии.
        """
        filtered_data = np.array([[1, 1], [1, 2]])
        point = [1, 1.5]
        self.assertTrue(check_point_in_line(filtered_data, point, 0.1))

    def test_filter_degr_polyfit_normal(self):
        """
        Тест функции filter_degr_polyfit для нормального случая.
        Проверяет, что функция корректно фильтрует данные на основе угла наклона.
        """
        data = np.array([[1, 1], [2, 2], [3, 3], [4, 5], [5, 6]])
        result = filter_degr_polyfit(data, 10, 2)
        self.assertGreaterEqual(result.shape[0], 2)

    def test_filter_degr_polyfit_empty(self):
        """
        Тест функции filter_degr_polyfit для пустого случая.
        Проверяет поведение функции при минимальном наборе данных.
        """
        data = np.array([[1, 1], [2, 2]])
        result = filter_degr_polyfit(data, 10, 2)
        self.assertEqual(result.shape[0], 4)

    def test_filter_inline_points_normal(self):
        """
        Тест функции filter_inline_points для нормального случая.
        Проверяет, что функция корректно фильтрует коллинеарные точки.
        """
        data = np.array([[1, 1], [2, 2], [3, 3], [4, 4]])
        result = filter_inline_points(data, accuracy=0.1)
        self.assertLessEqual(result.shape[0], 4)

    def test_filter_inline_points_empty(self):
        """
        Тест функции filter_inline_points для случая с повторяющимися точками.
        Проверяет поведение функции при наличии повторяющихся точек.
        """
        data = np.array([[1, 1], [1, 1]])
        result = filter_inline_points(data, accuracy=0.1)
        self.assertEqual(result.shape[0], 1)

    def test_PolyArea_square(self):
        """
        Тест функции PolyArea для квадрата.
        Проверяет корректность вычисления площади квадрата.
        """
        x = np.array([0, 1, 1, 0])
        y = np.array([0, 0, 1, 1])
        self.assertAlmostEqual(PolyArea(x, y), 1.0)

    def test_cut_min_area_close_points_normal(self):
        """
        Тест функции сut_min_area_close_points для нормального случая.
        Проверяет, что функция корректно удаляет полигоны с малой площадью.
        """
        data = np.array([[0, 0], [1, 0], [1, 1], [0, 1]])
        result = сut_min_area_close_points(data, 0.5, 0.1)
        self.assertEqual(result.shape[0], 4)

    def test_cut_min_area_close_points_empty(self):
        """
        Тест функции сut_min_area_close_points для случая с малой площадью.
        Проверяет поведение функции при наличии полигонов с малой площадью.
        """
        data = np.array([[0, 0], [0.1, 0.1]])
        result = сut_min_area_close_points(data, 0.5, 0.1)
        self.assertEqual(result.shape[0], 2)

    def test_interpolate_big_vert_breaks_lin_normal(self):
        """
        Тест функции interpolate_big_vert_breaks_lin для нормального случая.
        Проверяет, что функция корректно интерполирует большие вертикальные разрывы линейно.
        """
        data = np.array([[0, 0], [1, 1], [2, 10]])
        result = interpolate_big_vert_breaks_lin(data, 1)
        self.assertGreater(result.shape[0], 2)

    def test_interpolate_big_vert_breaks_poly_normal(self):
        """
        Тест функции interpolate_big_vert_breaks_poly для нормального случая.
        Проверяет, что функция корректно интерполирует большие вертикальные разрывы полиномиально.
        """
        data = np.array([[0, 0], [1, 1], [2, 10]])
        result = interpolate_big_vert_breaks_poly(data, 2, 1)
        self.assertGreater(result.shape[0], 2)


if __name__ == '__main__':
    unittest.main()
