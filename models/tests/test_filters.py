import unittest
import numpy as np
from EITSynthAI.models.filters import (
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
    def test_calc_lin_coef_normal(self):
        point1 = [1, 2]
        point2 = [3, 4]
        k, b = calc_lin_coef(point1, point2)
        self.assertAlmostEqual(k, 1.0)
        self.assertAlmostEqual(b, 1.0)

    def test_calc_lin_coef_vertical(self):
        point1 = [1, 2]
        point2 = [1, 4]
        with self.assertRaises(ValueError):
            calc_lin_coef(point1, point2)

    def test_calc_dist_euclidean(self):
        point1 = [1, 2]
        point2 = [4, 6]
        self.assertAlmostEqual(calc_dist(point1, point2), 5.0)

    def test_calc_dist_max_coord_dif(self):
        point1 = np.array([1, 2])
        point2 = np.array([4, 6])
        self.assertAlmostEqual(calc_dist(point1, point2, typ='max_coord_dif'), 4)

    def test_calc_dist_unknown_method(self):
        point1 = [1, 2]
        point2 = [4, 6]
        with self.assertRaises(ValueError):
            calc_dist(point1, point2, typ='unknown')

    def test_check_point_in_line_true(self):
        filtered_data = np.array([[1, 1], [2, 2]])
        point = [1.5, 1.5]
        self.assertTrue(check_point_in_line(filtered_data, point, 0.1))

    def test_check_point_in_line_false(self):
        filtered_data = np.array([[1, 1], [2, 2]])
        point = [1.5, 3.0]
        self.assertFalse(check_point_in_line(filtered_data, point, 0.1))

    def test_check_point_in_line_vertical(self):
        filtered_data = np.array([[1, 1], [1, 2]])
        point = [1, 1.5]
        self.assertTrue(check_point_in_line(filtered_data, point, 0.1))

    def test_filter_degr_polyfit_normal(self):
        data = np.array([[1, 1], [2, 2], [3, 3], [4, 5], [5, 6]])
        result = filter_degr_polyfit(data, 10, 2)
        self.assertGreaterEqual(result.shape[0], 2)

    def test_filter_degr_polyfit_empty(self):
        data = np.array([[1, 1], [2, 2]])
        result = filter_degr_polyfit(data, 10, 2)
        self.assertEqual(result.shape[0], 4)

    def test_filter_inline_points_normal(self):
        data = np.array([[1, 1], [2, 2], [3, 3], [4, 4]])
        result = filter_inline_points(data, accuracy=0.1)
        self.assertLessEqual(result.shape[0], 4)

    def test_filter_inline_points_empty(self):
        data = np.array([[1, 1], [1, 1]])
        result = filter_inline_points(data, accuracy=0.1)
        self.assertEqual(result.shape[0], 1)

    def test_PolyArea_square(self):
        x = np.array([0, 1, 1, 0])
        y = np.array([0, 0, 1, 1])
        self.assertAlmostEqual(PolyArea(x, y), 1.0)

    def test_cut_min_area_close_points_normal(self):
        data = np.array([[0, 0], [1, 0], [1, 1], [0, 1]])
        result = сut_min_area_close_points(data, 0.5, 0.1)
        self.assertEqual(result.shape[0], 4)

    def test_cut_min_area_close_points_empty(self):
        data = np.array([[0, 0], [0.1, 0.1]])
        result = сut_min_area_close_points(data, 0.5, 0.1)
        self.assertEqual(result.shape[0], 2)

    def test_interpolate_big_vert_breaks_lin_normal(self):
        data = np.array([[0, 0], [1, 1], [2, 10]])
        result = interpolate_big_vert_breaks_lin(data, 1)
        self.assertGreater(result.shape[0], 2)

    def test_interpolate_big_vert_breaks_poly_normal(self):
        data = np.array([[0, 0], [1, 1], [2, 10]])
        result = interpolate_big_vert_breaks_poly(data, 2, 1)
        self.assertGreater(result.shape[0], 2)


if __name__ == '__main__':
    unittest.main()
