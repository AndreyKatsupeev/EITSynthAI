import unittest
from unittest.mock import patch, MagicMock
from kt_service.ai_tools.femm_tools.synthetic_datasets_generator import (
    load_yolo,
    femm_create_problem,
    femm_add_contour
)


class TestSyntheticDatasetsGenerator(unittest.TestCase):
    def test_load_yolo(self):
        """Тест загрузки YOLO разметки"""
        # Создаем временный файл с тестовыми данными
        import tempfile
        with tempfile.NamedTemporaryFile(mode='w+') as tmp:
            tmp.write("0 0 0 10 0 10 10 0 10\n")
            tmp.write("1 2 2 8 2 8 8 2 8\n")
            tmp.seek(0)

            # Загружаем данные
            borders = load_yolo(tmp.name)

            # Проверяем результаты
            self.assertIn('0', borders)
            self.assertIn('1', borders)
            self.assertEqual(len(borders['0'][0][0]), 4)  # 4 точки x
            self.assertEqual(len(borders['0'][0][1]), 4)  # 4 точки y

    def test_load_yolo_invalid_data(self):
        """Тест загрузки некорректных данных"""
        with self.assertRaises(ValueError):
            # Создаем файл с непарными координатами
            import tempfile
            with tempfile.NamedTemporaryFile(mode='w+') as tmp:
                tmp.write("0 0 0 10 0 10\n")  # Нечетное количество координат
                tmp.seek(0)
                load_yolo(tmp.name)

    @patch('EITSynthAI.mesh_service.synthetic_datasets_generator.femm.openfemm')
    @patch('EITSynthAI.mesh_service.synthetic_datasets_generator.femm.newdocument')
    @patch('EITSynthAI.mesh_service.synthetic_datasets_generator.femm.ci_probdef')
    def test_femm_create_problem(self, mock_probdef, mock_newdoc, mock_open):
        """Тест создания задачи в FEMM"""
        femm_create_problem()

        # Проверяем вызовы FEMM API
        mock_open.assert_called_once()
        mock_newdoc.assert_called_once_with(3)
        mock_probdef.assert_called_once_with('millimeters', 'planar', 50000, 1e-8, 10)

    @patch('EITSynthAI.mesh_service.synthetic_datasets_generator.femm.ci_addnode')
    @patch('EITSynthAI.mesh_service.synthetic_datasets_generator.femm.ci_addsegment')
    def test_femm_add_contour(self, mock_segment, mock_node):
        """Тест добавления контура в FEMM"""
        test_coords = [[0, 5, 10], [0, 5, 0]]  # x и y координаты

        femm_add_contour(test_coords)

        # Проверяем что узлы и сегменты были добавлены
        self.assertEqual(mock_node.call_count, 3)
        self.assertEqual(mock_segment.call_count, 4)


if __name__ == '__main__':
    unittest.main()
