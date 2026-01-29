import unittest
import numpy as np
from EITSynthAI.kt_service.ai_tools.femm_tools.model_generator import (
    load_yolo,
    load_mesh,
    check_mesh_nodes,
    prepare_mesh,
    create_pyeit_model,
    add_skin,
    insert_electordes_to_polygone,
    Settings,
    classes_list
)


class TestModelGenerator(unittest.TestCase):

    def setUp(self):
        self.sample_yolo_path = '/EITSynthAI/kt_service/ai_tools/femm_tools/data/test_data.txt'
        self.sample_mesh_path = '/EITSynthAI/kt_service/ai_tools/femm_tools/data/tmp.txt'
        self.settings = Settings(Nelec=16, Relec=10, accuracy=0.5,
                                 min_area=100, polydeg=5, skinthick=1,
                                 I=0.005, Freq=50000, thin_coeff=5)

    def test_load_yolo(self):
        borders = load_yolo(self.sample_yolo_path, classes_list)
        self.assertIsInstance(borders, dict)
        self.assertTrue(len(borders) > 0)

    def test_load_mesh(self):
        mesh = load_mesh(self.sample_mesh_path, classes_list)
        self.assertIsInstance(mesh, dict)
        self.assertIn('element', mesh)
        self.assertIn('node', mesh)
        self.assertIn('cond', mesh)
        self.assertIn('classes_gr', mesh)

    def test_check_mesh_nodes(self):
        mesh = load_mesh(self.sample_mesh_path, classes_list)
        checked_mesh = check_mesh_nodes(mesh)
        self.assertIsInstance(checked_mesh, dict)
        self.assertIn('element', checked_mesh)
        self.assertIn('node', checked_mesh)

    def test_prepare_mesh(self):
        mesh = prepare_mesh(self.sample_mesh_path, classes_list)
        self.assertIsInstance(mesh, dict)
        self.assertIn('element', mesh)
        self.assertIn('node', mesh)

    def test_create_pyeit_model(self):
        mesh = prepare_mesh(self.sample_mesh_path, classes_list)
        mesh_obj = create_pyeit_model(mesh, self.settings.Nelec)
        self.assertIsNotNone(mesh_obj)

    def test_add_skin(self):
        data = np.array([[0, 0], [1, 0], [1, 1], [0, 1]])
        skin = add_skin(data, 0.1)
        self.assertIsInstance(skin, np.ndarray)
        self.assertEqual(skin.shape[1], 2)

    def test_insert_electrodes_to_polygone(self):
        polygone = np.array([[0, 0], [1, 0], [1, 1], [0, 1]])
        elecs = np.array([[[0.25, 0], [0.75, 0], [0.5, 0.1]],
                          [[0.25, 1], [0.75, 1], [0.5, 0.9]]])
        result = insert_electordes_to_polygone(polygone, elecs)
        self.assertIsInstance(result, np.ndarray)
        self.assertGreater(result.shape[0], polygone.shape[0])


if __name__ == '__main__':
    unittest.main()
