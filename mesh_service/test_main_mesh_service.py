import unittest
from fastapi.testclient import TestClient
from unittest.mock import patch
from main_mesh_service import app, MeshData
import numpy as np
import cv2


class TestMainMeshService(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(app)

    @patch('main_mesh_service.create_mesh')
    def test_create_mesh_endpoint(self, mock_create_mesh):
        """Тест эндпоинта /createMesh"""
        # Настраиваем мок
        test_image = np.zeros((100, 100, 3), dtype=np.uint8)
        mock_create_mesh.return_value = test_image

        # Тестовые данные
        test_data = {
            "params": [0.682, 0.682],
            "polygons": [
                "0 0 0 10 0 10 10 0 10",
                "1 2 2 8 2 8 8 2 8"
            ]
        }

        # Отправляем запрос
        response = self.client.post("/createMesh", json=test_data)

        # Проверяем результаты
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.headers["content-type"], "image/png")

        # Проверяем что вернулось изображение
        img_array = np.frombuffer(response.content, dtype=np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        self.assertEqual(img.shape, test_image.shape)

    def test_mesh_data_validation(self):
        """Тест валидации входных данных"""
        # Неверные данные - недостаточно параметров
        invalid_data = {
            "params": [0.682],
            "polygons": ["0 0 0 10 0"]
        }

        response = self.client.post("/createMesh", json=invalid_data)
        self.assertEqual(response.status_code, 422)  # Ошибка валидации


if __name__ == '__main__':
    unittest.main()