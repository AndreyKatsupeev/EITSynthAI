import unittest
from unittest.mock import patch
from fastapi.testclient import TestClient
from EITSynthAI.kt_service.ai_tools.mesh_tools.main_mesh_service import app, MeshData


class TestCreateMeshFromJson(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(app)

    @patch('EITSynthAI.kt_service.ai_tools.mesh_tools.main_mesh_service.create_mesh')
    def test_create_mesh_from_json_success(self, mock_create_mesh):
        # Подготовка тестовых данных
        test_data = {
            "params": [0.682, 0.682],
            "polygons": ["10,10,20,20", "30,30,40,40"]
        }

        # Настройка мока для возврата тестового изображения
        mock_create_mesh.return_value = bytearray([0 for _ in range(100)])

        # Выполнение запроса
        response = self.client.post("/createMesh", json=test_data)

        # Проверка статуса ответа
        self.assertEqual(response.status_code, 200)
        # Проверка типа содержимого ответа
        self.assertEqual(response.headers["content-type"], "application/json")

    @patch('EITSynthAI.kt_service.ai_tools.mesh_tools.main_mesh_service.create_mesh')
    def test_create_mesh_from_json_error(self, mock_create_mesh):
        # Подготовка тестовых данных
        test_data = {
            "params": [0.682, 0.682],
            "polygons": ["10,10,20,20", "30,30,40,40"]
        }

        # Настройка мока для генерации исключения
        mock_create_mesh.side_effect = Exception("Test error")

        # Выполнение запроса
        response = self.client.post("/createMesh", json=test_data)

        # Проверка статуса ответа
        self.assertEqual(response.status_code, 200)
        # Проверка содержимого ответа
        self.assertEqual(response.json(), {"status": "error", "message": "Test error"})


if __name__ == "__main__":
    unittest.main()
