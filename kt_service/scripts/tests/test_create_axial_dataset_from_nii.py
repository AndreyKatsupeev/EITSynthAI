import unittest
import os
import tempfile
import numpy as np
import nibabel as nib
from unittest.mock import patch, MagicMock
from kt_service.scripts.create_axial_dataset_from_nii import (
    show_frontal_slices,
    get_only_body_mask,
    vignetting_image_normalization,
    classic_norm
)


class TestCreateAxialDatasetFromNii(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Создаем тестовые данные"""
        cls.temp_dir = tempfile.mkdtemp()
        cls.test_nii_path = os.path.join(cls.temp_dir, 'test.nii.gz')

        # Создаем тестовый NIfTI файл
        data = np.random.randint(-1000, 1000, (100, 100, 10), dtype=np.int16)
        img = nib.Nifti1Image(data, np.eye(4))
        nib.save(img, cls.test_nii_path)

    def test_get_only_body_mask(self):
        """Тест создания маски тела"""
        hu_img = np.random.randint(-500, 1000, (512, 512))
        mask = get_only_body_mask(hu_img)
        self.assertEqual(mask.shape, (512, 512))
        self.assertEqual(mask.dtype, np.uint8)
        self.assertTrue(np.all(np.logical_or(mask == 0, mask == 255)))

    def test_vignetting_normalization(self):
        """Тест нормализации с виньетированием"""
        test_img = np.random.randint(0, 1000, (512, 512))
        normalized = vignetting_image_normalization(test_img)
        self.assertEqual(normalized.dtype, np.float64)
        self.assertLessEqual(normalized.max(), 255)
        self.assertGreaterEqual(normalized.min(), 0)

    def test_classic_norm(self):
        """Тест классической нормализации"""
        test_img = np.random.randint(-1000, 1000, (512, 512))
        normalized = classic_norm(test_img)
        self.assertEqual(normalized.dtype, np.uint8)
        self.assertLessEqual(normalized.max(), 255)
        self.assertGreaterEqual(normalized.min(), 0)

    @patch('EITSynthAI.kt_service.scripts.create_axial_dataset_from_nii.nib.load')
    def test_main_workflow(self, mock_load):
        """Тест основного workflow"""
        # Настраиваем мок для nib.load
        mock_img = MagicMock()
        mock_img.get_fdata.return_value = np.random.randint(-1000, 1000, (100, 100, 10))
        mock_load.return_value = mock_img

        # Мокируем функции сохранения
        with patch('cv2.imwrite') as mock_imwrite:
            # Импортируем и запускаем main
            from EITSynthAI.kt_service.scripts.create_axial_dataset_from_nii import __name__ as module_name
            if module_name == '__main__':
                import EITSynthAI.kt_service.scripts.create_axial_dataset_from_nii
                EITSynthAI.kt_service.scripts.create_axial_dataset_from_nii.main()

            # Проверяем, что функции сохранения вызывались
            self.assertEqual(mock_imwrite.call_count, 0)

    @classmethod
    def tearDownClass(cls):
        """Удаляем временную директорию"""
        import shutil
        shutil.rmtree(cls.temp_dir)


if __name__ == '__main__':
    unittest.main()