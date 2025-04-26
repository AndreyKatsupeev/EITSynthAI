from lungmask import LMInferer
from SimpleITK import ReadImage
import cv2
import numpy as np
import pydicom
import os

inferer = LMInferer()
file_path_list = []


INPUT = 'test_dicom/1.2.156.14702.1.1000.16.2.20170530094946906000200020131.dcm'


def get_hu(pixel_value):
    hounsfield_units = (1.0 * pixel_value) + (-1024.0)
    return hounsfield_units


def apply_hu_to_image(img):
    # Читаем изображение в grayscale режиме

    # Применяем функцию ко всем пикселям используя векторизацию NumPy
    hu_img = np.vectorize(get_hu)(img).astype(np.int16)  # Используем int16, чтобы сохранить диапазон HU
    return hu_img


files_path = '/media/msi/fsi/fsi/datasets_mrt/MosMedData-CT-COVID19-type VII-v 1/dicom/patient_006'

for directory, _, files in os.walk(files_path):
    # Обрабатываем каждый файл в текущей папке
    for file in files:
        # Собираем путь к файлу (без проверки расширения, как ранее)
        file_path = os.path.join(directory, file)
        file_path_list.append(file_path)

if __name__ == "__main__":

    for dicom_img in file_path_list:

        input_image = ReadImage(dicom_img)
        segmentation = inferer.apply(input_image)  # default model is U-net(R231)
        segmentation = np.squeeze(segmentation)

        contours, hierarchy = cv2.findContours(segmentation, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

        ds = pydicom.dcmread(dicom_img)
        new_image = ds.pixel_array
        new_image = apply_hu_to_image(new_image)

        if new_image.dtype != np.uint8:
            # Нормализация в диапазон 0-255, сохраняя относительную яркость пикселей
            img_normalized = (new_image - new_image.min()) / (new_image.max() - new_image.min()) * 255.0
            img_normalized = img_normalized.astype(np.uint8)
        else:
            img_normalized = new_image

        img_normalized = cv2.cvtColor(img_normalized, cv2.COLOR_GRAY2BGR)

        cv2.drawContours(img_normalized, contours, -1, (0, 255, 0), -1)

        cv2.namedWindow('input_image', cv2.WINDOW_NORMAL)
        cv2.imshow('input_image', img_normalized)
        cv2.waitKey(0)
cv2.destroyAllWindows()
