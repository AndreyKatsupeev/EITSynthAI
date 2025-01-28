"""Скрипт для создания датасета для поиска рёбер на фронтальном срезе"""

import cv2
import numpy
import pydicom
import os

from tqdm import tqdm

workdir = '/media/msi/FSI/fsi/fsi_draft/datasets'
save_dir = f'{workdir}/font_dataset'   # директория для сохранения файлов
dicom_dataset_dir = f'{workdir}/dicom_main'

dicom_folders_name = [name for name in os.listdir(dicom_dataset_dir) if
                      os.path.isdir(os.path.join(dicom_dataset_dir, name))]


def read_dicom_folder(folder_path):
    """
    Функция для чтения dicom-файла. Один dicom - это один срез с метаинформацией.
    После снятия КТ с пациента они помещаются в одну папку в виде dicom-файлов. Если в папке один dicom,
    то скорее всего это один срез.

    :param folder_path: путь к папке
    :return: функция возвращает список dicom-файлов из одной директории
    """
    slices = []
    for filename in os.listdir(folder_path):
        if ".dcm" in filename.lower():
            filepath = os.path.join(folder_path, filename)
            dicom_data_slice = pydicom.dcmread(filepath)  # Срез с метаданными
            slices.append(dicom_data_slice)
    return slices


def convert_to_3d(slices):
    """
    Преобразование срезов в 3D-массив
    В данной функции мы получаем параметр (0018, 5100) Patient Position. Он бывает:
        FFS - Feet First Supine (Ноги вперед, супинированное положение)
        HFS - Head First Supine (Голова вперед, супинированное положение)
        FFP - Feet First Prone (Ноги вперед, пронированное положение)
        HFP - Head First Prone (Голова вперед, пронированное положение)
    Данный параметр затем используется для определения ориентации среза, чтобы на картинке не было тела вверх ногами.
    :param slices: список срезов с метаинформацией
    :return:
    """
    # Сортировка срезов по положению (при необходимости)
    slices.sort(key=lambda x: int(x.InstanceNumber))
    # Извлечение массива пиксельных данных
    pixel_data = [slice_dicom.pixel_array for slice_dicom in slices]
    # Получение позиции пациента
    patient_position = slices[0][0x0018, 0x5100].value
    # Стекирование в 3D-массив
    img_3d = numpy.stack(pixel_data,
                         axis=-1)  # Axis=-1 для аксиальных срезов, предполагая, что третий измерение - это срезы
    return img_3d, patient_position


# Шаг 3: Ребайндинг в сагиттальную плоскость с коррекцией ориентации
def axial_to_sagittal(img_3d, patient_position='HFS'):
    """
    Полученный 3D массив транспонируем в нужную плоскость

    :param img_3d: Трехмерный массив продольных срезов
    :param patient_position: Позиция пациента
    :return: Получаем нарезку вертикальных срезов
    """
    # Простое транспонирование для аксиального в сагиттальный с коррекцией ориентации
    if patient_position == 'FFS':
        sagittal_view = numpy.transpose(img_3d, (2, 1, 0))  # Перестановка осей
        sagittal_view = numpy.flipud(sagittal_view)
    elif patient_position == 'HFS':
        sagittal_view = numpy.transpose(img_3d, (2, 1, 0))  # Перестановка осей
    return sagittal_view


if __name__ == "__main__":
    global_count = 0
    for dicom_dir in tqdm(dicom_folders_name):
        slices = read_dicom_folder(f'{dicom_dataset_dir}/{dicom_dir}')  # список всех dicom из папки
        file_name = "fsi_ribs_segmentation"  # создаем новое имя файла для обезличивания
        img_3d, patient_position = convert_to_3d(slices)
        sagittal_view = axial_to_sagittal(img_3d, patient_position)  # нарезка вертикальных срезов
        slice_mean = sagittal_view.shape[-1] // 2  # Вычисляем средний срез
        for i in range(-6, 7):  # Берем диапазоны от среднего (Всего 13)
            slise_save = sagittal_view[:, :, slice_mean + i]
            # Нормализуем пиксели в диапазоне 0....255
            slise_save = cv2.normalize(slise_save, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
            # сохраняем
            cv2.imwrite(f'{save_dir}/{file_name}_{global_count}_{i}.jpg', slise_save)
        global_count += 1
        # cv2.namedWindow('sagittal_view', cv2.WINDOW_NORMAL)
        # cv2.imshow('sagittal_view', sagittal_view)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
