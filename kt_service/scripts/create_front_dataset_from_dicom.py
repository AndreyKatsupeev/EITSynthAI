"""Скрипт для создания датасета для поиска рёбер на фронтальном срезе"""

import cv2
import numpy
import pydicom
import os

from collections import defaultdict
from tqdm import tqdm

workdir = '/media/msi/fsi/fsi/datasets_mrt'
save_dir = f'{workdir}/font_dataset/1'
dicom_dataset_dir = f'{workdir}/all_data_dicom'

dicom_folders_name = [name for name in os.listdir(dicom_dataset_dir) if
                      os.path.isdir(os.path.join(dicom_dataset_dir, name))]


def read_dicom_folder(folder_path):
    """
    Функция для чтения dicom-файла. Один dicom - это один срез с метаинформацией.
    После снятия КТ с пациента они помещаются в одну папку в виде dicom-файлов. Если в папке один dicom,
    то скорее всего это один срез. Встречаются папки с несколькими сериями, эта функция заносит их в словарь, где клоюч
    это номер серии.

    :param folder_path: путь к папке
    :return: функция возвращает словарь dicom-файлов из одной директории в пределах серии
    """
    series_dict = defaultdict(list)
    for filename in os.listdir(folder_path):
        try:
            filepath = os.path.join(folder_path, filename)
            dicom_data_slice = pydicom.dcmread(filepath)  # Срез с метаданными
            series_uid = dicom_data_slice.SeriesInstanceUID
            series_dict[series_uid].append(dicom_data_slice)
        except:
            pass
    return series_dict


def filter_arrays(array_list):
    # Создаем новый список, в который будем добавлять только массивы с размером 512x512
    filtered_list = [array for array in array_list if array.shape == (512, 512)]
    return filtered_list


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
    # print(len(pixel_data), '1')
    # pixel_data = filter_arrays(pixel_data)
    # print(len(pixel_data),'2')
    # Получение позиции пациента
    patient_position = slices[0][0x0018, 0x5100].value
    image_orientation = slices[0][0x0020, 0x0037].value  # Ориентация изображения (6 чисел)
    try:
        patient_orientation = slices[0][0x0020, 0x0020].value  # Ориентация пациента (например, A\P)
    except:
        patient_orientation = None
    # Стекирование в 3D-массив
    img_3d = numpy.stack(pixel_data,
                         axis=-1)  # Axis=-1 для аксиальных срезов, предполагая, что третий измерение - это срезы
    return img_3d, patient_position, image_orientation, patient_orientation


# Шаг 3: Ребайндинг в сагиттальную плоскость с коррекцией ориентации
def axial_to_sagittal(img_3d, patient_position, image_orientation, patient_orientation):
    """
    Преобразует 3D-изображение из аксиальной плоскости в сагиттальную с учетом ориентации пациента.

    :param img_3d: 3D-массив (аксиальные срезы)
    :param ds: DICOM dataset (содержит метаданные, включая PatientPosition, ImageOrientationPatient и PatientOrientation)
    :return: 3D-массив в сагиттальной плоскости
    """

    # Перестановка осей для преобразования аксиального в сагиттальный вид
    if patient_position == 'FFS':
        sagittal_view = numpy.transpose(img_3d, (2, 1, 0))  # Перестановка осей
        sagittal_view = numpy.flipud(sagittal_view)
    elif patient_position == 'HFS':
        sagittal_view = numpy.transpose(img_3d, (2, 1, 0))  # Перестановка осей

    # Коррекция на основе ImageOrientationPatient
    # Векторы ImageOrientationPatient описывают ориентацию строк и столбцов изображения
    # Первые три числа — направление строк (обычно X), последние три — направление столбцов (обычно Y)
    row_orientation = numpy.array(image_orientation[:3])  # Направление строк
    col_orientation = numpy.array(image_orientation[3:])  # Направление столбцов

    # Если направление строк или столбцов указывает в противоположную сторону, переворачиваем изображение
    if row_orientation[0] == -1:  # Если ось X направлена влево
        sagittal_view = numpy.flip(sagittal_view, axis=1)  # Переворот по оси Y
    if col_orientation[1] == -1:  # Если ось Y направлена назад
        sagittal_view = numpy.flip(sagittal_view, axis=2)  # Переворот по оси Z

    # Коррекция на основе PatientOrientation
    # PatientOrientation описывает, как пациент ориентирован относительно плоскости изображения
    if patient_position != 'HFS':
        if patient_orientation:
            if patient_orientation[0] == 'L':
                sagittal_view = numpy.fliplr(sagittal_view)  # Переворот по горизонтали (левая сторона станет слева)
            if patient_orientation[1] == 'P':
                sagittal_view = numpy.flipud(sagittal_view)  # Переворот по вертикали (задняя часть станет внизу)

    return sagittal_view


if __name__ == "__main__":
    global_count = 0
    for dicom_dir in tqdm(dicom_folders_name):
        slices = read_dicom_folder(f'{dicom_dataset_dir}/{dicom_dir}')  # список всех dicom из папки
        for i_slices in slices.values():
            try:
                file_name = dicom_dir  # создаем новое имя файла для обезличивания
                img_3d, patient_position, image_orientation, patient_orientation = convert_to_3d(i_slices)
                sagittal_view = axial_to_sagittal(img_3d, patient_position, image_orientation,
                                                  patient_orientation)  # нарезка вертикальных срезов
                slice_mean = sagittal_view.shape[-1] // 2  # Вычисляем средний срез
                for i in range(-3, 4):  # Берем диапазоны от среднего (Всего 13)
                    slise_save = sagittal_view[:, :, slice_mean + i]
                    # Нормализуем пиксели в диапазоне 0....255
                    slise_save = cv2.normalize(slise_save, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
                    # сохраняем
                    cv2.imwrite(f'{save_dir}/{file_name}_{global_count}_{i}.jpg', slise_save)
                global_count += 1
                # if global_count == 20:
                #     exit()

                # cv2.namedWindow('sagittal_view', cv2.WINDOW_NORMAL)
                # cv2.imshow('sagittal_view', sagittal_view)
                # cv2.waitKey(0)
                # cv2.destroyAllWindows()
            except:
                print(file_name)
