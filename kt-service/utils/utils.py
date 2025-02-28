import base64
import cv2
from collections import defaultdict
import logging
import numpy
import pydicom

from pydicom.filebase import DicomBytesIO

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_dicom_dict(zip_file):
    """"""
    series_dict = defaultdict(list)
    for file_name in zip_file.namelist():
        logger.info(f"Обработка файла: {file_name}")
        # Чтение файла в бинарном режиме и кодирование в base64
        file_data = zip_file.read(file_name)
        # extracted_files[file_name] = base64.b64encode(file_data).decode('utf-8')
        # Чтение DICOM-файла
        dicom_data = DicomBytesIO(file_data)
        dicom_data_slice_with_meta = pydicom.dcmread(dicom_data)

        series_uid = dicom_data_slice_with_meta.SeriesInstanceUID
        series_dict[series_uid].append(dicom_data_slice_with_meta)
    return series_dict, series_uid


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
