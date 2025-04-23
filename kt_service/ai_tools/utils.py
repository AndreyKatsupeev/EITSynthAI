import torch
import cv2
from collections import defaultdict
import logging
import numpy
import pydicom
import supervision as sv
from scipy.ndimage import label
from pydicom.filebase import DicomBytesIO
import base64
from PIL import Image
from fastapi.responses import JSONResponse
import nibabel as nib
import tempfile
import os
from io import BytesIO

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_dicom_dict(zip_file):
    """
    Извлекает DICOM файлы из zip-архива и возвращает срезы самой большой серии.
    Также проверяет наличие файла custom_input.txt и возвращает его содержимое.

    Args:
        zip_file: Объект ZipFile с DICOM файлами

    Returns:
        tuple: (list_of_dicom_slices, custom_input)
               где list_of_dicom_slices - список DICOM объектов,
               custom_input - содержимое файла custom_input.txt или None
    """
    series_dict = defaultdict(list)
    custom_input = None

    # Проверяем наличие custom_input.txt
    if 'custom_input.txt' in zip_file.namelist():
        with zip_file.open('custom_input.txt') as f:
            custom_input = f.read().decode('utf-8').strip()

    # Группируем все серии DICOM
    for file_name in zip_file.namelist():
        if file_name.lower().endswith('.dcm') or not file_name.lower().endswith('.txt'):
            try:
                with zip_file.open(file_name) as file:
                    dicom_data = DicomBytesIO(file.read())
                    dicom_slice = pydicom.dcmread(dicom_data)
                    series_dict[dicom_slice.SeriesInstanceUID].append(dicom_slice)
            except Exception as e:
                print(f"Ошибка при обработке файла {file_name}: {str(e)}")
                continue

    # Находим самую большую серию
    if not series_dict:
        return [], custom_input
    if custom_input is None:
        custom_input = 0
    largest_series = max(series_dict.values(), key=len)
    return largest_series, int(custom_input)


def create_image_dict(zip_file):
    """

    """
    series_dict = defaultdict(list)
    custom_input = None

    largest_series = max(series_dict.values(), key=len)
    return largest_series


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
    # for i in slices:
    #     print(slices[(0x0020, 0x0032)])
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
    patient_position = 'FFS'
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


def search_number_axial_slice(detections, custom_number_slise=0, image_width=512):
    """

    Detections(xyxy=array([[     100.45,      109.37,      116.43,      129.18],
       [     412.88,      162.68,      426.44,      182.76],
       [     90.846,      146.93,      105.72,      168.55],
       [     67.141,      236.86,      82.394,      262.65],
       [     79.154,      189.92,      94.161,      213.11],
       [     392.32,      93.775,       409.2,      111.35],
       [     114.18,      76.355,       130.5,      92.696],
       [     317.95,      19.249,      335.81,      31.386],
       [     131.96,       45.55,      147.82,        59.8],
       [      426.9,      243.08,      439.85,       269.9],
       [     180.57,      8.3435,      198.91,      21.686],
       [     404.69,      125.41,      419.11,      144.29],
       [     60.132,      291.74,      70.879,      312.55],
       [     373.74,      62.977,      389.99,      78.234],
       [     152.17,      26.801,      169.74,       38.47],
       [     416.98,      201.76,      430.79,      226.25],
       [     346.93,      39.076,      365.61,      51.212],
       [     435.91,      303.05,      446.96,      323.76],
       [     59.205,      352.68,      68.983,      362.67]], dtype=float32), mask=array([[[False, False, False, ..., False, False, False],
        [False, False, False, ..., False, False, False],
        [False, False, False, ..., False, False, False],
        ...,
        [False, False, False, ..., False, False, False],
        [False, False, False, ..., False, False, False],
        [False, False, False, ..., False, False, False]],

       [[False, False, False, ..., False, False, False],
        [False, False, False, ..., False, False, False],
        [False, False, False, ..., False, False, False],
        ...,
        [False, False, False, ..., False, False, False],
        [False, False, False, ..., False, False, False],
        [False, False, False, ..., False, False, False]],

       [[False, False, False, ..., False, False, False],
        [False, False, False, ..., False, False, False],
        [False, False, False, ..., False, False, False],
        ...,
        [False, False, False, ..., False, False, False],
        [False, False, False, ..., False, False, False],
        [False, False, False, ..., False, False, False]],

       ...,

       [[False, False, False, ..., False, False, False],
        [False, False, False, ..., False, False, False],
        [False, False, False, ..., False, False, False],
        ...,
        [False, False, False, ..., False, False, False],
        [False, False, False, ..., False, False, False],
        [False, False, False, ..., False, False, False]],

       [[False, False, False, ..., False, False, False],
        [False, False, False, ..., False, False, False],
        [False, False, False, ..., False, False, False],
        ...,
        [False, False, False, ..., False, False, False],
        [False, False, False, ..., False, False, False],
        [False, False, False, ..., False, False, False]],
       [[False, False, False, ..., False, False, False], [False, False, False, ..., False, False, False], [False,
       False, False, ..., False, False, False], ..., [False, False, False, ..., False, False, False], [False, False,
       False, ..., False, False, False], [False, False, False, ..., False, False, False]]]),
       confidence=array([
       0.79298,     0.79022,     0.77921,     0.77907,      0.7766,     0.77603,     0.77508,     0.77445,
       0.77365,     0.77216,     0.76479,     0.76349,     0.76013,      0.7598,     0.75843,     0.75402,
       0.75367,      0.7343,     0.69417], dtype=float32),
       class_id=array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
       0, 0, 0, 0, 0]),
       tracker_id=None,
       data={'class_name': array(['rib', 'rib', 'rib', 'rib', 'rib', 'rib', 'rib',
       'rib', 'rib', 'rib', 'rib', 'rib', 'rib', 'rib', 'rib', 'rib', 'rib', 'rib', 'rib'],
       dtype='<U3')}, metadata={})
    """
    number_axial_slice_list = []
    coordinates = detections.xyxy
    midpoint = image_width / 2
    # Фильтрация координат, оставляем только те, что правее середины
    right_side_coordinates = [box for box in coordinates if box[0] > midpoint]
    # Сортировка по оси Y (по второму элементу каждого бокса)
    sorted_right_side_coordinates = sorted(right_side_coordinates, key=lambda x: x[1])
    number_axial_slice = int((abs(sorted_right_side_coordinates[5][1] + sorted_right_side_coordinates[6][1])) / 2)
    number_axial_slice_list.append(int(sorted_right_side_coordinates[5][1]))
    number_axial_slice_list.append(int(sorted_right_side_coordinates[6][1]))
    number_axial_slice_list.append(number_axial_slice+custom_number_slise)
    return number_axial_slice_list


def classic_norm(volume, window_level=40, window_width=400):
    """"""
    # Нормализация HU
    hu_min = window_level - window_width // 2
    hu_max = window_level + window_width // 2
    clipped = numpy.clip(volume, hu_min, hu_max)
    normalized = ((clipped - hu_min) / (hu_max - hu_min) * 255).astype(numpy.uint8)
    normalized = cv2.rotate(normalized, cv2.ROTATE_180)
    return normalized


def draw_annotate(ribs_detections, front_slice, axial_slice_list_numbers):
    """"""
    box_annotator = sv.BoxAnnotator(color=sv.Color.BLUE)
    annotated_image = front_slice.copy()
    annotated_image = cv2.cvtColor(annotated_image, cv2.COLOR_GRAY2BGR)
    annotated_image = box_annotator.annotate(annotated_image, detections=ribs_detections)
    annotated_image = cv2.line(annotated_image, (0, axial_slice_list_numbers[-1]), (1000, axial_slice_list_numbers[-1]),
                               (0, 0, 255), 1)
    return annotated_image


def overlay_segmentation_masks(segmentation_dict):
    # Получаем размеры из первого изображения
    first_key = next(iter(segmentation_dict))
    height, width = segmentation_dict[first_key].shape[:2]

    # Создаем пустое RGB изображение
    overlay = numpy.zeros((height, width, 3), dtype=numpy.uint8)

    # Цвета для разных сегментов (BGR формат)
    colors = {
        "adipose": (0, 255, 255),
        "bone": (255, 255, 255),
        "muscles": (0, 0, 255),
        "lung": (255, 255, 0)
    }

    for name, mask in segmentation_dict.items():
        # Нормализуем маску (на случай если она не бинарная)
        if mask.dtype != numpy.uint8:
            mask = mask.astype(numpy.uint8)

        # Если маска трехканальная, преобразуем в одноканальную
        if len(mask.shape) == 3:
            mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)

        # Создаем цветную маску
        color = colors.get(name, [255, 255, 255])
        colored_mask = numpy.zeros((height, width, 3), dtype=numpy.uint8)

        # Применяем цвет только к ненулевым пикселям
        mask_bool = mask > 0
        colored_mask[mask_bool] = color

        # Накладываем на общее изображение
        overlay = cv2.add(overlay, colored_mask)
    return overlay


def create_segmentations_masks(axial_segmentations, img_size=512):
    clrs = {
        "adipose": (0, 255, 255),
        "bone": (255, 255, 255),
        "muscles": (0, 0, 255),
        "lung": (255, 255, 0)
    }
    # Получаем данные из результатов YOLO
    mask_coords_list = axial_segmentations.masks.data  # Координаты масок
    class_ids = axial_segmentations.boxes.cls.cpu().numpy()  # Классы
    # Создаем словарь для хранения изображений по классам
    class_images = {
        "bone": numpy.zeros((img_size, img_size, 3), dtype=numpy.uint8),
        "muscles": numpy.zeros((img_size, img_size, 3), dtype=numpy.uint8),
        "lung": numpy.zeros((img_size, img_size, 3), dtype=numpy.uint8),
        "adipose": numpy.zeros((img_size, img_size, 3), dtype=numpy.uint8)
    }

    # Обрабатываем каждую маску
    for i, mask in enumerate(mask_coords_list):
        # Перемещаем тензор на CPU и преобразуем в NumPy массив
        if torch.is_tensor(mask):
            mask = mask.cpu().numpy()

        class_id = class_ids[i]

        # Определяем имя класса
        if class_id == 0:
            class_name = "bone"
        elif class_id == 1:
            class_name = "muscles"
        elif class_id == 2:
            class_name = "lung"
        elif class_id == 3:
            class_name = "adipose"
        else:
            continue  # пропускаем неизвестные классы

        # Получаем цвет для текущего класса
        color = clrs[class_name]

        # Создаем маску в формате (H, W, 3)
        colored_mask = numpy.zeros((img_size, img_size, 3), dtype=numpy.uint8)
        # Применяем цвет там, где маска не равна 0
        colored_mask[mask > 0] = color

        # Добавляем маску к соответствующему изображению класса
        class_images[class_name] = cv2.add(class_images[class_name], colored_mask)

    return class_images


def get_axial_slice_body_mask(ds):
    """
    Функция для поиска маски среза тела

    Функция предназначена для отсечения посторонних предметов из среза КТ. Очень часто в срез попадает стол аппарата.
    Этот метод отсекает все меленькие маски и оставляет самую большую - тело.

    Args:
        hu_img: изображение 512х512, содержащее HU-коэффициенты

    Returns:
        only_body_mask: cv2.image

    """

    new_image = ds.pixel_array
    new_image = numpy.flipud(new_image)
    rescale_intercept = get_rescale_intercept(ds)
    rescale_slope = get_rescale_slope(ds)
    hu_img = numpy.vectorize(get_hu, excluded=['rescale_intercept', 'rescale_slope']) \
        (new_image, rescale_intercept, rescale_slope).astype(
        numpy.int16)  # Используем int16, чтобы сохранить диапазон HU

    kernel_only_body_mask = numpy.ones((5, 5), numpy.uint8)
    only_body_mask = numpy.where((hu_img > -500) & (hu_img < 1000), 1, 0)
    only_body_mask = only_body_mask.astype(numpy.uint8)

    only_body_mask = cv2.morphologyEx(only_body_mask, cv2.MORPH_OPEN, kernel_only_body_mask)

    contours, hierarchy = cv2.findContours(only_body_mask,
                                           cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    max_contour = max(contours, key=cv2.contourArea, default=None)
    if max_contour is not None:
        only_body_mask = numpy.zeros_like(only_body_mask)
    cv2.drawContours(only_body_mask, [max_contour], 0, 255, -1)
    return only_body_mask


def get_axial_slice_body_mask_nii(hu_img):
    """
    Функция для поиска маски среза тела

    Функция предназначена для отсечения посторонних предметов из среза КТ. Очень часто в срез попадает стол аппарата.
    Этот метод отсекает все меленькие маски и оставляет самую большую - тело.

    Args:
        hu_img: изображение 512х512, содержащее HU-коэффициенты

    Returns:
        only_body_mask: cv2.image

    """
    kernel_only_body_mask = numpy.ones((5, 5), numpy.uint8)
    only_body_mask = numpy.where((hu_img > -500) & (hu_img < 1000), 1, 0)
    only_body_mask = only_body_mask.astype(numpy.uint8)

    only_body_mask = cv2.morphologyEx(only_body_mask, cv2.MORPH_OPEN, kernel_only_body_mask)

    contours, hierarchy = cv2.findContours(only_body_mask,
                                           cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    max_contour = max(contours, key=cv2.contourArea, default=None)
    if max_contour is not None:
        only_body_mask = numpy.zeros_like(only_body_mask)
    cv2.drawContours(only_body_mask, [max_contour], 0, 255, -1)
    return only_body_mask


def get_rescale_intercept(dicom_data):
    """
    Параметр Rescale Intercept в DICOM-файле отвечает за смещение, которое применяется к значениям пикселей после
    их масштабирования с помощью Rescale Slope. Он используется в формуле преобразования сырых значений пикселей
    (как они хранятся в файле) в реальные физические значения, которые используются
    для интерпретации медицинских изображений.
    Args:
        dicom_data:

    Returns:

    """
    return int(dicom_data[(0x0028, 0x1052)].value)


def get_rescale_slope(dicom_data):
    """
    Параметр Rescale Slope в DICOM-файле отвечает за преобразование значений пикселей (вокселей) из их исходного
    формата (как они хранятся в файле) в реальные физические значения, которые используются для интерпретации данных.
    Args:
        dicom_data:

    Returns:

    """
    rescale_slope = int(dicom_data[(0x0028, 0x1053)].value)
    return rescale_slope


def get_hu(pixel_value, rescale_intercept=0, rescale_slope=1.0):
    """
    Функция для вычисления HU из значения пикселей dicom-файла

    Формула взята отсюда https://stackoverflow.com/questions/22991009/how-to-get-hounsfield-units-in-dicom-file-
    using-fellow-oak-dicom-library-in-c-sh

    Краткое справка приведена в начале скрипта

    Real Value=(Stored Pixel Value×Rescale Slope)+Rescale Intercept
    Stored Pixel Value — значение пикселя, как оно хранится в DICOM-файле.

    Rescale Slope — коэффициент масштабирования.

    Rescale Intercept — смещение, которое добавляется после умножения.

    Args:
        pixel_value:
        rescale_intercept:
        rescale_slope:

    Returns:

    """
    hounsfield_units = (rescale_slope * pixel_value) + rescale_intercept
    return hounsfield_units


def clear_color_output(only_body_mask, color_output, tolerance=5, min_polygon_size=5):
    mask_organs_processed = color_output.copy()
    h, w = mask_organs_processed.shape[:2]

    # 1. Закрашиваем почти чёрные пиксели внутри тела красным
    is_black = numpy.all(numpy.abs(color_output - [0, 0, 0]) <= tolerance, axis=2)
    is_in_body = (only_body_mask == 255)
    to_fill = is_black & is_in_body
    mask_organs_processed[to_fill] = [0, 0, 255]  # Красный в BGR

    # 2. Находим все связные области (полигоны), кроме фона (чёрного/красного)
    background_colors = [
        [0, 0, 0],  # Чёрный
        [0, 0, 255]  # Красный (уже закрашенные области)
    ]
    is_background = numpy.zeros((h, w), dtype=bool)
    for color in background_colors:
        is_background |= numpy.all(mask_organs_processed == color, axis=2)

    # Размечаем все связные области (каждый полигон получает уникальный label)
    labeled, num_features = label(~is_background)

    # 3. Проходим по всем полигонам и закрашиваем маленькие (<5 пикселей)
    for label_idx in range(1, num_features + 1):
        polygon_mask = (labeled == label_idx)
        polygon_size = numpy.sum(polygon_mask)

        if polygon_size < min_polygon_size:
            # Находим соседние цвета (игнорируя чёрный и красный)
            y, x = numpy.where(polygon_mask)
            neighbors = []

            # Проверяем 8-связных соседей для каждой точки полигона
            for dy, dx in [(-1, -1), (-1, 0), (-1, 1),
                           (0, -1), (0, 1),
                           (1, -1), (1, 0), (1, 1)]:
                ny, nx = y + dy, x + dx
                valid = (ny >= 0) & (ny < h) & (nx >= 0) & (nx < w)
                ny, nx = ny[valid], nx[valid]

                for color in mask_organs_processed[ny, nx]:
                    if not any(numpy.array_equal(color, bg_color) for bg_color in background_colors):
                        neighbors.append(tuple(color))  # Конвертируем в кортеж для хеширования

            if neighbors:
                # Находим самый частый цвет среди соседей (по хешу кортежа)
                from collections import Counter
                neighbor_color = Counter(neighbors).most_common(1)[0][0]
                mask_organs_processed[polygon_mask] = neighbor_color
            else:
                # Если соседей нет, закрашиваем красным (как фоновым)
                mask_organs_processed[polygon_mask] = [0, 0, 255]

    return mask_organs_processed


def highlight_small_masks(image, area_threshold=5):
    """
    Выделяет и перекрашивает маленькие маски (области) на изображении, заменяя их цветом соседних пикселей.

    Функция ищет маски определенных цветов (кости, мышцы, жир, воздух) на изображении и для тех масок,
    размер которых меньше заданного порога, заменяет их цвет на наиболее распространенный цвет соседних пикселей.

    Параметры:
    ----------
    image : numpy.ndarray
        Входное изображение в формате BGR (используется в OpenCV).
    area_threshold : int, optional
        Максимальный размер маски (в пикселях), которая считается маленькой и подлежит обработке.
        По умолчанию 5.

    Возвращает:
    -----------
    numpy.ndarray
        Изображение того же размера, что и входное, с перекрашенными маленькими масками.

    """

    # Цвета масок для разных типов тканей в формате BGR
    mask_colors = {
        "bone": (255, 255, 255),  # Белый - кости
        "muscle": (0, 0, 255),  # Красный - мышцы
        "fat": (0, 255, 255),  # Желтый - жир
        "air": (0, 150, 255),  # Оранжевый - воздух
    }

    # Создаем копию изображения для модификации
    output = image.copy()

    # Обрабатываем каждый тип ткани отдельно
    for tissue, target_color in mask_colors.items():
        # Определяем диапазон цветов для текущего типа ткани (±10 от целевого цвета)
        lower = numpy.array(target_color, dtype=numpy.int16) - 10
        upper = numpy.array(target_color, dtype=numpy.int16) + 10

        # Создаем бинарную маску для текущего цвета ткани
        mask = cv2.inRange(image, lower, upper)

        # Находим контуры всех масок текущего цвета
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Обрабатываем каждый контур отдельно
        for cnt in contours:
            # Если размер контура меньше порогового значения
            if len(cnt) <= area_threshold:
                # Создаем маску только для текущего контура
                contour_mask = numpy.zeros(image.shape[:2], dtype=numpy.uint8)
                cv2.drawContours(contour_mask, [cnt], -1, 255, cv2.FILLED)

                # Расширяем маску контура на 1 пиксель, чтобы получить соседние пиксели
                dilated = cv2.dilate(contour_mask, numpy.ones((3, 3), numpy.uint8), iterations=1)
                neighbors_mask = dilated - contour_mask

                # Получаем цвета соседних пикселей
                neighbor_colors = output[neighbors_mask == 255]

                if len(neighbor_colors) > 0:
                    # Фильтруем цвета: убираем целевой цвет и черный (фон)
                    neighbor_colors = [tuple(c) for c in neighbor_colors
                                       if not numpy.array_equal(c, target_color)
                                       and not numpy.array_equal(c, (0, 0, 0))]

                    if neighbor_colors:
                        # Выбираем наиболее часто встречающийся цвет соседей
                        from collections import Counter
                        fill_color = Counter(neighbor_colors).most_common(1)[0][0]
                    else:
                        # Если подходящих соседей нет, оставляем исходный цвет
                        fill_color = target_color
                else:
                    # Если совсем нет соседей, оставляем исходный цвет
                    fill_color = target_color

                # Преобразуем цвет в кортеж целых чисел (на случай, если был numpy array)
                fill_color = tuple(map(int, fill_color))

                # Закрашиваем маленькую маску выбранным цветом
                cv2.drawContours(output, [cnt], -1, fill_color, thickness=cv2.FILLED)

    return output


def overlay_masks_with_transparency(base_image, color_mask, alpha=0.8):
    """
    Наложение цветной маски на базовое изображение с прозрачностью

    Параметры:
    - base_image: базовое изображение (512, 512)
    - color_mask: цветная маска (512, 512, 3)
    - alpha: уровень прозрачности (0-1)
    """
    # 1. Конвертируем базовое изображение в RGB (если оно grayscale)
    if len(base_image.shape) == 2:
        base_image = cv2.cvtColor(base_image, cv2.COLOR_GRAY2BGR)

    # 2. Нормализуем изображения (если нужно)
    if base_image.dtype != numpy.uint8:
        base_image = cv2.normalize(base_image, None, 0, 255, cv2.NORM_MINMAX).astype(numpy.uint8)

    if color_mask.dtype != numpy.uint8:
        color_mask = cv2.normalize(color_mask, None, 0, 255, cv2.NORM_MINMAX).astype(numpy.uint8)

    # 3. Наложение с прозрачностью
    overlay = cv2.addWeighted(base_image, 1.0, color_mask, alpha, 0)

    return overlay


def create_segmentation_masks_full_image(segmentation_masks_image=None, only_body_mask=None,
                                       ribs_annotated_image=None, axial_slice_norm_body=None,
                                       img_mesh=None):
    """
    Создает комбинированное изображение из доступных масок и аннотаций.
    Если какой-то из аргументов пустой (None или пустой массив), он пропускается.
    Изображение меша (img_mesh) добавляется в конец сетки.

    Args:
        segmentation_masks_image: dict с сегментационными масками
        only_body_mask: маска тела
        ribs_annotated_image: изображение с аннотированными ребрами
        axial_slice_norm_body: аксиальный срез с нормализованным цветом
        img_mesh: изображение с меш-визуализацией (будет добавлено в конец)

    Returns:
        Комбинированное изображение с доступными компонентами
    """
    images_to_combine = []

    # 1. Обрабатываем ribs_annotated_image, если он есть
    if ribs_annotated_image is not None and numpy.any(ribs_annotated_image):
        images_to_combine.append(("1. Ribs Annotated", ribs_annotated_image))

    # 2. Обрабатываем axial_slice_norm_body, если он есть
    if axial_slice_norm_body is not None and numpy.any(axial_slice_norm_body):
        images_to_combine.append(("2. Axial Slice", axial_slice_norm_body))

    # 3. Обрабатываем segmentation_masks_image, если он есть
    if segmentation_masks_image is not None and len(segmentation_masks_image) > 0:
        color_output = create_color_output(segmentation_masks_image, only_body_mask)

        if axial_slice_norm_body is not None and numpy.any(axial_slice_norm_body):
            axial_slice_norm_body_with_color = overlay_masks_with_transparency(axial_slice_norm_body, color_output)
            images_to_combine.append(("3. Combined View", axial_slice_norm_body_with_color))

        images_to_combine.append(("4. Color Masks", color_output))

        # Добавляем отдельные маски из словаря
        for idx, (key, image) in enumerate(segmentation_masks_image.items(), start=5):
            if image is not None and numpy.any(image):
                images_to_combine.append((f"{idx}. {key}", image))

    # 4. Обрабатываем img_mesh, если он есть (добавляем в конец)
    if img_mesh is not None and numpy.any(img_mesh):
        images_to_combine.append(("Mesh Visualization", img_mesh))

    # Если нет изображений для объединения, возвращаем пустое изображение
    if not images_to_combine:
        return numpy.zeros((100, 100, 3), dtype=numpy.uint8)

    # 5. Приводим все изображения к одному размеру (берем максимальные размеры)
    max_height = max(img.shape[0] for _, img in images_to_combine)
    max_width = max(img.shape[1] for _, img in images_to_combine)

    # 6. Добавляем подписи и выравниваем размеры
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.8
    font_color = (255, 255, 255)
    thickness = 1

    labeled_images = []
    for label, image in images_to_combine:
        # Конвертируем в цветное если нужно
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        elif image.shape[2] == 1:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

        # Выравниваем размеры
        if image.shape[0] != max_height or image.shape[1] != max_width:
            image = cv2.resize(image, (max_width, max_height))

        # Создаем копию для подписи
        labeled = image.copy()
        h, w = labeled.shape[:2]

        # Добавляем подпись
        text_size = cv2.getTextSize(label, font, font_scale, thickness)[0]
        text_x = (w - text_size[0]) // 2
        text_y = h - 10  # Внизу изображения

        cv2.putText(labeled, label, (text_x, text_y), font,
                    font_scale, font_color, thickness, cv2.LINE_AA)

        labeled_images.append(labeled)

    # 7. Определяем размеры сетки
    num_images = len(labeled_images)
    cols = min(3, num_images)  # Не более 3 колонок, но меньше если изображений мало
    rows = (num_images + cols - 1) // cols  # Вычисляем нужное количество строк

    # 8. Создаем результирующее изображение
    result = numpy.zeros((max_height * rows, max_width * cols, 3), dtype=numpy.uint8)

    # 9. Заполняем сетку изображениями
    for i in range(rows):
        for j in range(cols):
            idx = i * cols + j
            if idx < num_images:
                y_start = i * max_height
                y_end = (i + 1) * max_height
                x_start = j * max_width
                x_end = (j + 1) * max_width
                result[y_start:y_end, x_start:x_end] = labeled_images[idx]

    return result


def create_color_output(segmentation_masks_image, only_body_mask=None):
    """
    Создает цветные маски сегментации.

    Args:
        segmentation_masks_image: dict с сегментационными масками
        only_body_mask: маска тела (опционально)

    Returns:
        Цветное изображение с наложенными масками
    """
    if segmentation_masks_image is None or len(segmentation_masks_image) == 0:
        return None

    color_output = overlay_segmentation_masks(segmentation_masks_image)

    if only_body_mask is not None and numpy.any(only_body_mask):
        color_output = clear_color_output(only_body_mask, color_output)

    color_output = highlight_small_masks(color_output)

    return color_output


def create_segmentation_results_cnt(axial_detections):
    """"""
    text = 'cnt'
    return text


def create_answer(segmentation_masks_full_image, segmentation_results_cnt, segmentation_time):
    """
    Формирует ответ для отправки клиенту, содержащий изображение и текстовые данные

    Args:
        segmentation_masks_full_image: изображение (numpy array)
        segmentation_results_cnt: текстовые данные (str)

    Returns:
        dict: словарь с ответом, содержащим изображение в base64 и текст
    """
    # Конвертируем numpy array в изображение PIL
    segmentation_masks_full_image = cv2.cvtColor(segmentation_masks_full_image, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(segmentation_masks_full_image)

    # Конвертируем изображение в байты
    img_byte_arr = BytesIO()
    pil_img.save(img_byte_arr, format='PNG')
    img_byte_arr = img_byte_arr.getvalue()

    # Кодируем изображение в base64
    img_base64 = base64.b64encode(img_byte_arr).decode('utf-8')

    # Формируем ответ
    answer = {
        "image": img_base64,
        "text_data": segmentation_results_cnt,
        "segmentation_time": segmentation_time,
        "status": "success",
        "message": "Processing completed successfully"
    }

    return JSONResponse(content=answer)


def get_nii_mean_slice(zip_file):
    """
    Args:
        zip_file: ZIP-архив с NIfTI-файлами (.nii.gz)
    Returns:
        Средний срез (после поворота на 90°)
    """
    # Проверяем наличие custom_input.txt
    if 'custom_input.txt' in zip_file.namelist():
        with zip_file.open('custom_input.txt') as f:
            custom_input = f.read().decode('utf-8').strip()

    data = None

    # Ищем .nii.gz файлы (игнорируем .tar.gz)
    for file_name in zip_file.namelist():
        if file_name.lower().endswith('.nii.gz') and not file_name.lower().endswith('.tar.gz'):
            try:
                with zip_file.open(file_name) as file:
                    file_content = file.read()  # Читаем файл в память

                    # Создаем временный файл
                    with tempfile.NamedTemporaryFile(delete=False, suffix='.nii.gz') as tmp_file:
                        tmp_file.write(file_content)
                        tmp_file_path = tmp_file.name

                    # Загружаем через nibabel
                    nii_data = nib.load(tmp_file_path)
                    data = nii_data.get_fdata().astype(numpy.int16)

                    # Удаляем временный файл
                    os.unlink(tmp_file_path)
                    hu_data = data * 1.0 - 0  # slope=1, intercept=-1024
                    slice_mean = int(hu_data.shape[-1] / 2)
                    slise_save = hu_data[:, :, slice_mean]
                    slise_save = cv2.rotate(slise_save, cv2.ROTATE_90_CLOCKWISE)

                    break  # Обрабатываем первый подходящий файл
            except Exception as e:
                print(f"Ошибка при обработке файла {file_name}: {str(e)}")
                if 'tmp_file_path' in locals() and os.path.exists(tmp_file_path):
                    os.unlink(tmp_file_path)
                continue

    if data is None:
        raise ValueError("Не удалось загрузить NIfTI файл из архива")

    return slise_save


def get_pixel_spacing(dicom_data):
    """
    Функция получения коэффициентов для преобразования значений в пикселях в миллиметры. Используется стандартный тег
    "Pixel Spacing" (0028, 0030)

    Args:
        dicom_data: прочитанный dicom

    Returns:
        pixel_spacing: (0028, 0030) Pixel Spacing DS: [0.753906, 0.753906] - можно обращаться через индекс

    """
    pixel_spacing = dicom_data[(0x0028, 0x0030)]
    return pixel_spacing


def create_list_crd_from_color_output(color_output, pixel_spacing):
    # Цвета масок и соответствующие классы
    color_class_map = {
        (0, 255, 255): "3",
        (255, 255, 255): "0",
        (0, 0, 255): "1",
        (255, 255, 0): "2"
    }

    result = []

    # Конвертируем в BGR, если изображение в RGB
    img = cv2.cvtColor(color_output, cv2.COLOR_RGB2BGR)

    for color, class_name in color_class_map.items():
        # Создаем маску для текущего цвета (в порядке BGR для OpenCV)
        bgr_color = color[::-1]  # конвертируем RGB в BGR
        lower = numpy.array(bgr_color, dtype=numpy.uint8)
        upper = numpy.array(bgr_color, dtype=numpy.uint8)
        mask = cv2.inRange(img, lower, upper)

        # Находим контуры
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for cnt in contours:
            # Упрощаем контур (уменьшаем количество точек)
            epsilon = 0.005 * cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, epsilon, True)

            # Проверяем замкнутость контура
            if len(approx) > 2:
                first_point = approx[0][0]
                last_point = approx[-1][0]

                # Если контур не замкнут, добавляем первую точку в конец
                if not numpy.array_equal(first_point, last_point):
                    approx = numpy.append(approx, [[first_point]], axis=0)

            # Формируем строку с координатами полигона
            points_str = " ".join([f"{p[0][0]} {p[0][1]}" for p in approx])
            polygon_str = f"{class_name} {points_str}"
            result.append(polygon_str)
    result.insert(0, str(pixel_spacing[1]))
    result.insert(0, str(pixel_spacing[0]))

    return result
