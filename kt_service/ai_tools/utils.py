import base64
import cv2
from collections import defaultdict
import logging
import numpy
import pydicom
import supervision as sv

from pydicom.filebase import DicomBytesIO

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_dicom_dict(zip_file):
    """
    Извлекает DICOM файлы из zip-архива и возвращает срезы самой большой серии.

    Args:
        zip_file: Объект ZipFile с DICOM файлами

    Returns:
        list: Список DICOM объектов (срезы из серии с максимальным количеством изображений)
              или пустой список, если не удалось прочитать файлы
    """
    series_dict = defaultdict(list)

    # Группируем все серии
    for file_name in zip_file.namelist():
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
        return []

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


def search_number_axial_slice(detections, image_width=512):
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
    number_axial_slice_list.append(number_axial_slice)
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
                               (0, 0, 255), 2)
    return annotated_image


def create_segmentations_masks(axial_segmentations, img_size=512):
    # Цвета для каждого класса
    # Цвета для каждого класса
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
        import torch
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


def create_segmentations_masks_full(segmentation_masks_image, axial_slice_norm_body, ribs_annotated_image):
    # Параметры для текста
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5
    font_color = (255, 255, 255)  # белый цвет
    thickness = 1

    # Создаем список изображений в нужном порядке с их названиями
    images_with_labels = [
        (ribs_annotated_image, "ribs_annotated_image"),
        (axial_slice_norm_body, "axial_slice_norm_body")
    ]

    # Добавляем сегментационные маски из словаря
    for key, image in segmentation_masks_image.items():
        images_with_labels.append((image, key))

    # Список для хранения изображений с текстом
    labeled_images = []

    # Добавляем текст к каждому изображению
    for image, label in images_with_labels:
        # Создаем копию изображения, чтобы не изменять оригинал
        labeled_img = image.copy()

        # Получаем размеры изображения
        height, width = labeled_img.shape[:2]

        # Вычисляем позицию текста (центрируем)
        text_size = cv2.getTextSize(label, font, font_scale, thickness)[0]
        text_x = (width - text_size[0]) // 2
        text_y = (height + text_size[1]) // 2

        # Добавляем текст на изображение
        cv2.putText(labeled_img, label, (text_x, text_y), font,
                    font_scale, font_color, thickness, cv2.LINE_AA)

        labeled_images.append(labeled_img)

    # Определяем размеры сетки
    num_images = len(labeled_images)
    if num_images <= 2:
        rows, cols = 1, num_images
    elif num_images <= 4:
        rows, cols = 2, 2
    elif num_images <= 6:
        rows, cols = 2, 3
    else:
        # Для большого количества изображений можно добавить больше вариантов
        rows, cols = 2, (num_images + 1) // 2

    # Получаем размеры первого изображения (предполагаем, что все одинакового размера)
    img_height, img_width = labeled_images[0].shape[:2]

    # Создаем пустое изображение для результата
    result = numpy.zeros((img_height * rows, img_width * cols, 3), dtype=numpy.uint8)

    # Заполняем сетку изображениями
    for i in range(rows):
        for j in range(cols):
            idx = i * cols + j
            if idx < len(labeled_images):
                result[i * img_height:(i + 1) * img_height,
                j * img_width:(j + 1) * img_width] = labeled_images[idx]

    # Показываем изображение
    cv2.namedWindow('Segmentation Masks', cv2.WINDOW_NORMAL)
    cv2.imshow("Segmentation Masks", result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def create_segmentation_results_cnt(axial_detections):
    """"""
    pass


def create_answer():
    pass
