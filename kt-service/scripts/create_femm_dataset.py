"""Скрипт для создания масок для ЭИТ из .dicom файлов

Справка для понимания

Глубина цвета (битность):
В DICOM глубина цвета может варьироваться в зависимости от типа медицинского изображения. Обычно используется 8, 10,
12, 16 или даже 32 бита на пиксель. Например, для КТ и МРТ часто используется 12 или 16 бит на пиксель, что позволяет
хранить значения в диапазоне 0–4095 (12 бит) или 0–65535 (16 бит).
В отличие от стандартных изображений (где диапазон пикселей обычно 0–255 для 8-битных изображений),
DICOM-файлы могут иметь гораздо больший диапазон значений. Это связано с тем, что медицинские изображения должны точно
передавать тонкие различия в плотности тканей или интенсивности сигнала. Например, в КТ-изображениях значения пикселей
представляют собой числа Хаунсфилда (Hounsfield Units, HU), которые могут быть отрицательными (для воздуха или жира)
и положительными (для костей или контрастных веществ). Диапазон HU обычно находится в пределах от -1000 до +3000.
"""

import cv2
import numpy
import os
import pydicom
import toml

from tqdm import tqdm
from os.path import basename

from sklearn.cluster import KMeans

# Путь к вашему файлу DICOM

path_to_save_image_log = f'../../save_test_masks/'


def check_work_folders(path):
    """"""
    if not os.path.exists(path):
        os.makedirs(path)
        print("Created save directories")


def get_files_path(config=None):
    """

    Args:
        files_path:

    Returns:

    """
    # os.path.join(os.getcwd(), os.pardir,config['main_settings'][''])
    # files_path = config['']['']
    file_path_list = []
    # files_path = '/media/msi/fsi/fsi/datasets_mrt/svalka/new/FT_CT_LUNGCR/CT_LUNGCR/1.2.643.5.1.13.13.12.2.77.8252.00060202040815091514020315000411/1.2.643.5.1.13.13.12.2.77.8252.13090901090012030215010003150408'
    files_path = '/media/msi/fsi/fsi/test'

    for dir, _, files in os.walk(files_path):
        # Обрабатываем каждый файл в текущей папке
        for file in files:
            # Собираем путь к файлу (без проверки расширения, как ранее)
            file_path = os.path.join(dir, file)
            file_path_list.append(file_path)
    return file_path_list


def get_pixels_hu(scan):
    image = numpy.stack(scan.pixel_array)

    # Convert to int16 (from sometimes int16),
    # should be possible as values should always be low enough (<32k)
    image = image.astype(numpy.int16)

    # Set outside-of-scan pixels to 1
    # The intercept is usually -1024, so air is approximately 0
    image[image == -2000] = 0

    # Convert to Hounsfield units (HU)
    intercept = scan.RescaleIntercept
    slope = scan.RescaleSlope

    if slope != 1:
        image = slope * image.astype(numpy.float64)
        image = image.astype(numpy.int16)

    image += numpy.int16(intercept)

    return numpy.array(image, dtype=numpy.int16)


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


def apply_hu_to_image(img):
    # Читаем изображение в grayscale режиме

    # Применяем функцию ко всем пикселям используя векторизацию NumPy
    hu_img = numpy.vectorize(get_hu)(img).astype(numpy.int16)  # Используем int16, чтобы сохранить диапазон HU
    # for i in hu_img:
    #     print(i)# Сохраняем результат
    # cv2.imwrite('output_hu_image.png', hu_img)
    return hu_img


def get_mask_bone(color_output):
    """"""
    lower_red = numpy.array([254, 254, 254])  # Нижний диапазон красного (чуть ниже, чтобы поймать вариации)
    upper_red = numpy.array([255, 255, 255])  # Верхний диапазон красного (чуть выше, чтобы поймать вариации)
    mask_bone = cv2.inRange(color_output, lower_red, upper_red)
    return mask_bone, 'bone'


def get_mask_muscles(color_output):
    """"""
    lower_red = numpy.array([0, 0, 254])  # Нижний диапазон красного (чуть ниже, чтобы поймать вариации)
    upper_red = numpy.array([0, 0, 255])  # Верхний диапазон красного (чуть выше, чтобы поймать вариации)
    mask_muscles = cv2.inRange(color_output, lower_red, upper_red)
    return mask_muscles, 'muscles'


def get_mask_lung(color_output):
    """"""
    lower_red = numpy.array([0, 254, 254])  # Нижний диапазон красного (чуть ниже, чтобы поймать вариации)
    upper_red = numpy.array([0, 255, 255])  # Верхний диапазон красного (чуть выше, чтобы поймать вариации)
    mask_lung = cv2.inRange(color_output, lower_red, upper_red)
    return mask_lung, 'lung'


def get_mask_adipose(color_output):
    """"""
    lower_red = numpy.array([254, 149, 0])  # Нижний диапазон красного (чуть ниже, чтобы поймать вариации)
    upper_red = numpy.array([255, 150, 0])  # Верхний диапазон красного (чуть выше, чтобы поймать вариации)
    mask_adipose = cv2.inRange(color_output, lower_red, upper_red)
    return mask_adipose, 'adipose'


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


def contour_approximate(contour):
    """"""
    # Задайте параметр epsilon, который определяет точность аппроксимации
    # Чем меньше epsilon, тем ближе аппроксимированный контур к исходному
    epsilon = 0.00001 * cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, epsilon, True)
    return approx


def find_point_conductivity(contour):
    """
    Находит точку внутри контура, близкую к центру.

    Параметры:
    - contour: контур из cv2.findContours() в формате (N, 1, 2)

    Возвращает:
    - (x, y): координаты точки внутри контура (int, int)
    """
    # Преобразуем контур в формат (N, 2) для cv2.moments()
    contour_reshaped = contour.reshape(-1, 2)

    # Вариант 1: Центр масс (центроид) контура
    M = cv2.moments(contour_reshaped)
    if M["m00"] != 0:
        x = int(M["m10"] / M["m00"])
        y = int(M["m01"] / M["m00"])
    else:
        # Если контур вырожден (например, линия), берём первую точку
        x, y = contour[0][0]

    # Проверяем, что точка действительно внутри контура
    # Важно: pointPolygonTest ожидает (float, float)!
    if cv2.pointPolygonTest(contour, (float(x), float(y)), False) >= 0:
        return x, y

    # Вариант 2: Если центроид снаружи, берём случайную точку внутри bounding box
    x_rect, y_rect, w_rect, h_rect = cv2.boundingRect(contour)
    for _ in range(100):
        rand_x = numpy.random.randint(x_rect, x_rect + w_rect)
        rand_y = numpy.random.randint(y_rect, y_rect + h_rect)
        if cv2.pointPolygonTest(contour, (float(rand_x), float(rand_y)), False) >= 0:
            return rand_x, rand_y

    # Если ничего не нашли, возвращаем первую точку контура
    return int(contour[0][0][0]), int(contour[0][0][1])

def create_femm_mask_file(mask_list, color_output, img_normalized, final_output, pixel_spacing, file_name):
    """

    Args:
        mask_list:
        color_output:
        img_normalized:
        final_output:
        pixel_spacing:
        file_name:

    Returns:

    """
    img_normalized_clear = img_normalized.copy()
    scale_factors = numpy.array([pixel_spacing[0], pixel_spacing[1]])
    classes_list = {'bone': '0', 'muscles': '1', 'lung': 2, 'adipose': 3}
    for msk in mask_list:
        class_msk = classes_list[msk[-1]]
        msk_contours, hierarchy = cv2.findContours(msk[0],
                                                   cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        for cnt in msk_contours:
            x_point_conductivity, y_point_conductivity = find_point_conductivity(cnt)

            scaled_contours_str = ""
            # cnt = contour_approximate(cnt)

            scaled_contour = (cnt * 1)
            cv2.drawContours(img_normalized, cnt, -1, (0, 255, 0), -1)
            # cv2.polylines(img_normalized, [pts], True, (0, 255, 255))
            # img_normalized_n = cv2.resize(img_normalized, (1000, 1000))
            # cv2.namedWindow('img_normalized', cv2.WINDOW_NORMAL)
            # cv2.imshow('img_normalized', img_normalized_n)
            # cv2.waitKey(0)
            for point in scaled_contour:
                scaled_contours_str += f"{float(point[0][0])} {float(point[0][1])} "

            path_to_save_labels = f'../../save_test_masks/{file_name}/'

            check_work_folders(os.path.join(os.getcwd(), path_to_save_labels))

            check_labels = os.path.exists(f'{path_to_save_labels}{file_name}.txt')

            if not check_labels:
                with open(f'{path_to_save_labels}/{file_name}.txt', "w") as file:
                    file.write(f'{class_msk} {scaled_contours_str}' "\n")
            else:
                with open(f'{path_to_save_labels}/{file_name}.txt', "a") as file:
                    file.seek(0, 2)  # перемещение курсора в конец файла
                    file.write(f'{class_msk} {scaled_contours_str}' "\n")

    cv2.imwrite(f'{path_to_save_labels}10_{file_name}_img_normalized.jpg', cv2.resize(img_normalized, (1000, 1000)))
    cv2.imwrite(f'{path_to_save_labels}{file_name}_final_output.jpg', cv2.resize(final_output, (1000, 1000)))
    cv2.imwrite(f'{path_to_save_labels}{file_name}.jpg', img_normalized_clear)

    return None


def get_only_body_mask(hu_img):
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


def clear_color_output(only_body_mask, color_output, tolerance=5):
    mask_organs_processed = color_output.copy()

    # Поиск почти черных пикселей с допуском
    is_black = numpy.all(numpy.abs(color_output - [0, 0, 0]) <= tolerance, axis=2)
    black_coords = numpy.where(is_black)

    # Фильтрация по маске тела (only_body_mask == 255)
    is_in_body = only_body_mask[black_coords] == 255
    filtered_coords = (black_coords[0][is_in_body], black_coords[1][is_in_body])

    # Замена на красный (BGR-формат: (0, 0, 255))
    mask_organs_processed[filtered_coords] = [0, 0, 255]

    return mask_organs_processed


def spine_bone_search(only_bone_hu_img):
    """

    Args:
        only_bone_hu_img:

    Returns:

    """
    print(only_bone_hu_img)
    only_spine_bone_hu_img = []

    return only_spine_bone_hu_img


def get_max_bone_mask(mask):
    """

    Returns:

    """
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Найдите контур с максимальной площадью
    max_area = 0
    largest_contour = None

    for contour in contours:
        area = cv2.contourArea(contour)
        if area > max_area:
            max_area = area
            largest_contour = contour

    # Создайте пустое изображение для выделения самой большой кости
    largest_bone_mask = numpy.zeros_like(mask)

    # Нарисуйте самый большой контур на пустом изображении
    cv2.drawContours(largest_bone_mask, [largest_contour], -1, 255, thickness=cv2.FILLED)
    cv2.imshow('Largest Bone Mask', largest_bone_mask)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def mask_filling(mask):
    # Найдите контуры на маске
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # Пройдитесь по каждому контуру
    for contour in contours:
        # Создайте маску для текущего контура
        contour_mask = numpy.zeros_like(mask)
        cv2.drawContours(contour_mask, [contour], -1, 255, -1)

        # Найдите разницу между маской контура и исходной маской
        difference = cv2.bitwise_and(contour_mask, cv2.bitwise_not(mask))

        # Если есть незакрашенное пространство внутри контура
        if cv2.countNonZero(difference) > 0:
            # Закрасьте полигон белым цветом
            cv2.drawContours(mask, [contour], -1, (255, 255, 255), -1)

    return mask


def filter_contours_by_area(contours, min_area):
    """
    Фильтрует контуры по минимальной площади

    Параметры:
    contours -- список контуров (каждый контур - массив точек формата numpy array)
    min_area -- минимальная площадь контура для сохранения

    Возвращает:
    Список контуров, у которых площадь >= min_area
    """
    filtered_contours = []

    for contour in contours:
        area = cv2.contourArea(contour)
        if area >= min_area:
            filtered_contours.append(contour)

    return filtered_contours

def create_bone_mask(mask, hu_img, color_output_all, file_name, color):
    color_output_bone = numpy.zeros((hu_img.shape[0], hu_img.shape[1], 3), dtype=numpy.uint8)
    # kernel = numpy.ones((5, 5), numpy.uint8)
    # mask = cv2.erode(mask, kernel, iterations=1)
    # mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    filter_contours_by_area(contours, 5)

    # areas = [cv2.contourArea(contour) for contour in contours]
    # mean_area = numpy.mean(areas)
    # threshold_area = mean_area * (1 - 0)
    # print(threshold_area)
    # filtered_contours = [contour for contour, area in zip(contours, areas) if area >= threshold_area]
    mask_bone = numpy.zeros((hu_img.shape[0], hu_img.shape[1]), dtype=numpy.uint8)
    cv2.drawContours(mask_bone, contours, -1, 255, thickness=-1)
    mask_bone = mask_filling(mask_bone)
    color_output_bone[numpy.logical_and(mask_bone, numpy.all(color_output_bone == (0, 0, 0), axis=2))] = color
    color_output_all[numpy.logical_and(mask_bone, numpy.all(color_output_all == (0, 0, 0), axis=2))] = color
    cv2.imwrite(f'{path_to_save_image_log}{file_name}/4_color_output_bone.jpg', color_output_bone)
    return color_output_all


def create_muscles_mask(mask, hu_img, color_output_all, file_name, color):
    color_output_muscles = numpy.zeros((hu_img.shape[0], hu_img.shape[1], 3), dtype=numpy.uint8)
    kernel = numpy.ones((5, 5), numpy.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    filter_contours_by_area(contours, 5)
    areas = [cv2.contourArea(contour) for contour in contours]
    mean_area = numpy.mean(areas)
    threshold_area = mean_area * (1 - 0.1)
    filtered_contours = [contour for contour, area in zip(contours, areas) if area >= threshold_area]
    mask_muscles = numpy.zeros((hu_img.shape[0], hu_img.shape[1]), dtype=numpy.uint8)
    cv2.drawContours(mask_muscles, filtered_contours, -1, 255, thickness=cv2.FILLED)
    mask_muscles = mask_filling(mask_muscles)
    color_output_muscles[numpy.logical_and(mask_muscles, numpy.all(color_output_muscles == (0, 0, 0), axis=2))] = color
    color_output_all[numpy.logical_and(mask_muscles, numpy.all(color_output_all == (0, 0, 0), axis=2))] = color
    cv2.imwrite(f'{path_to_save_image_log}{file_name}/5_color_output_muscles.jpg', color_output_muscles)
    # mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    # mask = cv2.dilate(mask, kernel, iterations=1)
    return color_output_all


def create_lung_mask(mask, hu_img, color_output_all, file_name, color):
    color_output_lung = numpy.zeros((hu_img.shape[0], hu_img.shape[1], 3), dtype=numpy.uint8)
    kernel = numpy.ones((3, 3), numpy.uint8)
    # mask = cv2.dilate(mask, kernel, iterations=1)
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    filter_contours_by_area(contours, 5)
    mask_lung = numpy.zeros((hu_img.shape[0], hu_img.shape[1]), dtype=numpy.uint8)
    cv2.drawContours(mask_lung, contours, -1, 255, thickness=cv2.FILLED)
    color_output_lung[numpy.logical_and(mask_lung, numpy.all(color_output_lung == (0, 0, 0), axis=2))] = color
    color_output_all[numpy.logical_and(mask_lung, numpy.all(color_output_all == (0, 0, 0), axis=2))] = color
    cv2.imwrite(f'{path_to_save_image_log}{file_name}/6_color_output_lung.jpg', color_output_lung)
    # mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    return color_output_all


def crerate_adipose_mask(mask, hu_img, color_output_all, file_name, color):
    color_output_adipose = numpy.zeros((hu_img.shape[0], hu_img.shape[1], 3), dtype=numpy.uint8)
    kernel = numpy.ones((5, 5), numpy.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    mask_adipose = numpy.zeros((hu_img.shape[0], hu_img.shape[1]), dtype=numpy.uint8)
    cv2.drawContours(mask_adipose, contours, -1, 255, thickness=cv2.FILLED)
    mask_adipose = mask_filling(mask_adipose)
    color_output_adipose[
        numpy.logical_and(mask_adipose, numpy.all(color_output_adipose == (0, 0, 0), axis=2))] = color
    color_output_all[
        numpy.logical_and(mask_adipose, numpy.all(color_output_all == (0, 0, 0), axis=2))] = color
    cv2.imwrite(f'{path_to_save_image_log}{file_name}/7_color_output_adipose.jpg', color_output_adipose)
    return color_output_all


def highlight_small_masks(image, area_threshold=5):
    """"""
    # Цвета масок (в BGR, так как OpenCV использует BGR, а не RGB)
    mask_colors = {
        "bone": (255, 255, 255),  # Белый (кость)
        "muscle": (0, 0, 255),  # Красный (мышца)
        "fat": (0, 255, 255),  # Жёлтый (жир)
        "air": (0, 150, 255),  # Оранжевый (воздух)
    }

    # Жёлтый цвет для заливки маленьких масок (BGR)
    YELLOW = (0, 255, 255)

    # Проходимся по каждому цвету маски
    for tissue, target_color in mask_colors.items():
        # Создаём маску для текущего цвета (допускаем небольшие вариации)
        lower = numpy.array(target_color, dtype=numpy.int16) - 10
        upper = numpy.array(target_color, dtype=numpy.int16) + 10
        mask = cv2.inRange(image, lower, upper)

        # Находим контуры
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Закрашиваем маленькие маски жёлтым
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < area_threshold:
                cv2.drawContours(image, [cnt], -1, YELLOW, thickness=cv2.FILLED)

    return image

def create_femm_dataset():
    """"""
    file_path_list = [
        '/media/msi/fsi/fsi/test']
    file_path_list = get_files_path()
    bone_mask = []
    for dcm_file in tqdm(file_path_list):
        file_name = basename(dcm_file).split('.dcm')[0]

        dir_name = dcm_file.split('/')[-2]
        dir_name = dir_name.replace('.', '_')
        file_name = file_name.replace('.', '_')
        ds = pydicom.dcmread(dcm_file)
        for i in ds:
            print(i)
        new_image = ds.pixel_array
        new_image = numpy.flipud(new_image)

        rescale_slope = get_rescale_slope(ds)
        rescale_intercept = get_rescale_intercept(ds)
        # Получаем оригинальную картинку с глубиной HU
        hu_img_orig = numpy.vectorize(get_hu, excluded=['rescale_intercept', 'rescale_slope']) \
            (new_image, rescale_intercept, rescale_slope).astype(
            numpy.int16)  # Используем int16, чтобы сохранить диапазон HU


        # Ищем маску тела
        only_body_mask = get_only_body_mask(hu_img_orig)
        check_work_folders(os.path.join(os.getcwd(), f'{path_to_save_image_log}{file_name}'))

        cv2.imwrite(f'{path_to_save_image_log}{file_name}/1_only_body_mask.jpg', only_body_mask)
        # Получаем оригинальную картинку только с телом (глубиной HU)
        only_body_hu_img = cv2.bitwise_and(hu_img_orig, hu_img_orig,
                                           mask=only_body_mask)  # Выделяем тело в изображении HU
        cv2.imwrite(f'{path_to_save_image_log}{file_name}/2_only_body_hu_img.jpg', only_body_hu_img)
        # Применяем небольшое размытие для полученного изображения
        hu_img = cv2.GaussianBlur(only_body_hu_img, (5, 5), 0)
        cv2.imwrite(f'{path_to_save_image_log}{file_name}/3_hu_img_blur.jpg', hu_img)

        hu_ranges = {
            "Воздух": ([-1100, -200], (255, 150, 0)),
            "Кость": ([70, 800], (255, 255, 255)),
            "Мышца": ([1, 50], (0, 0, 255)),
            "Жир": ([-150, -1], (0, 255, 255))  # Светло-зелёный
            # Коричневый
        }

        # Создаём исходное изображение для цветового вывода (в цветном формате)
        color_output_all = numpy.zeros((hu_img.shape[0], hu_img.shape[1], 3), dtype=numpy.uint8)
        for label, (hu_range, color) in hu_ranges.items():
            min_hu, max_hu = hu_range
            mask = numpy.logical_and(hu_img >= min_hu, hu_img <= max_hu)
            mask = mask.astype(numpy.uint8)
            if label == 'Кость':
                color_output_all = create_bone_mask(mask, hu_img, color_output_all, file_name, color)
            elif label == "Мышца":
                color_output_all = create_muscles_mask(mask, hu_img, color_output_all, file_name, color)
            elif label == "Жир":
                color_output_all = create_lung_mask(mask, hu_img, color_output_all, file_name, color)
            elif label == "Воздух":
                color_output_all = crerate_adipose_mask(mask, hu_img, color_output_all, file_name, color)
            else:
                continue

        cv2.imwrite(f'{path_to_save_image_log}{file_name}/8_color_output_all.jpg', color_output_all)

        if new_image.dtype != numpy.uint8:
            # Нормализация в диапазон 0-255, сохраняя относительную яркость пикселей
            img_normalized = (new_image - new_image.min()) / (new_image.max() - new_image.min()) * 255.0
            img_normalized = img_normalized.astype(numpy.uint8)
        else:
            img_normalized = new_image
        # img_normalized = cv2.drawContours(img_normalized, contours, -1, (0, 255, 0), 1)

        # Наложение изображений с прозрачностью

        # color_output2 = cv2.resize(color_output, (1000,1000))
        alpha_channel = img_normalized.astype(numpy.float32) / 255.0
        # Убедиться, что color_output имеет тип uint8 (стандартный тип для изображений в OpenCV)
        color_output = color_output_all.astype(numpy.uint8)

        # Добавить альфа-канал к color_output (теперь будет 4 канала: RGBA)
        rgba_output = cv2.merge([color_output, (alpha_channel * 255).astype(numpy.uint8)])

        # Наложение изображения с альфа-каналом на белый фон (или любой другой фон, который вы предпочитаете)
        white_background = numpy.full((512, 512, 3), 255, dtype=numpy.uint8)
        final_output = (
                white_background * (1 - alpha_channel[:, :, numpy.newaxis]) + rgba_output[:, :, :3] * alpha_channel[:,
                                                                                                      :,
                                                                                                      numpy.newaxis]).astype(
            numpy.uint8)

        mask_muscles = get_mask_muscles(color_output)
        mask_bone = get_mask_bone(color_output)
        mask_lung = get_mask_lung(color_output)
        mask_adipose = get_mask_adipose(color_output)
        color_output = clear_color_output(only_body_mask, color_output)
        color_output = highlight_small_masks(color_output)
        cv2.imwrite(f'{path_to_save_image_log}{file_name}/9_color_output.jpg', color_output)

        pixel_spacing = get_pixel_spacing(
            ds)  # Получаем инфу о количестве миллиметров в пикселе -> (0.753906, 0.753906)
        create_femm_mask_file([mask_muscles, mask_bone, mask_lung, mask_adipose], color_output, img_normalized,
                              final_output, pixel_spacing, file_name)


if __name__ == '__main__':
    create_femm_dataset()
