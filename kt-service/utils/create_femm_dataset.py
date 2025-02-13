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

# Путь к вашему файлу DICOM

files_path = '/media/msi/fsi/fsi/222/CT/LEONOVA-N-V/1.2.156.14702.1.1000.16.0.20170602082848921/1.2.156.14702.1.1000.16.1.2017060208323343700040001'
files_path = '/media/msi/fsi/fsi/datasets_mrt/MosMedData-CT-COVID19-type VII-v 1/dicom/patient_001/IOP.870'

config = toml.load(os.path.join(os.getcwd(), os.pardir, 'ai_fsi_config.toml'))


def check_work_folders(path):
    """"""
    if not os.path.exists(path):
        os.makedirs(path)
        print("Created save directories")


def get_files_path(config):
    """

    Args:
        files_path:

    Returns:

    """
    # os.path.join(os.getcwd(), os.pardir,config['main_settings'][''])
    # files_path = config['']['']
    file_path_list = []
    files_path = '/media/msi/fsi/fsi/datasets_mrt/MosMedData-CT-COVID19-type VII-v 1/dicom/patient_001/IOP.870'

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


def get_rescale_intercept():
    pass


def rescale_slope():
    pass


def get_hu(pixel_value, rescale_intercept=-1024.0, rescale_slope=1.0):
    """
    Функция для вычисления HU из значения пикселей dicom-файла

    Формула взята отсюда https://stackoverflow.com/questions/22991009/how-to-get-hounsfield-units-in-dicom-file-
    using-fellow-oak-dicom-library-in-c-sh

    Краткое справка приведена в начале скрипта

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


def remove_small_regions(image, min_size=5):
    # Создаем копию изображения, чтобы не изменять оригинал
    result_image = image.copy()

    # Получаем размеры изображения
    height, width, _ = image.shape

    # Проходим по каждому пикселю изображения
    for y in range(height):
        for x in range(width):
            # Получаем цвет текущего пикселя
            current_color = image[y, x]

            # Если цвет не черный (предполагаем, что черный - это фон)
            if not numpy.array_equal(current_color, [0, 0, 0]):
                # Проверяем размер области вокруг текущего пикселя
                region_size = 0
                for dy in range(-min_size // 2, min_size // 2 + 1):
                    for dx in range(-min_size // 2, min_size // 2 + 1):
                        ny, nx = y + dy, x + dx
                        if 0 <= ny < height and 0 <= nx < width:
                            if numpy.array_equal(image[ny, nx], current_color):
                                region_size += 1

                # Если размер области меньше min_size x min_size, закрашиваем соседним цветом
                if region_size < min_size * min_size:
                    # Ищем цвет соседних пикселей
                    for dy in range(-1, 2):
                        for dx in range(-1, 2):
                            ny, nx = y + dy, x + dx
                            if 0 <= ny < height and 0 <= nx < width:
                                if not numpy.array_equal(image[ny, nx], [0, 0, 0]):
                                    result_image[y, x] = image[ny, nx]
                                    break
                        else:
                            continue
                        break

    return result_image

def  create_femm_mask_file(mask_list, color_output, img_normalized, final_output, pixel_spacing, file_name):
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
    scale_factors = numpy.array([pixel_spacing[0], pixel_spacing[1]])
    classes_list = {'bone': '0', 'muscles': '1', 'lung': 2, 'adipose': 3}
    for msk in mask_list:
        class_msk = classes_list[msk[-1]]
        msk_contours, hierarchy = cv2.findContours(msk[0],
                                                   cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        for cnt in msk_contours:
            scaled_contours_str = ""
            print(len(cnt))
            cnt = contour_approximate(cnt)
            print(len(cnt))

            scaled_contour = (cnt * scale_factors)

            cv2.drawContours(img_normalized, cnt, -1, (0, 255, 0), -1)
            pts = cnt.reshape((-1, 1, 2))
            cv2.polylines(img_normalized, [pts], True, (0, 255, 255))
            img_normalized_n = cv2.resize(img_normalized, (1000, 1000))
            # cv2.namedWindow('img_normalized', cv2.WINDOW_NORMAL)
            # cv2.imshow('img_normalized', img_normalized_n)
            # cv2.waitKey(0)
            for point in scaled_contour:
                scaled_contours_str += f"{int(point[0][0])} {int(point[0][1])} "

            path_to_save_labels = f'../save_test_masks/{file_name}/'

            check_work_folders(os.path.join(os.getcwd(), path_to_save_labels))

            check_labels = os.path.exists(f'{path_to_save_labels}{file_name}.txt')

            if not check_labels:
                with open(f'{path_to_save_labels}/{file_name}.txt', "w") as file:
                    file.write(f'{class_msk} {scaled_contours_str}' "\n")
            else:
                with open(f'{path_to_save_labels}/{file_name}.txt', "a") as file:
                    file.seek(0, 2)  # перемещение курсора в конец файла
                    file.write(f'{class_msk} {scaled_contours_str}' "\n")

    cv2.imwrite(f'{path_to_save_labels}{file_name}_color_output.jpg', cv2.resize(color_output, (1000, 1000)))
    cv2.imwrite(f'{path_to_save_labels}{file_name}_img_normalized.jpg', cv2.resize(img_normalized, (1000, 1000)))
    cv2.imwrite(f'{path_to_save_labels}{file_name}_final_output.jpg', cv2.resize(final_output, (1000, 1000)))

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


def clear_color_output(only_body_mask, color_output):
    """"""
    mask_organs_processed = color_output.copy()
    zeros_in_organs_mask = numpy.where((color_output == [0, 0, 0]).all(axis=2))
    body_contour_filter = only_body_mask[zeros_in_organs_mask[0], zeros_in_organs_mask[1]] == 255
    filtered_zeros = (zeros_in_organs_mask[0][body_contour_filter],
                      zeros_in_organs_mask[1][body_contour_filter])
    mask_organs_processed[filtered_zeros] = (0, 0, 255)

    return mask_organs_processed


def create_femm_dataset(config):
    """"""
    file_path_list = [
        '/media/msi/fsi/fsi/datasets_mrt/MosMedData-CT-COVID19-type VII-v 1/dicom/patient_001/study_03_2020-03-11/IOP.870']
    # file_path_list = get_files_path(config)
    for dcm_file in tqdm(file_path_list):
        file_name = basename(dcm_file).split('.dcm')[0]
        dir_name = dcm_file.split('/')[-2]
        dir_name = dir_name.replace('.', '_')
        file_name = file_name.replace('.', '_')
        ds = pydicom.dcmread(dcm_file)
        for i in ds:
            print(i)
        new_image = ds.pixel_array
        hu_img = numpy.vectorize(get_hu)(new_image).astype(numpy.int16)  # Используем int16, чтобы сохранить диапазон HU
        only_body_mask = get_only_body_mask(hu_img)  # Ищем маску тела
        only_body_hu_img = cv2.bitwise_and(hu_img, hu_img, mask=only_body_mask)  # Выделяем тело в изображении HU
        hu_img = cv2.GaussianBlur(only_body_hu_img, (5, 5), 0)
        # cv2.namedWindow('masked_image', cv2.WINDOW_NORMAL)
        # cv2.imshow('masked_image', masked_image)

        hu_ranges = {
            "Воздух": ([-1100, -200], (255, 150, 0)),
            "Кость": ([90, 800], (255, 255, 255)),
            "Мышца": ([1, 70], (0, 0, 255)),
            "Жир": ([-150, -1], (0, 255, 255))  # Светло-зелёный
            # Коричневый
        }

        # 2. Создайте исходное изображение для цветового вывода (в цветном формате)
        color_output = numpy.zeros((hu_img.shape[0], hu_img.shape[1], 3), dtype=numpy.uint8)

        for label, (hu_range, color) in hu_ranges.items():
            min_hu, max_hu = hu_range
            mask = numpy.logical_and(hu_img >= min_hu, hu_img <= max_hu)
            mask = mask.astype(numpy.uint8)
            if label == 'Кость':
                kernel = numpy.ones((3, 3), numpy.uint8)
                # mask = cv2.erode(mask, kernel, iterations=1)
                # mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
                contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                areas = [cv2.contourArea(contour) for contour in contours]
                mean_area = numpy.mean(areas)
                threshold_area = mean_area * (1 - 0.1)
                filtered_contours = [contour for contour, area in zip(contours, areas) if area >= threshold_area]
                mask = numpy.zeros([512, 512])
                cv2.drawContours(mask, filtered_contours, -1, 255, thickness=cv2.FILLED)
            elif label == "Мышца":
                kernel = numpy.ones((5, 5), numpy.uint8)
                mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
                contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                areas = [cv2.contourArea(contour) for contour in contours]
                mean_area = numpy.mean(areas)
                threshold_area = mean_area * (1 - 0.1)
                filtered_contours = [contour for contour, area in zip(contours, areas) if area >= threshold_area]
                mask = numpy.zeros([512, 512])
                cv2.drawContours(mask, filtered_contours, -1, 255, thickness=cv2.FILLED)

                # mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
                # mask = cv2.dilate(mask, kernel, iterations=1)
            elif label == "Жир":
                kernel = numpy.ones((3, 3), numpy.uint8)
                mask = cv2.dilate(mask, kernel, iterations=1)
                contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(mask, contours, -1, 255, thickness=cv2.FILLED)

                # mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
            elif label == "Воздух":
                kernel = numpy.ones((5, 5), numpy.uint8)
                mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
                contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(mask, contours, -1, 255, thickness=cv2.FILLED)
            else:
                pass
            # mask = cv2.dilate(mask, kernel, iterations=1)

            color_output[numpy.logical_and(mask, numpy.all(color_output == (0, 0, 0), axis=2))] = color

            cv2.namedWindow('color_output', cv2.WINDOW_NORMAL)
            cv2.imshow('color_output', color_output)
            cv2.waitKey(0)

        # Визуализируйте результат

        # color_output2 = cv2.dilate(color_output, kernel, iterations=1)
        # opening = cv2.morphologyEx(color_output, cv2.MORPH_OPEN, kernel)
        # closing = cv2.morphologyEx(color_output, cv2.MORPH_CLOSE, kernel)
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
        color_output = color_output.astype(numpy.uint8)

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
        color_output = cv2.flip(color_output, 0)

        pixel_spacing = get_pixel_spacing(
            ds)  # Получаем инфу о количестве миллиметров в пикселе -> (0.753906, 0.753906)

        create_femm_mask_file([mask_muscles, mask_bone, mask_lung, mask_adipose], color_output, img_normalized,
                              final_output, pixel_spacing, file_name)


if __name__ == '__main__':
    create_femm_dataset(config)
