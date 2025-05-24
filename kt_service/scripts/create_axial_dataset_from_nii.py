import cv2
import os
import logging
import numpy as np
import sys
import nibabel as nib
from os.path import basename

from tqdm import tqdm
from scipy.ndimage import label as label_color

# Добавляем корень проекта в пути Python
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

dicom_dataset_dir = '/media/msi/fsi/fsi/datasets_mrt/svalka/COVID19_1110_neponytnye_failes/convert/all/'
path_to_save_image_log = f'../../save_test_masks/from_nii/'
path_to_save_dicom = f'../../save_test_masks/'

hu_ranges = {
    "Воздух": ([-1100, -200], (255, 255, 0)),
    "Кость": ([70, 800], (255, 255, 255)),
    "Мышца": ([1, 50], (0, 0, 255)),
    "Жир": ([-150, -1], (0, 255, 255))
}


def show_frontal_slices(volume, window_level=40, window_width=400):
    """
    Корректно отображает фронтальные (корональные) срезы из NIfTI-файла.

    Args:
        volume: 3D-массив данных (в HU)
        window_level: центр окна (по умолчанию 40 для мягких тканей)
        window_width: ширина окна (по умолчанию 400)
    """
    # Нормализация HU
    hu_min = window_level - window_width // 2
    hu_max = window_level + window_width // 2
    clipped = np.clip(volume, hu_min, hu_max)
    normalized = ((clipped - hu_min) / (hu_max - hu_min) * 255).astype(np.uint8)

    # Для NIfTI обычно нужно:
    # 1. Поменять порядок осей (чтобы фронтальные срезы были по второй оси)
    # 2. Возможно, сделать flip по некоторым осям

    # Переставляем оси для фронтальных срезов
    # (это эмпирическая настройка, может отличаться для разных сканеров)
    # frontal_slices = np.transpose(normalized, (0, 1, 2))
    # frontal_slices = np.flip(frontal_slices, axis=0)  # Часто требуется вертикальный flip

    # num_slices = frontal_slices.shape[2]
    # for i in range(num_slices):

    # Поворачиваем для правильной ориентации
    slice_img = cv2.rotate(normalized, cv2.ROTATE_90_CLOCKWISE)

    # Масштабируем для лучшего отображения
    display_img = cv2.resize(slice_img, (512, 512), interpolation=cv2.INTER_LINEAR)
    return display_img


def vignetting_image_normalization(new_image):
    """
    Виньетирование (отсечение крайних значений)
    DICOM-изображения могут содержать выбросы (очень тёмные/светлые пиксели), которые "сжимают" полезный диапазон.
    Args:
        new_image:

    Returns:

    """
    p_low, p_high = np.percentile(new_image, [2, 98])  # Отсекаем 2% самых тёмных и светлых пикселей
    img_clipped = np.clip(new_image, p_low, p_high)
    img_normalized = (img_clipped - p_low) / (p_high - p_low) * 255.0
    return img_normalized


def classic_norm(volume, window_level=40, window_width=400):
    """"""
    # Нормализация HU
    hu_min = window_level - window_width // 2
    hu_max = window_level + window_width // 2
    clipped = np.clip(volume, hu_min, hu_max)
    normalized = ((clipped - hu_min) / (hu_max - hu_min) * 255).astype(np.uint8)
    return normalized


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
    kernel_only_body_mask = np.ones((5, 5), np.uint8)
    only_body_mask = np.where((hu_img > -500) & (hu_img < 1000), 1, 0)
    only_body_mask = only_body_mask.astype(np.uint8)

    only_body_mask = cv2.morphologyEx(only_body_mask, cv2.MORPH_OPEN, kernel_only_body_mask)

    contours, hierarchy = cv2.findContours(only_body_mask,
                                           cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    max_contour = max(contours, key=cv2.contourArea, default=None)
    if max_contour is not None:
        only_body_mask = np.zeros_like(only_body_mask)
    cv2.drawContours(only_body_mask, [max_contour], 0, 255, -1)
    return only_body_mask


def check_work_folders(path):
    """"""
    if not os.path.exists(path):
        os.makedirs(path)
        print("Created save directories")


def mask_filling(mask):
    # Найдите контуры на маске
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # Пройдитесь по каждому контуру
    for contour in contours:
        # Создайте маску для текущего контура
        contour_mask = np.zeros_like(mask)
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
    color_output_bone = np.zeros((hu_img.shape[0], hu_img.shape[1], 3), dtype=np.uint8)
    # kernel = np.ones((5, 5), np.uint8)
    # mask = cv2.erode(mask, kernel, iterations=1)
    # mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    filter_contours_by_area(contours, 5)

    # areas = [cv2.contourArea(contour) for contour in contours]
    # mean_area = np.mean(areas)
    # threshold_area = mean_area * (1 - 0)
    # print(threshold_area)
    # filtered_contours = [contour for contour, area in zip(contours, areas) if area >= threshold_area]
    mask_bone = np.zeros((hu_img.shape[0], hu_img.shape[1]), dtype=np.uint8)
    cv2.drawContours(mask_bone, contours, -1, 255, thickness=-1)
    mask_bone = mask_filling(mask_bone)
    color_output_bone[np.logical_and(mask_bone, np.all(color_output_bone == (0, 0, 0), axis=2))] = color
    color_output_all[np.logical_and(mask_bone, np.all(color_output_all == (0, 0, 0), axis=2))] = color
    cv2.imwrite(f'{path_to_save_image_log}{file_name}/4_color_output_bone.jpg', color_output_bone)
    return color_output_all


def create_muscles_mask(mask, hu_img, color_output_all, file_name, color):
    color_output_muscles = np.zeros((hu_img.shape[0], hu_img.shape[1], 3), dtype=np.uint8)
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    filter_contours_by_area(contours, 5)
    areas = [cv2.contourArea(contour) for contour in contours]
    mean_area = np.mean(areas)
    threshold_area = mean_area * (1 - 0.1)
    filtered_contours = [contour for contour, area in zip(contours, areas) if area >= threshold_area]
    mask_muscles = np.zeros((hu_img.shape[0], hu_img.shape[1]), dtype=np.uint8)
    cv2.drawContours(mask_muscles, filtered_contours, -1, 255, thickness=cv2.FILLED)
    mask_muscles = mask_filling(mask_muscles)
    color_output_muscles[np.logical_and(mask_muscles, np.all(color_output_muscles == (0, 0, 0), axis=2))] = color
    color_output_all[np.logical_and(mask_muscles, np.all(color_output_all == (0, 0, 0), axis=2))] = color
    cv2.imwrite(f'{path_to_save_image_log}{file_name}/5_color_output_muscles.jpg', color_output_muscles)
    # mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    # mask = cv2.dilate(mask, kernel, iterations=1)
    return color_output_all


def crerate_adipose_mask(mask, hu_img, color_output_all, file_name, color):
    color_output_lung = np.zeros((hu_img.shape[0], hu_img.shape[1], 3), dtype=np.uint8)
    kernel = np.ones((3, 3), np.uint8)
    # mask = cv2.dilate(mask, kernel, iterations=1)
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    filter_contours_by_area(contours, 5)
    mask_lung = np.zeros((hu_img.shape[0], hu_img.shape[1]), dtype=np.uint8)
    cv2.drawContours(mask_lung, contours, -1, 255, thickness=cv2.FILLED)
    color_output_lung[np.logical_and(mask_lung, np.all(color_output_lung == (0, 0, 0), axis=2))] = color
    color_output_all[np.logical_and(mask_lung, np.all(color_output_all == (0, 0, 0), axis=2))] = color
    cv2.imwrite(f'{path_to_save_image_log}{file_name}/6_color_output_adipose.jpg', color_output_lung)
    # mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    return color_output_all


def create_lung_mask(mask, hu_img, color_output_all, file_name, color):
    color_output_adipose = np.zeros((hu_img.shape[0], hu_img.shape[1], 3), dtype=np.uint8)
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    mask_adipose = np.zeros((hu_img.shape[0], hu_img.shape[1]), dtype=np.uint8)
    cv2.drawContours(mask_adipose, contours, -1, 255, thickness=cv2.FILLED)
    mask_adipose = mask_filling(mask_adipose)
    color_output_adipose[
        np.logical_and(mask_adipose, np.all(color_output_adipose == (0, 0, 0), axis=2))] = color
    color_output_all[
        np.logical_and(mask_adipose, np.all(color_output_all == (0, 0, 0), axis=2))] = color
    cv2.imwrite(f'{path_to_save_image_log}{file_name}/7_color_output_lung.jpg', color_output_adipose)
    return color_output_all


def get_mask_bone(color_output):
    """"""
    lower_red = np.array([254, 254, 254])  # Нижний диапазон красного (чуть ниже, чтобы поймать вариации)
    upper_red = np.array([255, 255, 255])  # Верхний диапазон красного (чуть выше, чтобы поймать вариации)
    mask_bone = cv2.inRange(color_output, lower_red, upper_red)
    return mask_bone, 'bone'


def get_mask_muscles(color_output):
    """"""
    lower_red = np.array([0, 0, 254])  # Нижний диапазон красного (чуть ниже, чтобы поймать вариации)
    upper_red = np.array([0, 0, 255])  # Верхний диапазон красного (чуть выше, чтобы поймать вариации)
    mask_muscles = cv2.inRange(color_output, lower_red, upper_red)
    return mask_muscles, 'muscles'


def get_mask_adipose(color_output):
    """"""
    lower_red = np.array([0, 254, 254])  # Нижний диапазон красного (чуть ниже, чтобы поймать вариации)
    upper_red = np.array([0, 255, 255])  # Верхний диапазон красного (чуть выше, чтобы поймать вариации)
    mask_adipose = cv2.inRange(color_output, lower_red, upper_red)
    return mask_adipose, 'adipose'


def get_mask_lung(color_output):
    """"""
    lower_red = np.array([254, 254, 0])  # Нижний диапазон красного (чуть ниже, чтобы поймать вариации)
    upper_red = np.array([255, 255, 0])  # Верхний диапазон красного (чуть выше, чтобы поймать вариации)
    mask_lung = cv2.inRange(color_output, lower_red, upper_red)
    return mask_lung, 'lung'


def clear_color_output(only_body_mask, color_output, tolerance=5, min_polygon_size=5):
    mask_organs_processed = color_output.copy()
    h, w = mask_organs_processed.shape[:2]

    # 1. Закрашиваем почти чёрные пиксели внутри тела красным
    is_black = np.all(np.abs(color_output - [0, 0, 0]) <= tolerance, axis=2)
    is_in_body = (only_body_mask == 255)
    to_fill = is_black & is_in_body
    mask_organs_processed[to_fill] = [0, 0, 255]  # Красный в BGR

    # 2. Находим все связные области (полигоны), кроме фона (чёрного/красного)
    background_colors = [
        [0, 0, 0],  # Чёрный
        [0, 0, 255]  # Красный (уже закрашенные области)
    ]
    is_background = np.zeros((h, w), dtype=bool)
    for color in background_colors:
        is_background |= np.all(mask_organs_processed == color, axis=2)

    # Размечаем все связные области (каждый полигон получает уникальный label)
    labeled, num_features = label_color(~is_background)

    # 3. Проходим по всем полигонам и закрашиваем маленькие (<5 пикселей)
    for label_idx in range(1, num_features + 1):
        polygon_mask = (labeled == label_idx)
        polygon_size = np.sum(polygon_mask)

        if polygon_size < min_polygon_size:
            # Находим соседние цвета (игнорируя чёрный и красный)
            y, x = np.where(polygon_mask)
            neighbors = []

            # Проверяем 8-связных соседей для каждой точки полигона
            for dy, dx in [(-1, -1), (-1, 0), (-1, 1),
                           (0, -1), (0, 1),
                           (1, -1), (1, 0), (1, 1)]:
                ny, nx = y + dy, x + dx
                valid = (ny >= 0) & (ny < h) & (nx >= 0) & (nx < w)
                ny, nx = ny[valid], nx[valid]

                for color in mask_organs_processed[ny, nx]:
                    if not any(np.array_equal(color, bg_color) for bg_color in background_colors):
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
    mask_colors = {
        "bone": (255, 255, 255),
        "muscle": (0, 0, 255),
        "fat": (0, 255, 255),
        "air": (0, 150, 255),
    }

    output = image.copy()

    for tissue, target_color in mask_colors.items():
        lower = np.array(target_color, dtype=np.int16) - 10
        upper = np.array(target_color, dtype=np.int16) + 10
        mask = cv2.inRange(image, lower, upper)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for cnt in contours:
            if len(cnt) <= area_threshold:
                contour_mask = np.zeros(image.shape[:2], dtype=np.uint8)
                cv2.drawContours(contour_mask, [cnt], -1, 255, cv2.FILLED)

                dilated = cv2.dilate(contour_mask, np.ones((3, 3), np.uint8), iterations=1)
                neighbors_mask = dilated - contour_mask
                neighbor_colors = output[neighbors_mask == 255]

                if len(neighbor_colors) > 0:
                    neighbor_colors = [tuple(c) for c in neighbor_colors
                                       if not np.array_equal(c, target_color)
                                       and not np.array_equal(c, (0, 0, 0))]

                    if neighbor_colors:
                        # Use the most common neighbor color (as a tuple)
                        from collections import Counter
                        fill_color = Counter(neighbor_colors).most_common(1)[0][0]
                    else:
                        fill_color = target_color  # Fallback to original color
                else:
                    fill_color = target_color  # No neighbors: keep original

                # Ensure fill_color is a tuple of integers
                fill_color = tuple(map(int, fill_color))
                cv2.drawContours(output, [cnt], -1, fill_color, thickness=cv2.FILLED)

    return output


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
        rand_x = np.random.randint(x_rect, x_rect + w_rect)
        rand_y = np.random.randint(y_rect, y_rect + h_rect)
        if cv2.pointPolygonTest(contour, (float(rand_x), float(rand_y)), False) >= 0:
            return rand_x, rand_y

    # Если ничего не нашли, возвращаем первую точку контура
    return int(contour[0][0][0]), int(contour[0][0][1])


def abs_to_yolo(coord_str, image_shape):
    """Преобразует абсолютные координаты в YOLO-формат, замыкая контур, если нужно."""
    # Проверяем, замкнут ли контур (последняя точка == первой)
    first_point = coord_str[0][0]
    last_point = coord_str[-1][0]

    # Если контур не замкнут, добавляем первую точку в конец
    if (first_point != last_point).any():
        coord_str = np.append(coord_str, [[first_point]], axis=0)

    # Преобразуем в относительные координаты YOLO
    end_list = []
    for i in coord_str:
        xc = i[0][0] / image_shape[1]
        yc = i[0][1] / image_shape[0]
        end_list.append(xc)
        end_list.append(yc)

    return " ".join(str(x) for x in end_list)


def create_femm_mask_file(mask_list, color_output, img_normalized, final_output, file_name):
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
    classes_list = {'bone': '0', 'muscles': '1', 'lung': 2, 'adipose': 3}
    for msk in mask_list:
        class_msk = classes_list[msk[-1]]
        msk_contours, hierarchy = cv2.findContours(msk[0],
                                                   cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        for cnt in msk_contours:
            if len(cnt) < 5:
                continue
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
            scaled_contour = abs_to_yolo(scaled_contour, img_normalized_clear.shape)
            # print(scaled_contour)
            # for point in scaled_contour:
            #     scaled_contours_str += f"{float(point[0][0])} {float(point[0][1])} "

            path_to_save_labels = f'../../save_test_masks/from_nii/{file_name}/'

            check_work_folders(os.path.join(os.getcwd(), path_to_save_labels))

            check_labels = os.path.exists(f'{path_to_save_labels}{file_name}.txt')

            if not check_labels:
                with open(f'{path_to_save_labels}/{file_name}.txt', "w") as file:
                    file.write(f'{class_msk} {scaled_contour}' "\n")
            else:
                with open(f'{path_to_save_labels}/{file_name}.txt', "a") as file:
                    file.seek(0, 2)  # перемещение курсора в конец файла
                    file.write(f'{class_msk} {scaled_contour}' "\n")

    cv2.imwrite(f'{path_to_save_labels}10_{file_name}_img_normalized.jpg', cv2.resize(img_normalized, (1000, 1000)))
    cv2.imwrite(f'{path_to_save_labels}{file_name}_final_output.jpg', cv2.resize(final_output, (1000, 1000)))
    cv2.imwrite(f'{path_to_save_labels}{file_name}.jpg', img_normalized_clear)

    return None


if __name__ == "__main__":
    bad_list = []
    nii_archive_name = [name for name in os.listdir(dicom_dataset_dir)]

    for nii_arch in tqdm(nii_archive_name):
        # nii_arch - архив со срезами nii.gz (альтернатива dicom)
        img = nib.load(f'{dicom_dataset_dir}{nii_arch}')
        data = img.get_fdata().astype(np.int16)  # int16 → float32
        # Пересчёт в HU (стандартные параметры КТ)
        hu_data = data * 1.0 - 0  # slope=1, intercept=-1024
        slice_mean = int(hu_data.shape[-1] / 2)
        slise_save_list = []

        for slice in range(-3, 4):
            slise_save = []
            slise_save = hu_data[:, :, slice_mean + slice]
            slise_save = cv2.rotate(slise_save, cv2.ROTATE_90_CLOCKWISE)
            slise_save_list.append(slise_save)
        cnt = 0
        for hu_img_orig in slise_save_list:
            file_name = basename(nii_arch).split('.')[0]
            file_name = f'{file_name}_{cnt}'
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
            # Создаём исходное изображение для цветового вывода (в цветном формате)
            color_output_all = np.zeros((hu_img.shape[0], hu_img.shape[1], 3), dtype=np.uint8)

            for label, (hu_range, color) in hu_ranges.items():
                min_hu, max_hu = hu_range
                mask = np.logical_and(hu_img >= min_hu, hu_img <= max_hu)
                mask = mask.astype(np.uint8)
                if label == 'Кость':
                    color_output_all = create_bone_mask(mask, hu_img, color_output_all, file_name, color)
                elif label == "Мышца":
                    color_output_all = create_muscles_mask(mask, hu_img, color_output_all, file_name, color)
                elif label == "Воздух":
                    color_output_all = create_lung_mask(mask, hu_img, color_output_all, file_name, color)
                elif label == "Жир":
                    color_output_all = crerate_adipose_mask(mask, hu_img, color_output_all, file_name, color)
                else:
                    continue

            cv2.imwrite(f'{path_to_save_image_log}{file_name}/8_color_output_all.jpg', color_output_all)
            if hu_img_orig.dtype != np.uint8:
                # Нормализация в диапазон 0-255, сохраняя относительную яркость пикселей
                # img_normalized = (new_image - new_image.min()) / (new_image.max() - new_image.min()) * 255.0
                img_normalized = classic_norm(hu_img_orig)
                img_normalized = img_normalized.astype(np.uint8)
            else:
                img_normalized = hu_img_orig
            # img_normalized = cv2.drawContours(img_normalized, contours, -1, (0, 255, 0), 1)

            # Наложение изображений с прозрачностью

            # color_output2 = cv2.resize(color_output, (1000,1000))
            alpha_channel = img_normalized.astype(np.float32) / 255.0
            # Убедиться, что color_output имеет тип uint8 (стандартный тип для изображений в OpenCV)
            color_output = color_output_all.astype(np.uint8)

            # Добавить альфа-канал к color_output (теперь будет 4 канала: RGBA)
            rgba_output = cv2.merge([color_output, (alpha_channel * 255).astype(np.uint8)])

            # Наложение изображения с альфа-каналом на белый фон (или любой другой фон, который вы предпочитаете)
            white_background = np.full((512, 512, 3), 255, dtype=np.uint8)
            final_output = (
                    white_background * (1 - alpha_channel[:, :, np.newaxis]) + rgba_output[:, :, :3] * alpha_channel[
                                                                                                       :,
                                                                                                       :,
                                                                                                       np.newaxis]).astype(
                np.uint8)

            mask_muscles = get_mask_muscles(color_output)
            mask_bone = get_mask_bone(color_output)
            mask_lung = get_mask_lung(color_output)
            mask_adipose = get_mask_adipose(color_output)
            color_output = clear_color_output(only_body_mask, color_output)
            color_output = highlight_small_masks(color_output)
            cv2.imwrite(f'{path_to_save_image_log}{file_name}/9_color_output.jpg', color_output)
            create_femm_mask_file([mask_muscles, mask_bone, mask_lung, mask_adipose], color_output, img_normalized,
                                  final_output, file_name)
            cnt+=1
