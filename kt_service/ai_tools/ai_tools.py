import abc
import time
import requests
import cv2
import logging
import supervision as sv
import sys
import numpy
import matplotlib.pyplot as plt
from ultralytics import YOLO
from .utils import axial_to_sagittal, convert_to_3d, create_dicom_dict, search_number_axial_slice, \
    create_answer, classic_norm, draw_annotate, create_segmentations_masks, create_segmentation_results_cnt, \
    get_axial_slice_body_mask, create_segmentation_masks_full_image, get_axial_slice_body_mask_nii, get_nii_mean_slice, \
    create_list_crd_from_color_output, get_pixel_spacing, create_color_output, get_axial_slice_size

from .mesh_tools.femm_generator import create_mesh

from pathlib import Path

from skimage.exposure import equalize_adapthist

# Добавляем папку `kt-service` в PYTHONPATH
sys.path.append(str(Path(__file__).parent))

from .. import kt_service_config
import zipfile

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DICOMSequencesToMask(abc.ABC):
    """Класс для автоматического поиска нужного среза из dicom-серии.
       Является родителем для классов DICOMSequencesToMaskCustom, DICOMToMask, ImageToMask, NIIToMask

    """

    def __init__(self, ribs_model_path=None, axial_model_256_path=None, axial_model_512_path=None):
        """
        Инициализация моделей для сегментации.
        Загружает две версии модели (256 и 512), если пути не заданы — берётся из конфига.
        """
        # Модель для сегментации рёбер
        if ribs_model_path:
            self.ribs_model_path = ribs_model_path
        else:
            self.ribs_model_path = kt_service_config.ribs_segm_model
        self.ribs_model = self._load_model(self.ribs_model_path)

        # Модели для аксиальной сегментации (разные разрешения)
        if axial_model_256_path:
            self.axial_model_256_path = axial_model_256_path
        else:
            self.axial_model_256_path = kt_service_config.axial_slice_segm_model_256

        if axial_model_512_path:
            self.axial_model_512_path = axial_model_512_path
        else:
            self.axial_model_512_path = kt_service_config.axial_slice_segm_model_512

        # Загружаем обе модели заранее
        self.axial_model_256 = self._load_model(self.axial_model_256_path)
        self.axial_model_512 = self._load_model(self.axial_model_512_path)

    def _load_model(self, model_path):
        """Загружает YOLO-модель для сегментации."""
        return YOLO(model_path, task='segment')

    def _search_front_slise(self, zip_buffer):
        """
        Основная функция для поиска фронтального среза

        Функция открывает архив со срезами, преобразует их в массив и выполняет поиск среднего фронтального среза
        Args:
            zip_buffer: архив с dicom-файлами

        Returns:
            front_slice_norm: нормализованный фронтальный срез
            img_3d: массив ненормализованных срезов в формате numpy
            i_slices: dicom-серия со всеми метаданными
            custom_number_slise: номер среза для коррекции (для кастомной настройки)

        """
        # Разархивирование в память
        with zipfile.ZipFile(zip_buffer, 'r') as zip_file:
            i_slices, custom_number_slise = create_dicom_dict(zip_file)
            img_3d, patient_position, image_orientation, patient_orientation = convert_to_3d(i_slices)
            sagittal_view = axial_to_sagittal(img_3d, patient_position, image_orientation,
                                              patient_orientation)  # нарезка вертикальных срезов
            # Вычисляем номер среднего среза -> int
            front_slice_mean_num = sagittal_view.shape[-1] // 2
            front_slice_mean = sagittal_view[:, :, front_slice_mean_num]  # Срез без нормализации
            # Нормализуем пиксели в диапазоне 0....255
            front_slice_norm = cv2.normalize(front_slice_mean, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
        return front_slice_norm, img_3d, i_slices, custom_number_slise

    def _ribs_predict(self, front_slice):
        """
        Функция для предсказания координат рёбер пациента на фронтальном срезе

        Args:
            front_slice: нормализованный фронтальный срез

        Returns:
            detections: класс YOLO с полным набором предсказаний (содержание описано в функции search_number_axial_slice)

        """
        front_slice = cv2.cvtColor(front_slice, cv2.COLOR_BGR2RGB)
        results = self.ribs_model(front_slice, conf=0.3, verbose=False, device=0, show_conf=False,
                                  show_labels=False)
        detections = sv.Detections.from_ultralytics(results[0])
        return detections

    def _axial_slice_predict(self, axial_slice):
        """
        Выполняет сегментацию тканей тела на аксиальном срезе КТ с помощью предобученной модели.

        Основные этапы работы:
        1. Конвертирует цветовое пространство из BGR в RGB (требование модели)
        2. Определяет размер среза и выбирает соответствующую модель (256 или 512)
        3. Выполняет предсказание с заданными параметрами
        4. Возвращает результаты сегментации и время выполнения

        Args:
            axial_slice (numpy.ndarray): Аксиальный срез КТ в формате BGR (синий, зеленый, красный)
                                       Размерность: (H, W, 3), dtype: uint8

        Returns:
            tuple: Кортеж содержащий:
                - results (ultralytics.engine.results.Results): Результаты детекции/сегментации модели
                - segmentation_time (float): Время выполнения предсказания в секундах

        Note:
            - Использует GPU (device=0) для ускорения
            - Порог уверенности (conf) = 0.3
            - Размер входного изображения (imgsz) зависит от модели: 256 или 512
            - Для корректной работы модели требуется конвертация в RGB
        """
        # Конвертируем из BGR в RGB
        axial_slice_rgb = cv2.cvtColor(axial_slice, cv2.COLOR_BGR2RGB)

        # Получаем целевой размер изображения (например, 256 или 512)
        axial_slice_size = get_axial_slice_size(axial_slice)

        # Выбираем модель в зависимости от размера
        if axial_slice_size == 256:
            model = self.axial_model_256
        else:
            model = self.axial_model_512

        # Замер времени
        t1 = time.time()

        # Предсказание
        results = model(
            axial_slice_rgb,
            conf=0.3,
            verbose=False,
            device=0,
            imgsz=axial_slice_size
        )[0]

        segmentation_time = time.time() - t1

        return results, segmentation_time

    def _search_axial_slice(self, detections, i_slices, custom_number_slise=0):
        """
        Функция для поиска аксиального среза

        Args:
            detections: класс YOLO с полным набором предсказаний (содержание описано в функции search_number_axial_slice)
            i_slices: dicom-серия
            custom_number_slise: номер среза для ручной поправки (по умолчанию равен 0)

        Returns:
            axial_slice_list: список выбранных dicom-срезов с метаданными
            number_slice_eit_list: номера dicom-срезов с метаданными (6,7 ребро и между ними)

        """
        axial_slice_list = []
        number_slice_eit_list = search_number_axial_slice(detections, custom_number_slise)
        for i in number_slice_eit_list:
            axial_slice_list.append(i_slices[i])
        return axial_slice_list, number_slice_eit_list

    def get_coordinate_slice_from_dicom(self, zip_buffer):
        """
        Основная функция для получения координат биологических тканей из dicom-файла

        Args:
            zip_buffer: архив с dicom-файлами

        Returns:
                answer = {
                    "image": img_base64,
                    "text_data": segmentation_results_cnt,
                    "segmentation_time": segmentation_time,
                    "status": "success",
                    "message": "Processing completed successfully"}
        """
        img_mesh = None
        front_slice, img_3d, i_slices, _ = self._search_front_slise(zip_buffer)

        ribs_detections = self._ribs_predict(front_slice)
        axial_slice, number_slice_eit_list = self._search_axial_slice(ribs_detections, i_slices)
        axial_slice_norm = classic_norm(axial_slice[-1].pixel_array)
        only_body_mask = get_axial_slice_body_mask(axial_slice[-1])
        pixel_spacing = get_pixel_spacing(axial_slice[-1])
        axial_slice_norm_body = cv2.bitwise_and(axial_slice_norm, axial_slice_norm,
                                                mask=only_body_mask)
        ribs_annotated_image = draw_annotate(ribs_detections, front_slice, number_slice_eit_list)
        axial_segmentations, segmentation_time = self._axial_slice_predict(axial_slice_norm_body)
        segmentation_masks_image = create_segmentations_masks(axial_segmentations)
        color_output = create_color_output(segmentation_masks_image, only_body_mask)
        list_crd_from_color_output = create_list_crd_from_color_output(color_output, pixel_spacing)
        segmentation_results_cnt = create_segmentation_results_cnt(axial_segmentations)
        img_mesh = create_mesh(list_crd_from_color_output[:2], list_crd_from_color_output[2:])
        img_mesh = cv2.flip(img_mesh, 0)
        segmentation_masks_full_image = create_segmentation_masks_full_image(
            segmentation_masks_image, only_body_mask, ribs_annotated_image,
            axial_slice_norm_body, img_mesh
        )
        answer = create_answer(segmentation_masks_full_image, segmentation_results_cnt, segmentation_time)
        return answer


class DICOMSequencesToMaskCustom(DICOMSequencesToMask):
    """Класс для кастомного поиска среза из dicom-серии. Наследуется от get_coordinate_slice_from_dicom и отличается тем,
    что принимает значение нужного среза с фронта, которое вводит пользователь. Если пользователь не вводит, то алгоритм
    отрабатывает также как в методе get_coordinate_slice_from_dicom """

    def get_coordinate_slice_from_dicom_custom(self, zip_buffer, answer=None):
        """
        Функция для получения координат биологических тканей из dicom-серии с возможностью выбора среза

        Args:
            zip_buffer: архив с dicom-файлами

        Returns:
                answer = {
                    "image": img_base64,
                    "text_data": segmentation_results_cnt,
                    "segmentation_time": segmentation_time,
                    "status": "success",
                    "message": "Processing completed successfully"}
        """
        img_mesh = None
        front_slice, img_3d, i_slices, custom_number_slise = self._search_front_slise(zip_buffer)
        ribs_detections = self._ribs_predict(front_slice)
        axial_slice, number_slice_eit_list = self._search_axial_slice(ribs_detections, i_slices, custom_number_slise)
        axial_slice_norm = classic_norm(axial_slice[-1].pixel_array)
        only_body_mask = get_axial_slice_body_mask(axial_slice[-1])
        pixel_spacing = get_pixel_spacing(axial_slice[-1])
        axial_slice_norm_body = cv2.bitwise_and(axial_slice_norm, axial_slice_norm,
                                                mask=only_body_mask)
        ribs_annotated_image = draw_annotate(ribs_detections, front_slice, number_slice_eit_list)
        axial_segmentations, segmentation_time = self._axial_slice_predict(axial_slice_norm_body)
        segmentation_masks_image = create_segmentations_masks(axial_segmentations)
        color_output = create_color_output(segmentation_masks_image, only_body_mask)
        list_crd_from_color_output = create_list_crd_from_color_output(color_output, pixel_spacing)
        segmentation_results_cnt = create_segmentation_results_cnt(axial_segmentations)
        img_mesh = create_mesh(list_crd_from_color_output[:2], list_crd_from_color_output[2:])
        img_mesh = cv2.flip(img_mesh, 0)
        segmentation_masks_full_image = create_segmentation_masks_full_image(
            segmentation_masks_image, only_body_mask, ribs_annotated_image,
            axial_slice_norm_body, img_mesh
        )

        answer = create_answer(segmentation_masks_full_image, segmentation_results_cnt, segmentation_time)

        return answer


#
class DICOMToMask(DICOMSequencesToMask):
    """
    Класс для обработки одиночного dicom-файла. Наследуется от класса DICOMSequencesToMask
    """

    def get_coordinate_slice_from_dicom_frame(self, zip_buffer, answer=None):
        """
        Основная функция для получения координат биологических тканей из одиночного dicom-файла

        Args:
            zip_buffer: архив с dicom-файлами

        Returns:
                answer = {
                    "image": img_base64,
                    "text_data": segmentation_results_cnt,
                    "segmentation_time": segmentation_time,
                    "status": "success",
                    "message": "Processing completed successfully"}
        """
        # Разархивирование в память
        ribs_annotated_image = None
        with zipfile.ZipFile(zip_buffer, 'r') as zip_file:
            i_slices, _ = create_dicom_dict(zip_file)
        axial_slice_norm = classic_norm(i_slices[-1].pixel_array)
        pixel_spacing = get_pixel_spacing(i_slices[-1])
        only_body_mask = get_axial_slice_body_mask(i_slices[-1])
        axial_slice_norm_body = cv2.bitwise_and(axial_slice_norm, axial_slice_norm,
                                                mask=only_body_mask)
        axial_segmentations, segmentation_time = self._axial_slice_predict(axial_slice_norm_body)
        segmentation_masks_image = create_segmentations_masks(axial_segmentations)
        color_output = create_color_output(segmentation_masks_image, only_body_mask)
        list_crd_from_color_output = create_list_crd_from_color_output(color_output, pixel_spacing)
        segmentation_results_cnt = create_segmentation_results_cnt(axial_segmentations)
        img_mesh = create_mesh(list_crd_from_color_output[:2], list_crd_from_color_output[2:])
        img_mesh = cv2.flip(img_mesh, 0)
        segmentation_masks_full_image = create_segmentation_masks_full_image(
            segmentation_masks_image, only_body_mask, ribs_annotated_image,
            axial_slice_norm_body, img_mesh
        )
        answer = create_answer(segmentation_masks_full_image, segmentation_results_cnt, segmentation_time)
        return answer


#
#
class ImageToMask(DICOMSequencesToMask):
    """
    Класс для запуска сегментации КТ-снимка в формате нормализованного изображения.
    Наследуется от класса DICOMSequencesToMask
    """

    def get_coordinate_slice_from_image(self, axial_slice_norm_body):
        """
        Функция для сегментации КТ-снимка в формате нормализованного изображения

        Args:
            axial_slice_norm_body: нормализованное изображение

        Returns:
                answer = {
                    "image": img_base64,
                    "text_data": segmentation_results_cnt,
                    "segmentation_time": segmentation_time,
                    "status": "success",
                    "message": "Processing completed successfully"}
        """
        only_body_mask = None
        ribs_annotated_image = None
        pixel_spacing = [1, 1]
        axial_segmentations, segmentation_time = self._axial_slice_predict(axial_slice_norm_body)
        segmentation_masks_image = create_segmentations_masks(axial_segmentations)
        color_output = create_color_output(segmentation_masks_image, only_body_mask)
        list_crd_from_color_output = create_list_crd_from_color_output(color_output, pixel_spacing)
        segmentation_results_cnt = create_segmentation_results_cnt(axial_segmentations)
        img_mesh = create_mesh(list_crd_from_color_output[:2], list_crd_from_color_output[2:])
        img_mesh = cv2.flip(img_mesh, 0)
        segmentation_masks_full_image = create_segmentation_masks_full_image(segmentation_masks_image, only_body_mask,
                                                                             ribs_annotated_image,
                                                                             axial_slice_norm_body, img_mesh
                                                                             )
        answer = create_answer(segmentation_masks_full_image, segmentation_results_cnt, segmentation_time)
        return answer


class NIIToMask(DICOMSequencesToMask):
    """
    Класс для сегментации КТ-серии в формате nii.  Наследуется от класса DICOMSequencesToMask.
    """

    def get_coordinate_slice_from_nii(self, zip_buffer, answer=None):
        """
        У nii файлов меньше срезов пачке, поэтому фронтальный срез не получается хорошего качества. При обработке nii
        просто берется средний срез в пачке

        Args:
            zip_buffer: архив с dicom-файлами

        Returns:
                answer = {
                    "image": img_base64,
                    "text_data": segmentation_results_cnt,
                    "segmentation_time": segmentation_time,
                    "status": "success",
                    "message": "Processing completed successfully"}
        """
        ribs_annotated_image = None
        pixel_spacing = [1, 1]
        with zipfile.ZipFile(zip_buffer, 'r') as zip_file:
            nii_mean_slice = get_nii_mean_slice(zip_file)
            axial_slice_norm = classic_norm(nii_mean_slice)
            axial_slice_norm = cv2.rotate(axial_slice_norm, cv2.ROTATE_180)
            only_body_mask = get_axial_slice_body_mask_nii(nii_mean_slice)
            axial_slice_norm_body = cv2.bitwise_and(axial_slice_norm, axial_slice_norm,
                                                    mask=only_body_mask)  # Выделяем тело в изображении HU
            axial_segmentations, segmentation_time = self._axial_slice_predict(axial_slice_norm_body)
            segmentation_masks_image = create_segmentations_masks(axial_segmentations)
            color_output = create_color_output(segmentation_masks_image, only_body_mask)
            list_crd_from_color_output = create_list_crd_from_color_output(color_output, pixel_spacing)
            segmentation_results_cnt = create_segmentation_results_cnt(axial_segmentations)

            img_mesh = create_mesh(list_crd_from_color_output[:2], list_crd_from_color_output[2:])
            img_mesh = cv2.flip(img_mesh, 0)
            segmentation_masks_full_image = create_segmentation_masks_full_image(
                segmentation_masks_image, only_body_mask, ribs_annotated_image,
                axial_slice_norm_body, img_mesh)
            answer = create_answer(segmentation_masks_full_image, segmentation_results_cnt, segmentation_time)

            return answer
