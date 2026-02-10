import abc
import time
from datetime import datetime
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

from .femm_tools.synthetic_datasets_generator import simulate_EIT_monitoring_pyeit

from pathlib import Path

from skimage.exposure import equalize_adapthist

# Добавляем папку `kt-service` в PYTHONPATH
sys.path.append(str(Path(__file__).parent))

from .. import kt_service_config
import zipfile
import torch
# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)



class DICOMabc(abc.ABC):
    """

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
        front_slice_norm, img_3d, i_slices, custom_number_slise = [], [], [], []
        try:
            with zipfile.ZipFile(zip_buffer, 'r') as zip_file:
                i_slices, custom_number_slise = create_dicom_dict(zip_file)
                img_3d, patient_position, image_orientation, patient_orientation = convert_to_3d(i_slices)
                logger.info(f"✅ Функция _search_front_slise | patient_position {patient_position}, image_orientation {image_orientation}, patient_orientation {patient_orientation}")
                sagittal_view = axial_to_sagittal(img_3d, patient_position, image_orientation,
                                                patient_orientation)  # нарезка вертикальных срезов
                # Вычисляем номер среднего среза -> int
                front_slice_mean_num = sagittal_view.shape[-1] // 2
                front_slice_mean = sagittal_view[:, :, front_slice_mean_num]  # Срез без нормализации
                # Нормализуем пиксели в диапазоне 0....255
                front_slice_norm = cv2.normalize(front_slice_mean, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
            logger.info(f"✅ Выход функции _search_front_slise | размер front_slice_norm {front_slice_norm.shape}, размер img_3d {img_3d.shape}, i_slices_len {len(i_slices)}, номер выбранного среза {custom_number_slise}")
        except:
            logger.error(f"🔴 Ошибка в функции _search_front_slise | i_slices_len {len(i_slices)}")
        return front_slice_norm, img_3d, i_slices, custom_number_slise

    def _ribs_predict(self, front_slice):
        """
        Функция для предсказания координат рёбер пациента на фронтальном срезе

        Args:
            front_slice: нормализованный фронтальный срез

        Returns:
            detections: класс YOLO с полным набором предсказаний (содержание описано в функции search_number_axial_slice)

        """
        try:
            logger.info(f"✅ Функция _ribs_predict | поступил срез размером {front_slice.shape}")
            front_slice = cv2.cvtColor(front_slice, cv2.COLOR_BGR2RGB)
            results = self.ribs_model(front_slice, conf=0.3, verbose=False, show_conf=False,
                                    show_labels=False)
            detections = sv.Detections.from_ultralytics(results[0])
            logger.info(f"✅ Функция _ribs_predict | предсказано рёбер {len(detections.confidence)}")
        except:
            logger.error(f"🔴 Ошибка в функции _ribs_predict | предсказано рёбер {len(detections.confidence)}")
        return detections

    def _axial_slice_predict(self, axial_slice):
        """
        Выполняет сегментацию тканей тела на аксиальном срезе КТ с помощью предобученной модели.
        """
        try:
            # Конвертируем из BGR в RGB
            axial_slice_rgb = cv2.cvtColor(axial_slice, cv2.COLOR_BGR2RGB)
            
            # Получаем целевой размер изображения
            axial_slice_size = get_axial_slice_size(axial_slice)
            
            # Выбираем модель в зависимости от размера
            if axial_slice_size == 256:
                logger.info(f"✅ Выбрана модель на разрешение 256")
                model = self.axial_model_256
            else:
                logger.info(f"✅ Выбрана модель на разрешение 512")
                model = self.axial_model_512
            
            # Проверка доступности CUDA
            logger.info(f"✅ Device name: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
            
            # Замер времени
            t1 = time.time()
            results = model(axial_slice_rgb, conf=0.3, verbose=False, imgsz=axial_slice_size)[0]                
            segmentation_time = round(time.time() - t1, 3)
            logger.info(f"⏱️ Время сегментации: {segmentation_time:.2f} seconds")
        except:
            logger.error(f"🔴 Ошибка в функции _axial_slice_predict | len_axial_slice {len(axial_slice)}")
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
        axial_slice_list, number_slice_eit_list = [], []
        try:
            axial_slice_list = []
            number_slice_eit_list = search_number_axial_slice(detections, custom_number_slise)
            for i in number_slice_eit_list:
                axial_slice_list.append(i_slices[i])
        except:
            logger.error(f"🔴 Ошибка в функции _search_axial_slice")
        return axial_slice_list, number_slice_eit_list


class DICOMSequencesToMask(DICOMabc):
    """
    """
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
        answer = []
        try:
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
            list_crd_from_color_output = create_list_crd_from_color_output(color_output, pixel_spacing, only_body_mask)

            segmentation_results_cnt = create_segmentation_results_cnt(axial_segmentations)
            img_mesh, meshdata = create_mesh(list_crd_from_color_output[:2], list_crd_from_color_output[2:])
            img_mesh = cv2.flip(img_mesh, 0)
            segmentation_masks_full_image = create_segmentation_masks_full_image(
                segmentation_masks_image, only_body_mask, ribs_annotated_image,
                axial_slice_norm_body, img_mesh
            )
            simulation_results, saved_file_name, simulation_time = self.get_synthetic_dataset(meshdata)
            answer = create_answer(segmentation_masks_full_image, segmentation_results_cnt, segmentation_time, saved_file_name, simulation_time)
        except:
            logger.error("🔴 Ошибка в классе DICOMSequencesToMask, функция get_coordinate_slice_from_dicom")
        return answer

    def get_synthetic_dataset(self, meshdata):
        """
            Генерирование синтетического набора данных ЭИТ для заданной сетки.

            Функция запускает численное моделирование ЭИТ-мониторинга с изменяющейся
            во времени проводимостью лёгких, сохраняет результаты в файл с именем,
            содержащим текущее время (вплоть до секунд), и возвращает вычисленные данные.

            :param meshdata: dict, данные сетки, полученные из FEMM-генератора
            :return: tuple:
                - simulation_results: list[np.ndarray], рассчитанные ЭИТ-векторы
                - filename: str, путь к файлу с сохранёнными результатами
                - simulation_time: float, время генерации датасета в секундах
            """
        try:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"/app/generation_results/results_{ts}.dat"
            logger.info(f"✅ Начало simulate_EIT_monitoring_pyeit")
            simulation_results, simulation_time = simulate_EIT_monitoring_pyeit(meshdata, isSaveToFile=True, filename=filename, materials_location="/app/kt_service/ai_tools/femm_tools")
            logger.info(f"✅ Конец simulate_EIT_monitoring_pyeit")
        except:
            logger.error("🔴 Ошибка в классе DICOMSequencesToMask, функция get_synthetic_dataset")
        return simulation_results, filename, simulation_time


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
        answer = []
        try:
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
            list_crd_from_color_output = create_list_crd_from_color_output(color_output, pixel_spacing, only_body_mask)
            segmentation_results_cnt = create_segmentation_results_cnt(axial_segmentations)
            img_mesh, meshdata = create_mesh(list_crd_from_color_output[:2], list_crd_from_color_output[2:])
            img_mesh = cv2.flip(img_mesh, 0)
            segmentation_masks_full_image = create_segmentation_masks_full_image(
                segmentation_masks_image, only_body_mask, ribs_annotated_image,
                axial_slice_norm_body, img_mesh
            )

            simulation_results, saved_file_name, simulation_time = self.get_synthetic_dataset(meshdata)
            answer = create_answer(segmentation_masks_full_image, segmentation_results_cnt, segmentation_time, saved_file_name, simulation_time)
        except:
            logger.error("🔴 Ошибка в классе DICOMSequencesToMaskCustom, функция get_coordinate_slice_from_dicom_custom")
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
        answer = []
        try:
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
            list_crd_from_color_output = create_list_crd_from_color_output(color_output, pixel_spacing, only_body_mask)
            segmentation_results_cnt = create_segmentation_results_cnt(axial_segmentations)
            img_mesh, meshdata = create_mesh(list_crd_from_color_output[:2], list_crd_from_color_output[2:])
            img_mesh = cv2.flip(img_mesh, 0)
            segmentation_masks_full_image = create_segmentation_masks_full_image(
                segmentation_masks_image, only_body_mask, ribs_annotated_image,
                axial_slice_norm_body, img_mesh
            )
            simulation_results, saved_file_name, simulation_time = self.get_synthetic_dataset(meshdata)
            answer = create_answer(segmentation_masks_full_image, segmentation_results_cnt, segmentation_time, saved_file_name, simulation_time)
        except:
            logger.error("🔴 Ошибка в классе DICOMToMask, функция get_coordinate_slice_from_dicom_frame")
        return answer


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
        answer = []
        try:
            only_body_mask = None
            ribs_annotated_image = None
            pixel_spacing = [0.753906, 0.753906]
            axial_segmentations, segmentation_time = self._axial_slice_predict(axial_slice_norm_body)
            segmentation_masks_image = create_segmentations_masks(axial_segmentations)
            color_output = create_color_output(segmentation_masks_image, only_body_mask)
            list_crd_from_color_output = create_list_crd_from_color_output(color_output, pixel_spacing)
            segmentation_results_cnt = create_segmentation_results_cnt(axial_segmentations)
            img_mesh, meshdata = create_mesh(list_crd_from_color_output[:2], list_crd_from_color_output[2:])
            img_mesh = cv2.flip(img_mesh, 0)
            segmentation_masks_full_image = create_segmentation_masks_full_image(segmentation_masks_image, only_body_mask,
                                                                                ribs_annotated_image,
                                                                                axial_slice_norm_body, img_mesh
                                                                                )
            simulation_results, saved_file_name, simulation_time = self.get_synthetic_dataset(meshdata)
            answer = create_answer(segmentation_masks_full_image, segmentation_results_cnt, segmentation_time, saved_file_name, simulation_time)
        except:
            logger.error("🔴 Ошибка в классе ImageToMask, функция get_coordinate_slice_from_image")
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
        answer = []
        try:
            ribs_annotated_image = None
            pixel_spacing = [0.662, 0.662]  
            with zipfile.ZipFile(zip_buffer, 'r') as zip_file:
                nii_mean_slice, pixel_spacing = get_nii_mean_slice(zip_file)
                axial_slice_norm = classic_norm(nii_mean_slice)
                axial_slice_norm = cv2.rotate(axial_slice_norm, cv2.ROTATE_180)
                only_body_mask = get_axial_slice_body_mask_nii(nii_mean_slice)
                axial_slice_norm_body = cv2.bitwise_and(axial_slice_norm, axial_slice_norm,
                                                        mask=only_body_mask)  # Выделяем тело в изображении HU
                axial_segmentations, segmentation_time = self._axial_slice_predict(axial_slice_norm_body)
                segmentation_masks_image = create_segmentations_masks(axial_segmentations)
                color_output = create_color_output(segmentation_masks_image, only_body_mask)
                list_crd_from_color_output = create_list_crd_from_color_output(color_output, pixel_spacing, only_body_mask)
                segmentation_results_cnt = create_segmentation_results_cnt(axial_segmentations)

                img_mesh, meshdata = create_mesh(list_crd_from_color_output[:2], list_crd_from_color_output[2:])
                img_mesh = cv2.flip(img_mesh, 0)
                segmentation_masks_full_image = create_segmentation_masks_full_image(
                    segmentation_masks_image, only_body_mask, ribs_annotated_image,
                    axial_slice_norm_body, img_mesh)
                simulation_results, saved_file_name, simulation_time = self.get_synthetic_dataset(meshdata)
                answer = create_answer(segmentation_masks_full_image, segmentation_results_cnt, segmentation_time, saved_file_name, simulation_time)
        except:
            logger.error("🔴 Ошибка в классе NIIToMask, функция get_coordinate_slice_from_nii")
        return answer
