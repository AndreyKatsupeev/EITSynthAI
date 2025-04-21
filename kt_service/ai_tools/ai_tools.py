import abc
import time

import cv2
import logging
import supervision as sv
import sys
from ultralytics import YOLO
from .utils import axial_to_sagittal, convert_to_3d, create_dicom_dict, search_number_axial_slice, \
    create_answer, classic_norm, draw_annotate, create_segmentations_masks, create_segmentation_results_cnt, \
    get_axial_slice_body_mask, create_segmentations_masks_full, get_axial_slice_body_mask_nii, get_nii_mean_slice

from pathlib import Path

# Добавляем папку `kt-service` в PYTHONPATH
sys.path.append(str(Path(__file__).parent))

from .. import config
import zipfile

from typing import Dict

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DICOMSequencesToMask(abc.ABC):
    def __init__(self, ribs_model_path=None, axial_model_path=None):
        if ribs_model_path:
            self.ribs_model_path = ribs_model_path
        else:
            #  Модель для сегментации ребер
            self.ribs_model_path = config.ribs_segm_model
        self.ribs_model = self._load_model(self.ribs_model_path)

        if axial_model_path:
            #  Модель для сегментации тканей
            self.axial_model_path = axial_model_path
        else:
            self.axial_model_path = config.axial_slice_segm_model
        self.axial_model = self._load_model(self.axial_model_path)

    def _load_model(self, model_path):
        return YOLO(model_path, task='segment')

    def _search_front_slise(self, zip_buffer, custom_number_slise=0):
        """"""
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
        """"""
        front_slice = cv2.cvtColor(front_slice, cv2.COLOR_BGR2RGB)
        results = self.ribs_model(front_slice, conf=0.3, verbose=False, device=0)
        detections = sv.Detections.from_ultralytics(results[0])
        return detections

    def _axial_slice_predict(self, axial_slice):
        """"""
        axial_slice = cv2.cvtColor(axial_slice, cv2.COLOR_BGR2RGB)
        t1 = time.time()
        results = self.axial_model(axial_slice, conf=0.3, verbose=False, device=0, imgsz=512)[0]
        segmentation_time = time.time() - t1
        return results, segmentation_time

    def _search_axial_slice(self, detections, i_slices, custom_number_slise=0):
        """"""
        axial_slice_list = []
        number_slice_eit_list = search_number_axial_slice(detections, custom_number_slise)
        for i in number_slice_eit_list:
            axial_slice_list.append(i_slices[i])
        return axial_slice_list, number_slice_eit_list

    def get_coordinate_slice_from_dicom(self, zip_buffer, answer=None):
        """Класс для автоматического поиска нужного среза из dicom-серии"""

        front_slice, img_3d, i_slices, _ = self._search_front_slise(zip_buffer)

        ribs_detections = self._ribs_predict(front_slice)
        axial_slice, number_slice_eit_list = self._search_axial_slice(ribs_detections, i_slices)
        axial_slice_norm = classic_norm(axial_slice[-1].pixel_array)
        only_body_mask = get_axial_slice_body_mask(axial_slice[-1])

        axial_slice_norm_body = cv2.bitwise_and(axial_slice_norm, axial_slice_norm,
                                                mask=only_body_mask)

        ribs_annotated_image = draw_annotate(ribs_detections, front_slice, number_slice_eit_list)
        axial_segmentations, segmentation_time = self._axial_slice_predict(axial_slice_norm_body)
        segmentation_masks_image = create_segmentations_masks(axial_segmentations)
        segmentation_masks_full_image = create_segmentations_masks_full(segmentation_masks_image, only_body_mask,
                                                                        ribs_annotated_image, axial_slice_norm_body)

        segmentation_results_cnt = create_segmentation_results_cnt(axial_segmentations)
        answer = create_answer(segmentation_masks_full_image, segmentation_results_cnt, segmentation_time)
        return answer


class DICOMSequencesToMaskCustom(DICOMSequencesToMask):
    """Класс для кастомного поиска среза из dicom-серии. Наследуется от get_coordinate_slice_from_dicom и отличается тем,
    что принимает значение нужного среза с фронта, которое вводит пользователь. Если пользователь не вводит, то алгоритм
    отрабатывает также как в методе get_coordinate_slice_from_dicom """

    def get_coordinate_slice_from_dicom_custom(self, zip_buffer, answer=None):
        """"""
        front_slice, img_3d, i_slices, custom_number_slise = self._search_front_slise(zip_buffer)
        ribs_detections = self._ribs_predict(front_slice)
        axial_slice, number_slice_eit_list = self._search_axial_slice(ribs_detections, i_slices, custom_number_slise)
        axial_slice_norm = classic_norm(axial_slice[-1].pixel_array)
        only_body_mask = get_axial_slice_body_mask(axial_slice[-1])

        axial_slice_norm_body = cv2.bitwise_and(axial_slice_norm, axial_slice_norm,
                                                mask=only_body_mask)

        ribs_annotated_image = draw_annotate(ribs_detections, front_slice, number_slice_eit_list)
        axial_segmentations, segmentation_time = self._axial_slice_predict(axial_slice_norm_body)
        segmentation_masks_image = create_segmentations_masks(axial_segmentations)
        segmentation_masks_full_image = create_segmentations_masks_full(segmentation_masks_image, only_body_mask,
                                                                        ribs_annotated_image, axial_slice_norm_body)

        segmentation_results_cnt = create_segmentation_results_cnt(axial_segmentations)
        answer = create_answer(segmentation_masks_full_image, segmentation_results_cnt, segmentation_time)
        return answer


#
class DICOMToMask(DICOMSequencesToMask):
    def get_coordinate_slice_from_dicom_frame(self, zip_buffer, answer=None):
        """"""
        # Разархивирование в память
        ribs_annotated_image = None
        with zipfile.ZipFile(zip_buffer, 'r') as zip_file:
            i_slices, _ = create_dicom_dict(zip_file)
        axial_slice_norm = classic_norm(i_slices[-1].pixel_array)
        only_body_mask = get_axial_slice_body_mask(i_slices[-1])
        axial_slice_norm_body = cv2.bitwise_and(axial_slice_norm, axial_slice_norm,
                                                mask=only_body_mask)
        axial_segmentations, segmentation_time = self._axial_slice_predict(axial_slice_norm_body)
        segmentation_masks_image = create_segmentations_masks(axial_segmentations)
        segmentation_masks_full_image = create_segmentations_masks_full(segmentation_masks_image, only_body_mask,
                                                                        ribs_annotated_image, axial_slice_norm_body)

        segmentation_results_cnt = create_segmentation_results_cnt(axial_segmentations)
        answer = create_answer(segmentation_masks_full_image, segmentation_results_cnt, segmentation_time)
        return answer


#
#
class ImageToMask(DICOMSequencesToMask):
    """"""

    def get_coordinate_slice_from_image(self, axial_slice_norm_body):
        """"""
        only_body_mask = None
        ribs_annotated_image = None
        axial_segmentations, segmentation_time = self._axial_slice_predict(axial_slice_norm_body)
        segmentation_masks_image = create_segmentations_masks(axial_segmentations)
        segmentation_masks_full_image = create_segmentations_masks_full(segmentation_masks_image, only_body_mask,
                                                                        ribs_annotated_image, axial_slice_norm_body)

        segmentation_results_cnt = create_segmentation_results_cnt(axial_segmentations)
        answer = create_answer(segmentation_masks_full_image, segmentation_results_cnt, segmentation_time)
        return answer


class NIIToMask(DICOMSequencesToMask):

    def get_coordinate_slice_from_nii(self, zip_buffer, answer=None):
        """У nii файлов меньше срезов пачке, поэтому фронтальный срез не получается хорошего качества. При обработке nii
        просто берется средний срез в пачке
        """
        ribs_annotated_image = None
        with zipfile.ZipFile(zip_buffer, 'r') as zip_file:
            nii_mean_slice = get_nii_mean_slice(zip_file)
            axial_slice_norm = classic_norm(nii_mean_slice)
            axial_slice_norm = cv2.rotate(axial_slice_norm, cv2.ROTATE_180)
            only_body_mask = get_axial_slice_body_mask_nii(nii_mean_slice)
            only_body_hu_img = cv2.bitwise_and(axial_slice_norm, axial_slice_norm,
                                               mask=only_body_mask)  # Выделяем тело в изображении HU
            axial_segmentations, segmentation_time = self._axial_slice_predict(only_body_hu_img)
            segmentation_masks_image = create_segmentations_masks(axial_segmentations)
            segmentation_masks_full_image = create_segmentations_masks_full(segmentation_masks_image, only_body_mask,
                                                                            ribs_annotated_image, only_body_hu_img)

            segmentation_results_cnt = create_segmentation_results_cnt(axial_segmentations)
            answer = create_answer(segmentation_masks_full_image, segmentation_results_cnt, segmentation_time)
            return answer





