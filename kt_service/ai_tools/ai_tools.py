import abc
import cv2
import numpy
import logging
import os
import supervision as sv
import sys
import torch
from ultralytics import YOLO
from .utils import axial_to_sagittal, convert_to_3d, create_dicom_dict, search_number_axial_slice, \
    create_answer, classic_norm, draw_annotate, create_segmentations_masks, create_segmentation_results_cnt, \
    get_axial_slice_body_mask, create_segmentations_masks_full

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
            self.ribs_model_path = config.ribs_segm_model
        self.ribs_model = self.__load_model(self.ribs_model_path)

        if axial_model_path:
            self.axial_model_path = axial_model_path
        else:
            self.axial_model_path = config.axial_slice_segm_model
        self.axial_model = self.__load_model(self.axial_model_path)

    def __load_model(self, model_path):
        return YOLO(model_path, task='segment')

    def __search_front_slise(self, zip_buffer):
        """"""
        # Словарь для хранения файлов в памяти
        extracted_files: Dict[str, str] = {}
        # Разархивирование в память
        with zipfile.ZipFile(zip_buffer, 'r') as zip_file:
            i_slices = create_dicom_dict(zip_file)
            img_3d, patient_position, image_orientation, patient_orientation = convert_to_3d(i_slices)
            sagittal_view = axial_to_sagittal(img_3d, patient_position, image_orientation,
                                              patient_orientation)  # нарезка вертикальных срезов
            front_slice_mean_num = sagittal_view.shape[-1] // 2  # Вычисляем номер среднего среза -> int
            front_slice_mean = sagittal_view[:, :, front_slice_mean_num]  # Срез без нормализации
            # Нормализуем пиксели в диапазоне 0....255
            front_slice_norm = cv2.normalize(front_slice_mean, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
        return front_slice_norm, img_3d, i_slices

    def ribs_predict(self, front_slice):
        """"""
        front_slice = cv2.cvtColor(front_slice, cv2.COLOR_BGR2RGB)
        results = self.ribs_model(front_slice, conf=0.3, verbose=False, device=0)
        detections = sv.Detections.from_ultralytics(results[0])
        return detections

    def axial_slice_predict(self, axial_slice):
        """"""
        axial_slice = cv2.cvtColor(axial_slice, cv2.COLOR_BGR2RGB)
        results = self.axial_model(axial_slice, conf=0.3, verbose=False, device=0, imgsz=512)[0]
        return results

    def search_axial_slice(self, detections, i_slices):
        """"""
        axial_slice_list = []
        number_slice_eit_list = search_number_axial_slice(detections)
        for i in number_slice_eit_list:
            axial_slice_list.append(i_slices[i])
        return axial_slice_list, number_slice_eit_list

    def get_coordinate_slice_from_dicom(self, zip_buffer, answer=None):
        """"""
        front_slice, img_3d, i_slices = self.__search_front_slise(zip_buffer)
        ribs_detections = self.ribs_predict(front_slice)
        axial_slice, number_slice_eit_list = self.search_axial_slice(ribs_detections, i_slices)
        axial_slice_norm = classic_norm(axial_slice[-1].pixel_array)
        only_body_mask = get_axial_slice_body_mask(axial_slice[-1])

        axial_slice_norm_body = cv2.bitwise_and(axial_slice_norm, axial_slice_norm,
                                                mask=only_body_mask)

        ribs_annotated_image = draw_annotate(ribs_detections, front_slice, number_slice_eit_list)
        axial_segmentations = self.axial_slice_predict(axial_slice_norm_body)
        segmentation_masks_image = create_segmentations_masks(axial_segmentations)
        segmentation_masks_full_image = create_segmentations_masks_full(segmentation_masks_image, axial_slice_norm_body, ribs_annotated_image)
        segmentation_results_cnt = create_segmentation_results_cnt(axial_segmentations)

        axial_slice_mask = 0
        axial_slice_mask_coord = 0

        answer = create_answer(ribs_annotated_image, ribs_detections, axial_slice)
        return answer

#
# class DICOMToMask():
#     pass
#
#
# class ImageToMask():
#     pass
#
#
# class NIIToMask():
#     pass


# class SearchSlice(abc.ABC):
#     def __init__(self, ribs_model_path=None):
#         if ribs_model_path:
#             self.ribs_model_path = ribs_model_path
#         else:
#             self.ribs_model_path = config.ribs_model_path
#         self.model = self.__load_model(self.ribs_model_path)
#
#     def __load_model(self, ribs_model_path):
#         return model.load(ribs_model_path)
#
#     def __get_slice(self, images):
#         return slise
#
#     def __ribs_predict(self, front_slice):
#         pass
