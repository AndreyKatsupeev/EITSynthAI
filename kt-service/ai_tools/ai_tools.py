import abc
import cv2
import numpy
import logging
import os
import supervision as sv

from ultralytics import YOLO
from ai_tools.utils import axial_to_sagittal, convert_to_3d, create_dicom_dict, search_number_axial_slice, create_answer

import config
import zipfile

from typing import Dict

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DICOMSequencesToMask(abc.ABC):
    def __init__(self, model_path=None):
        if model_path:
            self.model_path = model_path
        else:
            print(os.getcwd())
            self.model_path = config.ribs_segm_model
        self.model = self.__load_model(self.model_path)

    def __load_model(self, model_path):
        return YOLO(model_path)

    def __search_front_slise(self, zip_buffer):
        """"""
        # Словарь для хранения файлов в памяти
        extracted_files: Dict[str, str] = {}

        # Разархивирование в память
        with zipfile.ZipFile(zip_buffer, 'r') as zip_file:
            dicom_dict = create_dicom_dict(zip_file)
            for i_slices in dicom_dict[0].values():
                try:
                    img_3d, patient_position, image_orientation, patient_orientation = convert_to_3d(i_slices)
                    sagittal_view = axial_to_sagittal(img_3d, patient_position, image_orientation,
                                                      patient_orientation)  # нарезка вертикальных срезов
                    front_slice_mean_num = sagittal_view.shape[-1] // 2  # Вычисляем номер среднего среза -> int
                    front_slice_mean = sagittal_view[:, :, front_slice_mean_num]  # Срез без нормализации
                    # Нормализуем пиксели в диапазоне 0....255
                    front_slice_norm = cv2.normalize(front_slice_mean, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
                except Exception as e:
                    pass
        return front_slice_norm, img_3d

    def ribs_predict(self, front_slice):
        """"""
        front_slice = cv2.cvtColor(front_slice, cv2.COLOR_BGR2RGB)
        cv2.namedWindow('slise_save', cv2.WINDOW_NORMAL)
        cv2.imshow('slise_save', front_slice)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        results = self.model(front_slice, conf=0.3, verbose=False)
        detections = sv.Detections.from_ultralytics(results[0])
        return detections

    def search_axial_slice(self, detections, i_slices):
        """"""
        axial_slice_list = []
        number_slice_eit_list = search_number_axial_slice(detections)
        for i in number_slice_eit_list:
            axial_slice_list.append(i_slices[i])
        return axial_slice_list

    def get_abs_coordinate_slice_from_dicom(self, zip_buffer, answer=None):
        """"""
        front_slice, img_3d = self.__search_front_slise(zip_buffer)
        ribs_boxes = self.ribs_predict(front_slice)
        axial_slice = self.search_axial_slice(ribs_boxes)
        axial_slice_mask = 0
        axial_slice_mask_coord = 0
        # box_annotator = sv.BoxAnnotator(color=sv.Color.RED)
        # annotated_image = front_slice.copy()
        # annotated_image = box_annotator.annotate(annotated_image, detections=detections)

        answer = create_answer(front_slice, ribs_boxes, axial_slice)
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
#     def __init__(self, model_path=None):
#         if model_path:
#             self.model_path = model_path
#         else:
#             self.model_path = config.model_path
#         self.model = self.__load_model(self.model_path)
#
#     def __load_model(self, model_path):
#         return model.load(model_path)
#
#     def __get_slice(self, images):
#         return slise
#
#     def __ribs_predict(self, front_slice):
#         pass
