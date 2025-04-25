import base64
import cv2
import os
from collections import defaultdict
import logging
import numpy
import pydicom
import sys
import traceback

# Добавляем корень проекта в пути Python
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pydicom.filebase import DicomBytesIO
from ai_tools.ai_tools import DICOMSequencesToMask
from ai_tools.utils import axial_to_sagittal, convert_to_3d, create_dicom_dict, search_number_axial_slice, create_answer

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
dicom_dataset_dir = '/media/msi/fsi/fsi/datasets_mrt/all_data_dicom'

path_to_save_dicom = f'../../save_test_masks/'
dicom_seq_to_mask = DICOMSequencesToMask()


def read_dicom_folder(folder_path):
    """
    Функция для чтения dicom-файла. Один dicom - это один срез с метаинформацией.
    После снятия КТ с пациента они помещаются в одну папку в виде dicom-файлов. Если в папке один dicom,
    то скорее всего это один срез. Встречаются папки с несколькими сериями, эта функция заносит их в словарь, где клоюч
    это номер серии.

    :param folder_path: путь к папке
    :return: функция возвращает словарь dicom-файлов из одной директории в пределах серии
    """
    series_dict = defaultdict(list)
    for filename in os.listdir(folder_path):
        try:
            filepath = os.path.join(folder_path, filename)
            dicom_data_slice = pydicom.dcmread(filepath)  # Срез с метаданными
            series_uid = dicom_data_slice.SeriesInstanceUID
            series_dict[series_uid].append(dicom_data_slice)
        except:
            pass
    return series_dict


def search_dicom(series_dict, target_instance_number=117):
    """
    Функция для поиска DICOM-файла с заданным InstanceNumber в словаре серий.

    :param series_dict: словарь DICOM-серий (ключ - SeriesInstanceUID, значение - список DICOM-файлов)
    :param target_instance_number: искомый номер InstanceNumber
    :return: найденный DICOM-файл или None, если не найден
    """
    for series_uid, dicom_list in series_dict.items():
        for dicom in dicom_list:
            if dicom.InstanceNumber == target_instance_number:
                return dicom
            else:
                return None


def check_work_folders(path):
    """"""
    if not os.path.exists(path):
        os.makedirs(path)
        print("Created save directories")

def log_image_normalization(new_image):
    """"""
    img_log = numpy.log1p(new_image)
    img_log_normalized = (img_log - img_log.min()) / (img_log.max() - img_log.min()) * 255.0
    return img_log_normalized



def vignetting_image_normalization(new_image):
    """
    Виньетирование (отсечение крайних значений)
    DICOM-изображения могут содержать выбросы (очень тёмные/светлые пиксели), которые "сжимают" полезный диапазон.
    Args:
        new_image:

    Returns:

    """
    p_low, p_high = numpy.percentile(new_image, [2, 98])  # Отсекаем 2% самых тёмных и светлых пикселей
    img_clipped = numpy.clip(new_image, p_low, p_high)
    img_normalized = (img_clipped - p_low) / (p_high - p_low) * 255.0
    return img_normalized


def z_image_normalization(new_image):
    """"""
    mean = new_image.mean()
    std = new_image.std()
    img_zscore = (new_image - mean) / std
    # Масштабируем обратно в [0, 255]
    img_zscore_normalized = (img_zscore - img_zscore.min()) / (img_zscore.max() - img_zscore.min()) * 255.0
    return img_zscore_normalized


if __name__ == "__main__":
    bad_list = []
    dicom_folders_name = [name for name in os.listdir(dicom_dataset_dir) if
                          os.path.isdir(os.path.join(dicom_dataset_dir, name))]
    for dicom_dir in dicom_folders_name:
        series_dict = read_dicom_folder(f'{dicom_dataset_dir}/{dicom_dir}')
        for i_slices in series_dict.values():
            print(i_slices[0])
            img_3d, patient_position, image_orientation, patient_orientation = convert_to_3d(i_slices)
            sagittal_view = axial_to_sagittal(img_3d, patient_position, image_orientation,
                                              patient_orientation)  # нарезка вертикальных срезов
            front_slice_mean_num = sagittal_view.shape[-1] // 2  # Вычисляем номер среднего среза -> int
            front_slice_mean = sagittal_view[:, :, front_slice_mean_num]  # Срез без нормализации
            # Нормализуем пиксели в диапазоне 0....255
            front_slice_norm = cv2.normalize(front_slice_mean, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
            detections = dicom_seq_to_mask.ribs_predict(front_slice_norm)
            number_slice_eit_list, number_slice_eit_list = dicom_seq_to_mask.search_axial_slice(detections, i_slices)
            # Сохранение DICOM-файла
            cnt = 0
            for axial_slice_image in number_slice_eit_list:
                new_image = axial_slice_image.pixel_array
                # img_normalized = (new_image - new_image.min()) / (new_image.max() - new_image.min()) * 255.0
                img_normalized = vignetting_image_normalization(new_image)
                img_normalized = img_normalized.astype(numpy.uint8)
                check_work_folders(f'{path_to_save_dicom}/dicom_axial_dataset/dicom/')
                check_work_folders(f'{path_to_save_dicom}/dicom_axial_dataset/image/')
                axial_slice_image.save_as(f'{path_to_save_dicom}/dicom_axial_dataset/dicom/{dicom_dir}_{cnt}.dcm')
                cv2.imwrite(f'{path_to_save_dicom}/dicom_axial_dataset/image/{dicom_dir}_{cnt}.jpg', img_normalized)
                cnt+=1
                # cv2.namedWindow('slise_save', cv2.WINDOW_NORMAL)
                # cv2.imshow('slise_save', img_normalized)
                # cv2.waitKey(0)
                # cv2.destroyAllWindows()




    # ['patient_661', 'patient_661', 'patient_565', 'patient_185', 'patient_185', 'patient_421', 'patient_421', 'patient_116', 'patient_116', 'patient_116', 'patient_116', 'patient_121', 'patient_620', 'patient_620', 'patient_503', 'patient_503', 'patient_102', 'patient_105', 'patient_105', 'patient_278', 'patient_278', 'patient_438', 'patient_113', 'patient_287', 'patient_287', 'patient_274', 'patient_737', 'patient_276', 'patient_413', 'patient_413', 'patient_291', 'patient_679', 'patient_234', 'patient_234', 'patient_119', 'patient_119', 'patient_119', 'patient_96', 'patient_296', 'patient_296', 'patient_100', 'patient_687', 'patient_197', 'patient_197', 'patient_107', 'patient_107', 'patient_519', 'patient_442', 'patient_442', 'patient_149', 'patient_76', 'patient_76', 'patient_76', 'patient_76', 'patient_156', 'patient_548', 'patient_548', 'patient_668', 'patient_668', 'patient_668', 'patient_476', 'patient_476', 'patient_89', 'patient_89', 'patient_289', 'patient_289', 'patient_289', 'patient_289', 'patient_684', 'patient_524', 'patient_524', 'patient_731', 'patient_117', 'patient_117', 'patient_1', 'patient_154', 'patient_154', 'patient_154', 'patient_154', 'patient_352', 'patient_410', 'patient_410', 'patient_37', 'patient_408', 'patient_408', 'patient_87', 'patient_87', 'patient_24', 'patient_109', 'patient_109', 'patient_270', 'patient_270', 'patient_98', 'patient_98', 'patient_669', 'patient_669', 'patient_220', 'patient_220', 'patient_299', 'patient_82', 'patient_82', 'patient_95', 'patient_95', 'patient_104', 'patient_104', 'patient_118', 'patient_118', 'patient_115', 'patient_115', 'patient_414', 'patient_122', 'patient_122', 'patient_161', 'patient_161', 'patient_14', 'patient_447', 'patient_447', 'patient_532', 'patient_554', 'patient_554', 'patient_554', 'patient_554', 'patient_108', 'patient_108', 'patient_215', 'patient_93', 'patient_93', 'patient_685', 'patient_537', 'patient_187', 'patient_187', 'patient_394', 'patient_394', 'patient_384', 'patient_384', 'patient_43', 'patient_78', 'patient_688', 'patient_83', 'patient_83', 'patient_481', 'patient_466', 'patient_466', 'patient_683', 'patient_32', 'patient_103', 'patient_103', 'patient_686', 'patient_261', 'patient_261', 'patient_261', 'patient_261', 'patient_691', 'patient_691', 'patient_691', 'patient_691', 'patient_691', 'patient_691', 'patient_691', 'patient_691', 'patient_691', 'patient_691', 'patient_691', 'patient_691', 'patient_691', 'patient_691', 'patient_691', 'patient_691', 'patient_691', 'patient_691', 'patient_691', 'patient_691', 'patient_691', 'patient_691', 'patient_691', 'patient_691', 'patient_691', 'patient_691', 'patient_691', 'patient_691', 'patient_691', 'patient_691', 'patient_691', 'patient_691', 'patient_691', 'patient_691', 'patient_691', 'patient_691', 'patient_691', 'patient_691', 'patient_691', 'patient_691', 'patient_691', 'patient_691', 'patient_691', 'patient_691', 'patient_691', 'patient_691', 'patient_691', 'patient_691', 'patient_691', 'patient_691', 'patient_691', 'patient_691', 'patient_691', 'patient_691', 'patient_691', 'patient_691', 'patient_691', 'patient_691', 'patient_691', 'patient_691', 'patient_691', 'patient_691', 'patient_691', 'patient_691', 'patient_691', 'patient_691', 'patient_691', 'patient_691', 'patient_691', 'patient_691', 'patient_691', 'patient_691', 'patient_691', 'patient_691', 'patient_691', 'patient_691', 'patient_691', 'patient_691', 'patient_691', 'patient_691', 'patient_691', 'patient_691', 'patient_691', 'patient_691', 'patient_691', 'patient_691', 'patient_691', 'patient_691', 'patient_691', 'patient_691', 'patient_691', 'patient_691', 'patient_691', 'patient_691', 'patient_691', 'patient_305', 'patient_305', 'patient_541', 'patient_91', 'patient_91', 'patient_38', 'patient_302', 'patient_302', 'patient_81', 'patient_81', 'patient_114', 'patient_110', 'patient_110', 'patient_42', 'patient_504', 'patient_674', 'patient_133', 'patient_133', 'patient_660', 'patient_94', 'patient_94', 'patient_600', 'patient_681', 'patient_530', 'patient_316', 'patient_316', 'patient_147', 'patient_147', 'patient_365', 'patient_365', 'patient_128', 'patient_45', 'patient_269', 'patient_269', 'patient_269', 'patient_269', 'patient_269', 'patient_269', 'patient_269', 'patient_269', 'patient_269', 'patient_285', 'patient_285', 'patient_646', 'patient_646', 'patient_80', 'patient_111', 'patient_111', 'patient_507', 'patient_101', 'patient_99', 'patient_99', 'patient_44', 'patient_65', 'patient_65', 'patient_65', 'patient_322', 'patient_322', 'patient_322', 'patient_566', 'patient_566', 'patient_680', 'patient_542', 'patient_690', 'patient_690', 'patient_690', 'patient_690', 'patient_690', 'patient_690', 'patient_690', 'patient_690', 'patient_690', 'patient_690', 'patient_690', 'patient_690', 'patient_690', 'patient_690', 'patient_690', 'patient_690', 'patient_690', 'patient_690', 'patient_690', 'patient_690', 'patient_690', 'patient_690', 'patient_690', 'patient_690', 'patient_690', 'patient_690', 'patient_690', 'patient_690', 'patient_690', 'patient_690', 'patient_690', 'patient_690', 'patient_690', 'patient_690', 'patient_690', 'patient_690', 'patient_690', 'patient_690', 'patient_690', 'patient_690', 'patient_690', 'patient_690', 'patient_690', 'patient_690', 'patient_690', 'patient_690', 'patient_690', 'patient_690', 'patient_690', 'patient_690', 'patient_690', 'patient_690', 'patient_690', 'patient_690', 'patient_690', 'patient_690', 'patient_690', 'patient_690', 'patient_690', 'patient_690', 'patient_690', 'patient_690', 'patient_690', 'patient_690', 'patient_690', 'patient_690', 'patient_690', 'patient_690', 'patient_690', 'patient_690', 'patient_690', 'patient_690', 'patient_690', 'patient_690', 'patient_690', 'patient_690', 'patient_690', 'patient_690', 'patient_690', 'patient_690', 'patient_690', 'patient_690', 'patient_690', 'patient_690', 'patient_690', 'patient_690', 'patient_690', 'patient_690', 'patient_690', 'patient_690', 'patient_690', 'patient_690', 'patient_690', 'patient_690', 'patient_690', 'patient_595', 'patient_595', 'patient_77', 'patient_77', 'patient_77', 'patient_240', 'patient_240', 'patient_500', 'patient_351', 'patient_351', 'patient_422', 'patient_422', 'patient_693', 'patient_693', 'patient_693', 'patient_92', 'patient_92', 'patient_677', 'patient_676', 'patient_4', 'patient_213', 'patient_213', 'patient_150', 'patient_150', 'patient_150', 'patient_150', 'patient_150', 'patient_150', 'patient_150', 'patient_150', 'patient_150', 'patient_120', 'patient_86', 'patient_86', 'patient_682', 'patient_85', 'patient_85', 'patient_464', 'patient_464', 'patient_489', 'patient_692', 'patient_692', 'patient_692', 'patient_692', 'patient_297', 'patient_297', 'patient_297', 'patient_297', 'patient_58', 'patient_58', 'patient_58', 'patient_432', 'patient_738', 'patient_738', 'patient_738']