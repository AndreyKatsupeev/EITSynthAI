import cv2
import numpy
import os
import pydicom

from os.path import basename
from tqdm import tqdm

workdir = '/media/msi/FSI/fsi/fsi_draft/datasets'
save_dir = f'{workdir}/font_dataset'
dicom_dataset_dir = f'{workdir}/dicom_main'

dicom_folders_name = [name for name in os.listdir(dicom_dataset_dir) if
                      os.path.isdir(os.path.join(dicom_dataset_dir, name))]

def read_dicom_folder(folder_path):
    """

    :param folder_path:
    :return:
    """
    slices = []
    for filename in os.listdir(folder_path):
        if ".dcm" in filename.lower():
            filepath = os.path.join(folder_path, filename)
            slice_dcm = pydicom.dcmread(filepath)
            slices.append(slice_dcm)
    return slices


# Шаг 2: Преобразование в 3D-массив
def convert_to_3d(slices):
    # Сортировка срезов по положению (при необходимости)
    slices.sort(key=lambda x: int(x.InstanceNumber))

    # Извлечение массива пиксельных данных
    pixel_data = [slice_dicom.pixel_array for slice_dicom in slices]
    patient_position = slices[0][0x0018, 0x5100].value
    # Стекирование в 3D-массив
    img_3d = numpy.stack(pixel_data,
                      axis=-1)  # Axis=-1 для аксиальных срезов, предполагая, что третий измерение - это срезы
    return img_3d, patient_position


# Шаг 3: Ребайндинг в сагиттальную плоскость с коррекцией ориентации
def axial_to_sagittal(img_3d, patient_position='HFS'):
    """

    :param img_3d:
    :param patient_position:
    :return:
    """
    # Простое транспонирование для аксиального в сагиттальный с коррекцией ориентации
    if patient_position == 'FFS':
        sagittal_view = numpy.transpose(img_3d, (2, 1, 0))  # Перестановка осей
        sagittal_view = numpy.flipud(sagittal_view)
    elif patient_position == 'HFS':
        sagittal_view = numpy.transpose(img_3d, (2, 1, 0))  # Перестановка осей
    return sagittal_view

if __name__ == "__main__":
    global_count = 0
    for dicom_dir in tqdm(dicom_folders_name):
        slices = read_dicom_folder(f'{dicom_dataset_dir}/{dicom_dir}')
        file_name = "fsi_ribs_segmentation"
        img_3d, patient_position = convert_to_3d(slices)
        sagittal_view = axial_to_sagittal(img_3d, patient_position)
        slice_mean = sagittal_view.shape[-1] // 2
        for i in range(-6,7):
            slise_save = sagittal_view[:, :, slice_mean+i]
            slise_save = cv2.normalize(slise_save, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)

            cv2.imwrite(f'{save_dir}/{file_name}_{global_count}_{i}.jpg', slise_save)
        global_count +=1
        # print(sagittal_view)
        #
        # cv2.namedWindow('sagittal_view', cv2.WINDOW_NORMAL)
        # cv2.imshow('sagittal_view', sagittal_view)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
