import cv2
import numpy
import pydicom
import os

from collections import defaultdict
from tqdm import tqdm

workdir = '/media/msi/fsi9/fsi/datasets_mrt'
save_dir = f'{workdir}/font_dataset_with_name/1'
dicom_dataset_dir = f'{workdir}/all_data_dicom'

dicom_folders_name = [name for name in os.listdir(dicom_dataset_dir) if
                      os.path.isdir(os.path.join(dicom_dataset_dir, name))]


def read_dicom_folder(folder_path):
    series_dict = defaultdict(list)
    for filename in os.listdir(folder_path):
        try:
            filepath = os.path.join(folder_path, filename)
            dicom_data_slice = pydicom.dcmread(filepath)
            series_uid = dicom_data_slice.SeriesInstanceUID
            series_dict[series_uid].append(dicom_data_slice)
        except:
            pass
    return series_dict


def filter_arrays(array_list):
    filtered_list = [array for array in array_list if array.shape == (512, 512)]
    return filtered_list


def convert_to_3d(slices):
    slices.sort(key=lambda x: int(x.InstanceNumber))
    pixel_data = [slice_dicom.pixel_array for slice_dicom in slices]
    patient_position = slices[0][0x0018, 0x5100].value
    image_orientation = slices[0][0x0020, 0x0037].value
    try:
        patient_orientation = slices[0][0x0020, 0x0020].value
    except:
        patient_orientation = None
    img_3d = numpy.stack(pixel_data, axis=-1)
    return img_3d, patient_position, image_orientation, patient_orientation


def axial_to_sagittal(img_3d, patient_position, image_orientation, patient_orientation):
    if patient_position == 'FFS':
        sagittal_view = numpy.transpose(img_3d, (2, 1, 0))
        sagittal_view = numpy.flipud(sagittal_view)
    elif patient_position == 'HFS':
        sagittal_view = numpy.transpose(img_3d, (2, 1, 0))

    row_orientation = numpy.array(image_orientation[:3])
    col_orientation = numpy.array(image_orientation[3:])

    if row_orientation[0] == -1:
        sagittal_view = numpy.flip(sagittal_view, axis=1)
    if col_orientation[1] == -1:
        sagittal_view = numpy.flip(sagittal_view, axis=2)

    if patient_position != 'HFS':
        if patient_orientation:
            if patient_orientation[0] == 'L':
                sagittal_view = numpy.fliplr(sagittal_view)
            if patient_orientation[1] == 'P':
                sagittal_view = numpy.flipud(sagittal_view)

    return sagittal_view


if __name__ == "__main__":
    global_count = 0
    for dicom_dir in tqdm(dicom_folders_name):
        slices = read_dicom_folder(f'{dicom_dataset_dir}/{dicom_dir}')
        for i_slices in slices.values():
            try:
                file_name = dicom_dir
                img_3d, patient_position, image_orientation, patient_orientation = convert_to_3d(i_slices)
                sagittal_view = axial_to_sagittal(img_3d, patient_position, image_orientation, patient_orientation)
                slice_mean = sagittal_view.shape[-1] // 2

                output_folder = os.path.join(save_dir, file_name)
                if not os.path.exists(output_folder):
                    os.makedirs(output_folder)

                for i in range(-3, 4):
                    slise_save = sagittal_view[:, :, slice_mean + i]
                    slise_save = cv2.normalize(slise_save, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
                    output_file = os.path.join(output_folder, f'{file_name}_{global_count}_{i}.jpg')
                    cv2.imwrite(output_file, slise_save)

                global_count += 1

            except Exception as e:
                print(f"Error processing {file_name}: {e}")

