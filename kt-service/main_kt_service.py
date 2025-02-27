import cv2
import logging
import zipfile

from collections import defaultdict
from fastapi import FastAPI, File, UploadFile, HTTPException
from io import BytesIO
from typing import Dict

from utils.utils import axial_to_sagittal, convert_to_3d, create_dicom_list

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

@app.post("/uploadDicomSequence")
async def upload_file(file: UploadFile = File(...)):
    try:
        logger.info("Начата обработка архива")
        contents = await file.read()
        zip_buffer = BytesIO(contents)

        # Словарь для хранения файлов в памяти
        extracted_files: Dict[str, str] = {}
        series_dict = defaultdict(list)

        # Разархивирование в память
        with zipfile.ZipFile(zip_buffer, 'r') as zip_file:
            dicom_list = create_dicom_list(zip_file)
            for i_slices in dicom_list.values():
                try:
                    img_3d, patient_position, image_orientation, patient_orientation = convert_to_3d(i_slices)
                    sagittal_view = axial_to_sagittal(img_3d, patient_position, image_orientation,
                                                      patient_orientation)  # нарезка вертикальных срезов
                    slice_mean = sagittal_view.shape[-1] // 2  # Вычисляем средний срез
                    # Нормализуем пиксели в диапазоне 0....255
                    slise_save = cv2.normalize(slice_mean, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
                    # ribs = predict(slise_save)
                    # slice_eit = search_slice(ribs)
                    #
                    # masks_list = predict_maks(slice_eit)
                    # save_coord(masks_list)

                except Exception as e:
                    logger.error(f"Ошибка при чтении файла {file_name}: {e}")
                    continue  # Пропускаем файл, если он не может быть прочитан

        logger.info("Архив успешно обработан")
        return {"message": "Файлы успешно разархивированы", "files": extracted_files}

    except zipfile.BadZipFile:
        logger.error("Загруженный файл не является корректным ZIP-архивом")
        raise HTTPException(status_code=400, detail="Загруженный файл не является корректным ZIP-архивом")
    except Exception as e:
        logger.error(f"Ошибка при обработке файла: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Ошибка при обработке файла: {str(e)}")