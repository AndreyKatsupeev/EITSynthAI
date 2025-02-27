from fastapi import FastAPI, File, UploadFile, HTTPException
import zipfile
from io import BytesIO
from typing import Dict
import logging
import base64
import pydicom
from pydicom.filebase import DicomBytesIO

from collections import defaultdict


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
            for file_name in zip_file.namelist():
                logger.info(f"Обработка файла: {file_name}")
                try:
                    # Чтение файла в бинарном режиме и кодирование в base64
                    file_data = zip_file.read(file_name)
                    extracted_files[file_name] = base64.b64encode(file_data).decode('utf-8')

                    # Чтение DICOM-файла
                    dicom_data = DicomBytesIO(file_data)
                    dicom_data_slice = pydicom.dcmread(dicom_data)
                    series_uid = dicom_data_slice.SeriesInstanceUID
                    series_dict[series_uid].append(dicom_data_slice)

                    logger.info(f"Метаданные DICOM: PatientName={patient_name}, StudyDate={study_date}")

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