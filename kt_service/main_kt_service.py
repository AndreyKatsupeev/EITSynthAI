import logging
import zipfile
from PIL import Image
from fastapi.responses import StreamingResponse, JSONResponse
import numpy
import io
import time
import base64

from fastapi import FastAPI, File, UploadFile, HTTPException, Response
from io import BytesIO

from .ai_tools.ai_tools import DICOMSequencesToMask
import sys
from pathlib import Path

# Добавляем папку `kt-service` в PYTHONPATH
sys.path.append(str(Path(__file__).parent))


# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

dicom_seq_to_mask = DICOMSequencesToMask()


@app.post("/uploadDicomSequence")
async def upload_file(file: UploadFile = File(...)):
    try:
        logger.info("Начата обработка архива")
        contents = await file.read()
        zip_buffer = BytesIO(contents)

        # Замеряем время выполнения функции search_front_slise
        start_time = time.time()  # Засекаем начальное время
        answer = dicom_seq_to_mask.get_coordinate_slice_from_dicom(zip_buffer)
        end_time = time.time()  # Засекаем конечное время
        execution_time = round(end_time - start_time, 2)  # Вычисляем время выполнения

        # Проверяем, что answer — это массив NumPy
        if isinstance(answer, numpy.ndarray):
            # Преобразуем массив NumPy в изображение PIL
            if answer.dtype != numpy.uint8:  # Если данные не в формате uint8, нормализуем их
                answer = (answer * 255).astype(numpy.uint8)
            pil_image = Image.fromarray(answer)

            # Преобразуем изображение в байтовый поток
            img_byte_arr = io.BytesIO()
            pil_image.save(img_byte_arr, format='JPEG')  # Сохраняем в формате JPEG
            img_byte_arr.seek(0)  # Перемещаем указатель в начало потока

            # Кодируем изображение в base64 (если нужно)
            img_base64 = base64.b64encode(img_byte_arr.getvalue()).decode('utf-8')

            logger.info("Архив успешно обработан")
            # Возвращаем JSON с изображением и временем выполнения
            return JSONResponse(content={
                "message": "Файлы успешно обработаны",
                "execution_time": execution_time,  # Время выполнения
                "image": img_base64  # Изображение в формате base64
            })

        # Если answer — это не массив NumPy, возвращаем ошибку
        logger.error("Ожидался массив NumPy, но получен другой тип данных")
        raise HTTPException(status_code=500, detail="Ожидался массив NumPy, но получен другой тип данных")

    except zipfile.BadZipFile:
        logger.error("Загруженный файл не является корректным ZIP-архивом")
        raise HTTPException(status_code=400, detail="Загруженный файл не является корректным ZIP-архивом")
    except Exception as e:
        logger.error(f"Ошибка при обработке файла: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Ошибка при обработке файла: {str(e)}")