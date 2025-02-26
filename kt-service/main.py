from ai_tools.tools import image_decoder
from utils.fsi_logger import write_log

from fastapi import FastAPI, UploadFile, File, Request
from typing import List, Optional

app = FastAPI()


@app.post('/uploadDicom')
def dicom_handler(files: List[UploadFile] = File(...)):
    """
    1 - автоматический. Принимается dicom-серия и автоматически выбирается нужный срез.
    2 - Кастомный. Принимается dicom-серия и пользователь может сам задать номер среза, который ему нужен. Мы ищем
    центральный поперечный срез, а пользователь может извлечь нужный срез, начиная с центрального. +1,+2,-1,-2 и т.п.
    3 - Обработка одного dicom-среза. Бывает так, что есть только 1 срез.
    4 - обработка картинки. Поперечный срез тела в формате jpg, png.

    Args:
        files:

    Returns:

    {
      "mode": "auto",
      "files": [
        {
          "filename": "file1.dcm",
          "content": "base64-encoded-dicom-file"
        },
        {
          "filename": "file2.dcm",
          "content": "base64-encoded-dicom-file"
        }
      ]
    }


    {
      "mode": "custom",
      "slice_offset": 2,  // Смещение относительно центрального среза (+2, -1 и т.д.)
      "files": [
        {
          "filename": "file1.dcm",
          "content": "base64-encoded-dicom-file"
        },
        {
          "filename": "file2.dcm",
          "content": "base64-encoded-dicom-file"
        }
      ]
    }

        {
      "mode": "image",
      "files": [
        {
          "filename": "image1.jpg",
          "content": "base64-encoded-image-file"
        }
      ]
    }

    {
      "status": "success",
      "message": "Описание результата",
      "result": {
        "image": "base64-encoded-result-image",  // Если результат - изображение
        "metadata": {  // Если нужно вернуть метаданные
          "slice_number": 5,
          "series_description": "Some description"
        }
      }
    }


    {
      "mode": "auto",
      "archive": {
        "filename": "series.zip",
        "content": "base64-encoded-zip-file"
      }
    }

пример клиента

import os
import zipfile
import base64
from io import BytesIO

def create_zip_from_folder(folder_path):
    """
    Создает ZIP-архив из всех файлов в указанной папке.
    Возвращает архив в виде base64-строки.
    """
    # Создаем архив в памяти
    zip_buffer = BytesIO()

    with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
        # Рекурсивно обходим папку и добавляем файлы в архив
        for root, _, files in os.walk(folder_path):
            for file in files:
                file_path = os.path.join(root, file)
                # Добавляем файл в архив с относительным путем
                arcname = os.path.relpath(file_path, folder_path)
                zip_file.write(file_path, arcname=arcname)

    # Перемещаем указатель в начало буфера
    zip_buffer.seek(0)

    # Кодируем архив в base64
    zip_base64 = base64.b64encode(zip_buffer.getvalue()).decode('utf-8')
    return zip_base64

# Пример использования
folder_path = "path/to/your/dicom/folder"
zip_base64 = create_zip_from_folder(folder_path)

# Теперь можно отправить zip_base64 на сервер
payload = {
    "mode": "auto",
    "archive": {
        "filename": "dicom_series.zip",
        "content": zip_base64
    }
}

url = "http://your-server-address/uploadDicom"
response = requests.post(url, json=payload)

print(response.status_code)
print(response.json())


пример сервера
from fastapi import FastAPI
from pydantic import BaseModel
import base64
import zipfile
from io import BytesIO

app = FastAPI()

class Archive(BaseModel):
    filename: str
    content: str  # base64-encoded содержимое архива

class DicomRequest(BaseModel):
    mode: str
    archive: Archive

@app.post('/uploadDicom')
async def dicom_handler(request: DicomRequest):
    mode = request.mode
    archive = request.archive

    # Декодируем архив из base64
    zip_content = base64.b64decode(archive.content)
    zip_buffer = BytesIO(zip_content)

    # Распаковываем архив
    with zipfile.ZipFile(zip_buffer, 'r') as zip_file:
        # Извлекаем и обрабатываем файлы
        for file_name in zip_file.namelist():
            with zip_file.open(file_name) as file:
                file_content = file.read()
                # Здесь можно обработать DICOM-файл
                print(f"Processing {file_name} with size {len(file_content)} bytes")

    # Возвращаем результат
    return {
        "status": "success",
        "message": "Обработка завершена",
        "result": {
            "processed_files": len(zip_file.namelist())
        }
    }

    """

    list_image = image_decoder(files)
    number_out, max_proba_area, max_proba_image_index, error_message = numberRecognition(). \
        digits_detector(list_image, device, model_yolo_area, model_yolo_num)  # run neural networks
    write_log("out_digits_detector", [number_out, max_proba_area, max_proba_image_index, error_message])

    yolo_message = create_yolo_message(number_out, max_proba_area,
                                       max_proba_image_index, error_message)  # create yolo message response
    return yolo_message


@app.post('/uploadImage')
def image_handler(files: List[UploadFile] = File(...)):
    """

    Args:
        files:

    Returns:

    """
    pass