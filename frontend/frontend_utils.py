import os
import zipfile


from io import BytesIO
from loguru import logger


def dicom_sequence_to_zip(uploaded_files, custom_input=None):
    """"""
    zip_buffer = None
    if uploaded_files:
        # Создаем архив в памяти
        zip_buffer = BytesIO()
        with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
            for uploaded_file in uploaded_files:
                # Добавляем каждый файл в архив
                zip_file.writestr(uploaded_file.name, uploaded_file.getvalue())
    return zip_buffer


def dicom_sequence_custom_to_zip(uploaded_files, custom_input=None):
    """
    Создает ZIP-архив в памяти из загруженных DICOM файлов и дополнительного пользовательского ввода.

    Args:
        uploaded_files: Список загруженных файлов
        custom_input: Дополнительные пользовательские данные (строка или число)

    Returns:
        BytesIO: Буфер с ZIP-архивом
    """
    zip_buffer = None
    if uploaded_files:
        # Создаем архив в памяти
        zip_buffer = BytesIO()
        with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
            # Добавляем DICOM файлы
            for uploaded_file in uploaded_files:
                zip_file.writestr(uploaded_file.name, uploaded_file.getvalue())

            # Добавляем пользовательский ввод как текстовый файл
            if custom_input is not None:
                zip_file.writestr("custom_input.txt", str(custom_input))

    return zip_buffer


def dicom_frame_to_zip(uploaded_files, custom_input=None):
    """"""
    zip_buffer = None
    if uploaded_files:
        # Создаем архив в памяти
        zip_buffer = BytesIO()
        with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
            for uploaded_file in uploaded_files:
                # Добавляем каждый файл в архив
                zip_file.writestr(uploaded_file.name, uploaded_file.getvalue())
    return zip_buffer


def image_axial_slice_to_zip(uploaded_files, custom_input=None):
    """"""
    zip_buffer = None
    if uploaded_files:
        # Создаем архив в памяти
        zip_buffer = BytesIO()
        with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
            for uploaded_file in uploaded_files:
                # Добавляем каждый файл в архив
                zip_file.writestr(uploaded_file.name, uploaded_file.getvalue())
    return zip_buffer


def nii_sequence_to_zip(uploaded_files):
    """"""
    zip_buffer = None
    if uploaded_files:
        # Создаем архив в памяти
        zip_buffer = BytesIO()
        with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
            for uploaded_file in uploaded_files:
                # Добавляем каждый файл в архив
                zip_file.writestr(uploaded_file.name, uploaded_file.getvalue())
    return zip_buffer


def add_log(log_path, log_name, level_log):
    """"""
    full_log_name = os.path.join(log_path, f"{log_name}.log")
    logger.add(
        sink=full_log_name,
        format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {file}:{line} | {message}",
        level=level_log,
        rotation="100 MB",  # Новый файл при >100 МБ
        retention="7 days",  # Автоудаление старых
        compression="zip",  # Сжатие старых файлов
        enqueue=True  # Безопасность
    )