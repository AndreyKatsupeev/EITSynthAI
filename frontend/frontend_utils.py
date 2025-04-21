import zipfile

from io import BytesIO


def dicom_sequence_to_zip(uploaded_files, custom_input=None):
    """"""
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
    if uploaded_files:
        # Создаем архив в памяти
        zip_buffer = BytesIO()
        with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
            for uploaded_file in uploaded_files:
                # Добавляем каждый файл в архив
                zip_file.writestr(uploaded_file.name, uploaded_file.getvalue())
    return zip_buffer