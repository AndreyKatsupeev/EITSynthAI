import zipfile

from io import BytesIO


def dicom_sequence_to_zip(uploaded_files):
    """"""
    if uploaded_files:
        # Создаем архив в памяти
        zip_buffer = BytesIO()
        with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
            for uploaded_file in uploaded_files:
                # Добавляем каждый файл в архив
                zip_file.writestr(uploaded_file.name, uploaded_file.getvalue())
    return zip_buffer
