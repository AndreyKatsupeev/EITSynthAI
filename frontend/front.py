
import base64
import requests
import streamlit as st
import time
import io

import frontend_config as config


from PIL import Image


from frontend_utils import dicom_sequence_to_zip, dicom_sequence_custom_to_zip, dicom_frame_to_zip, \
    image_axial_slice_to_zip, nii_sequence_to_zip


# Настройка страницы
st.set_page_config(page_title="", layout="wide")
st.markdown("<h2 style='text-align: center; color: white;'>Сервис формирования датасета для ЭИТ</h2>",
            unsafe_allow_html=True)
col1, col2 = st.columns(2)

with col1:
    with st.expander("Описание решения"):
        st.markdown("""Сервис позволяет генерировать датасеты для ЭИТ. Перед запуском необходимо выбрать режим генерации
        и загрузить соответствующий файл. Сервис поддерживает файлы .dicom, .nii, .jpg, .png.
        """)
with col2:
    with st.expander("Описание режимов генерации датасета для ЭИТ"):
        st.markdown("""
    * dicom_sequences_auto - Автоматический режим. Принимается dicom-серия и автоматически выбирается нужный срез.
    * dicom_sequences_custom - Ручной режим. Принимается dicom-серия и пользователь может сам задать номер среза, 
    который ему нужен. Мы ищем центральный поперечный срез, а пользователь может извлечь нужный срез, начиная с 
    центрального. +1,+2,-1,-2 и т.п. При положительном значении будут извлекаться срезы ниже нулевого. При 
    отрицательном выше нулевого. Если значение не задано, то будет выбран срез между 6 и 7 ребром (по аналогии с 
    режимом dicom_sequences_auto).
    * dicom_frame - Обработка одного dicom-среза. Режим применяется, если в наличии есть только один срез.
    * jpg_png - Обработка изображений. Поперечный срез тела в формате jpg, png.
    * nii - Формат файла исследования .nii""")

# Логотип в сайдбаре
st.sidebar.image("logo.jpg", use_container_width=True)  # Замените "logo.png" на путь к вашему логотипу

# Выбор маркера в сайдбаре
generation_mode = st.sidebar.radio(
    "Выберите режим генерации датасета:",
    ("dicom_sequences_auto", "dicom_sequences_custom", "dicom_frame", "jpg_png", "nii")
)

# Поле для ввода текста, если выбран dicom_sequences_custom
if generation_mode == "dicom_sequences_custom":
    custom_input = st.sidebar.text_input("Введите номер среза относительно центрального (+1,+2,-1,-2):")

if __name__ == "__main__":
    # Загрузка файла
    uploaded_file = st.file_uploader("Загрузите файл", accept_multiple_files=True)
    button_flag = st.button('Запустить генерацию датасета для ЭИТ')

    # Обработка загруженного файла
    if button_flag and uploaded_file is not None:
        st.write("Файл успешно загружен!")
        with st.spinner('Обработка DICOM файлов...'):
            if generation_mode == "dicom_sequences_auto":
                try:
                    dicom_zip = dicom_sequence_to_zip(uploaded_file)
                    files = {'file': ('dicom_files.zip', dicom_zip.getvalue(), 'application/zip')}
                    t_start = time.time()
                    response = requests.post(config.upload_dicom_sequence_http, files=files)
                    t_finish = time.time() - t_start

                    if response.status_code == 200:
                        result = response.json()

                        # Отображаем время выполнения
                        st.success(f"Обработка завершена за {result.get('execution_time', int(t_finish))} с")
                        st.success(f"Время сегментации {result['segmentation_time']} c")

                        # Отображаем текстовые данные
                        if 'text_data' in result:
                            st.subheader("Результаты сегментации:")
                            st.text(result['text_data'])

                        # Отображаем изображение
                        if 'image' in result:
                            img_bytes = base64.b64decode(result['image'].encode('utf-8'))
                            img = Image.open(io.BytesIO(img_bytes))
                            st.image(img, caption="Результат сегментации", use_container_width=True)
                    else:
                        st.error(f"Ошибка обработки: {response.text}")

                except requests.exceptions.RequestException as e:
                    st.error(f"Ошибка соединения с сервером: {str(e)}")
                except Exception as e:
                    st.error(f"Неожиданная ошибка: {str(e)}")
            elif generation_mode == "dicom_sequences_custom":
                try:
                    dicom_zip = dicom_sequence_custom_to_zip(uploaded_file, custom_input)
                    files = {'file': ('dicom_files.zip', dicom_zip.getvalue(), 'application/zip')}
                    t_start = time.time()
                    response = requests.post(config.upload_dicom_sequence_custom_http, files=files)
                    t_finish = time.time() - t_start

                    if response.status_code == 200:
                        result = response.json()

                        # Отображаем время выполнения
                        st.success(f"Обработка завершена за {result.get('execution_time', int(t_finish))} с")
                        st.success(f"Время сегментации {result['segmentation_time']} c")

                        # Отображаем текстовые данные
                        if 'text_data' in result:
                            st.subheader("Результаты сегментации:")
                            st.text(result['text_data'])

                        # Отображаем изображение
                        if 'image' in result:
                            img_bytes = base64.b64decode(result['image'].encode('utf-8'))
                            img = Image.open(io.BytesIO(img_bytes))
                            st.image(img, caption="Результат сегментации", use_container_width=True)
                    else:
                        st.error(f"Ошибка обработки: {response.text}")

                except requests.exceptions.RequestException as e:
                    st.error(f"Ошибка соединения с сервером: {str(e)}")
                except Exception as e:
                    st.error(f"Неожиданная ошибка: {str(e)}")
            elif generation_mode == "dicom_frame":
                try:
                    dicom_zip = dicom_frame_to_zip(uploaded_file)
                    files = {'file': ('dicom_files.zip', dicom_zip.getvalue(), 'application/zip')}
                    t_start = time.time()
                    response = requests.post(config.upload_dicom_frame_http, files=files)
                    t_finish = time.time() - t_start

                    if response.status_code == 200:
                        result = response.json()

                        # Отображаем время выполнения
                        st.success(f"Обработка завершена за {result.get('execution_time', int(t_finish))} с")
                        st.success(f"Время сегментации {result['segmentation_time']} c")

                        # Отображаем текстовые данные
                        if 'text_data' in result:
                            st.subheader("Результаты сегментации:")
                            st.text(result['text_data'])

                        # Отображаем изображение
                        if 'image' in result:
                            img_bytes = base64.b64decode(result['image'].encode('utf-8'))
                            img = Image.open(io.BytesIO(img_bytes))
                            st.image(img, caption="Результат сегментации", use_container_width=True)
                    else:
                        st.error(f"Ошибка обработки: {response.text}")

                except requests.exceptions.RequestException as e:
                    st.error(f"Ошибка соединения с сервером: {str(e)}")
                except Exception as e:
                    st.error(f"Неожиданная ошибка: {str(e)}")
            elif generation_mode == "jpg_png":
                try:
                    image_axial_slice_zip = image_axial_slice_to_zip(uploaded_file)
                    files = {'file': ('dicom_files.zip', image_axial_slice_zip.getvalue(), 'application/zip')}
                    t_start = time.time()
                    response = requests.post(config.upload_image_axial_slice_http, files=files)
                    t_finish = time.time() - t_start

                    if response.status_code == 200:
                        result = response.json()

                        # Отображаем время выполнения
                        st.success(f"Обработка завершена за {result.get('execution_time', int(t_finish))} с")
                        st.success(f"Время сегментации {result['segmentation_time']} c")

                        # Отображаем текстовые данные
                        if 'text_data' in result:
                            st.subheader("Результаты сегментации:")
                            st.text(result['text_data'])

                        # Отображаем изображение
                        if 'image' in result:
                            img_bytes = base64.b64decode(result['image'].encode('utf-8'))
                            img = Image.open(io.BytesIO(img_bytes))
                            st.image(img, caption="Результат сегментации", use_container_width=True)
                    else:
                        st.error(f"Ошибка обработки: {response.text}")

                except requests.exceptions.RequestException as e:
                    st.error(f"Ошибка соединения с сервером: {str(e)}")
                except Exception as e:
                    st.error(f"Неожиданная ошибка: {str(e)}")
            elif generation_mode == "nii":
                try:
                    nii_zip = nii_sequence_to_zip(uploaded_file)
                    files = {'file': ('dicom_files.zip', nii_zip.getvalue(), 'application/zip')}
                    t_start = time.time()
                    response = requests.post(config.upload_nii_http, files=files)
                    t_finish = time.time() - t_start

                    if response.status_code == 200:
                        result = response.json()

                        # Отображаем время выполнения
                        st.success(f"Обработка завершена за {result.get('execution_time', int(t_finish))} с")
                        st.success(f"Время сегментации {result['segmentation_time']} c")

                        # Отображаем текстовые данные
                        if 'text_data' in result:
                            st.subheader("Результаты сегментации:")
                            st.text(result['text_data'])

                        # Отображаем изображение
                        if 'image' in result:
                            img_bytes = base64.b64decode(result['image'].encode('utf-8'))
                            img = Image.open(io.BytesIO(img_bytes))
                            st.image(img, caption="Результат сегментации", use_container_width=True)
                    else:
                        st.error(f"Ошибка обработки: {response.text}")

                except requests.exceptions.RequestException as e:
                    st.error(f"Ошибка соединения с сервером: {str(e)}")
                except Exception as e:
                    st.error(f"Неожиданная ошибка: {str(e)}")
            else:
                st.error(f"Ошибка generation_mode")


