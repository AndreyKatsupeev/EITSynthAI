import streamlit as st

# Настройка страницы
st.set_page_config(page_title="", layout="wide")
st.markdown("<h2 style='text-align: center; color: white;'>Сервис формирования датасета для ЭИТ</h2>", unsafe_allow_html=True)
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
    uploaded_file = st.file_uploader("Загрузите файл", type=["dicom", "jpg", "png", "nii"])
    button_flag = st.button('Запустить генерацию датасета для ЭИТ')

    # Обработка загруженного файла
    if button_flag and uploaded_file is not None:
        st.write("Файл успешно загружен!")
        if generation_mode == "dicom_sequences_auto":
            st.write('run dicom_sequences_auto')
        elif generation_mode == "dicom_sequences_custom":
            st.write('run dicom_sequences_custom')
        elif generation_mode == "dicom_frame":
            st.write('run dicom_frame')
        elif generation_mode == "jpg_png":
            st.write('run jpg_png')
        elif generation_mode == "nii":
            st.write('run nii')
        else:
            print('error')