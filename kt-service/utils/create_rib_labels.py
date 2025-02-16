"""Скрипт для создания датасета на основе предобученных весов"""
import cv2
import os
import numpy
from ultralytics import YOLO

# Загружаем обученную модель YOLOv8
model = YOLO("best.pt")  # Укажите путь к вашим весам

# Путь к папке с изображениями
image_folder = "1"
# Путь к папке для сохранения результатов
save_folder = "save"

# Создаем папку для сохранения, если она не существует
os.makedirs(save_folder, exist_ok=True)

def check_work_folders(path):
    if not os.path.exists(path):
        os.makedirs(path)
        print("Created save directories")

def polygon_to_str(arr):
    """

    Args:
        arr:

    Returns:

    """

    result = numpy.insert(arr, 0, 0).astype(object)  # Используем object для смешанных типов
    result[0] = int(result[0])  # Явно преобразуем первый элемент в int

    # Преобразуем в строку
    result_str = ' '.join(map(str, result))
    return result_str

check_work_folders('images')
check_work_folders('labels')
check_work_folders('save')

# Перебираем все изображения в папке
for image_name in os.listdir(image_folder):
    image_path = os.path.join(image_folder, image_name)
    image_name_clear = image_name.split('.')[0]
    if not os.path.isfile(image_path):
        continue

    # Загружаем изображение
    image = cv2.imread(image_path)
    cv2.imwrite(f'images/{image_name}', image)
    if image is None:
        print(f"Ошибка загрузки изображения: {image_path}")
        continue

    # Делаем предсказание
    results = model.predict(image, imgsz=640, conf=0.3, verbose=False)

    # Отрисовываем полигоны (сегментацию) на изображении
    for result in results:
        result.save(f'save/{image_name}', labels=False)
        try:
            polygon_list_yolo = result.masks.xyn
            for one_polygon in polygon_list_yolo:
                if len(one_polygon):
                    coord_norm = polygon_to_str(one_polygon)
                    if len(coord_norm):
                        check_labels = os.path.exists(f'labels/{image_name_clear}.txt')
                        if not check_labels:
                            with open(f'labels/{image_name_clear}.txt', "w") as file:
                                file.write(f'{coord_norm}' "\n")
                        else:
                            with open(f'labels/{image_name_clear}.txt', "a") as file:
                                file.seek(0, 2)  # перемещение курсора в конец файла
                                file.write(f'{coord_norm}' "\n")  # собственно, запись
        except:
            print('no detections', image_name)

        # print(result.masks.xyn)
        # exit()
        # save_path_ = os.path.join(save_folder, image_name)
        # result.save(save_path, labels=False)
    #     if result.masks is not None:  # Проверяем, есть ли маски (полигоны)
    #         print(result.masks)
    #         masks = result.masks.xy  # Получаем маски
    #         for mask in masks:
    #             cv2.drawContours(image, mask, -1, (0, 255, 0), 2)
    #
    # # Сохраняем изображение с результатами
    # save_path = os.path.join(save_folder, image_name)
    # cv2.imwrite(save_path, image)
    # print(f"Результат сохранен: {save_path}")

print("Обработка завершена.")
