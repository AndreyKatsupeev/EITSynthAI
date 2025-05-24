import cv2
import numpy as np

# Размеры изображения
width, height = 200, 200

# Создаем черное изображение
image = np.zeros((height, width, 3), dtype=np.uint8)

# Координаты, которые вы предоставили
coordinates = [
    112.658247, 177.171944, 111.695356, 178.134835, 111.695356, 179.09772600000002,
    110.732465, 180.060617, 110.732465, 181.02350800000002, 109.769574, 181.986399,
    109.769574, 182.94929000000002, 109.769574, 181.986399, 110.732465, 181.02350800000002,
    110.732465, 180.060617, 111.695356, 179.09772600000002, 111.695356, 178.134835
]

# Преобразуем координаты в список кортежей (x, y)
points = [(int(coordinates[i]), int(coordinates[i + 1])) for i in range(0, len(coordinates), 2)]

# Рисуем линии между точками
for i in range(len(points) - 1):
    cv2.line(image, points[i], points[i+1], (255, 255, 255), thickness=1)

    # Показываем изображение
    cv2.namedWindow('Image', cv2.WINDOW_NORMAL)
    cv2.imshow("Image", image)
    cv2.waitKey(0)
cv2.destroyAllWindows()
