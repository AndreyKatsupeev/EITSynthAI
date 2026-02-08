# Примеры использования функционала библиотеки

## Обзор

`femm_generator.py` — это инструмент для генерации 2D треугольных сеток из контурных данных с классификацией элементов по классам. Результат может быть экспортирован в формате, совместимом с FEMM (Finite Element Method Magnetics) или визуализирован через Gmsh GUI.

## Основные возможности

- Создание треугольной сетки из полигональных контуров
- Классификация элементов сетки по заданным контурам
- Добавление "кожи" (внешнего слоя) вокруг основного контура
- Экспорт в текстовый формат для FEMM
- Визуализация через Gmsh GUI
- Оптимизация контуров (удаление коллинеарных точек)

## Установка зависимостей

```bash
pip install gmsh numpy shapely opencv-python
```

**Важно:** Для работы с Gmsh GUI требуется установленный Gmsh (версия 4.15.0 или выше). Убедитесь, что на вашем компьютере установлен Gmsh.

### 1. Создание архитектурного фасада

**Файл:** `create_architectural_facade.py`

```python
from kt_service.ai_tools.mesh_tools.femm_generator import create_mesh

contours = [
    # Основной контур здания
    '0 0 0 600 0 600 400 500 400 500 450 100 450 100 400 0 400',
    # Цокольный этаж
    '1 20 20 580 20 580 100 20 100',
    # ... другие контуры
]

def create_architectural_facade(polygon):
    """Создает сложный архитектурный фасад"""
    create_mesh(['1', '1'], polygon,
                7,
                1.3, 1, True,
                show_meshing_result_method="gmsh",
                number_of_showed_class=3,
                is_saving_to_file=True,
                export_filename="tmp.txt")

if __name__ == "__main__":
    create_architectural_facade(contours)
```

**Особенности:**
- Определяет различные архитектурные элементы (этажи, окна, двери, колонны)
- Использует разные классы для разных элементов
- Создает реалистичную геометрию здания
- **Новый параметр:** `is_saving_to_file=True` вместо `is_exporting_to_femm`

### 2. Генеративное искусство

**Файл:** `create_generative_art.py`

```python
from kt_service.ai_tools.mesh_tools.femm_generator import create_mesh

def create_contours():
    """Создает абстрактное генеративное искусство"""
    import random
    import math
    
    contours = []
    random.seed(42)
    
    # Базовый контур
    contours.append('0 0 0 500 0 500 500 0 500')
    
    # Концентрические слои со случайными колебаниями
    for layer in range(5):
        points = []
        num_points = 20 + layer * 5
        # ... генерация точек
        
    return contours

def create_generative_art(polygon):
    create_mesh(['1', '1'], polygon,
                7,
                1.3, 1, True,
                show_meshing_result_method="gmsh",
                number_of_showed_class=3,
                is_saving_to_file=True,
                export_filename="tmp.txt")

contours = create_contours()

if __name__ == "__main__":
    create_generative_art(contours)
```

**Особенности:**
- Создает абстрактные узоры
- Использует математические функции (синусы, спирали)
- Включает правильные многоугольники
- **Структура:** разделена на функцию создания контуров и функцию создания сетки

### 3. Механическая сборка (реалистичная шестерня)

**Файл:** `create_mechanical_assembly.py`

```python
from EITSynthAI.kt_service.ai_tools.mesh_tools.femm_generator import create_mesh
import math

def create_contours():
    """Создает настоящую шестерню с зубьями"""
    contours = []

    center_x, center_y = 200, 200
    pitch_radius = 120  # радиус делительной окружности
    addendum = 20  # высота головки зуба
    dedendum = 15  # высота ножки зуба
    teeth = 12  # количество зубьев

    # 1. Внешний контур (с зубьями)
    gear_points = []
    steps_per_tooth = 10

    for tooth in range(teeth):
        for step in range(steps_per_tooth):
            # Угол для текущего шага
            angle = (tooth + step / steps_per_tooth) * 2 * math.pi / teeth
            # ... расчет радиуса и координат
            
    contours.append(f'0 ' + ' '.join(f'{p:.1f}' for p in gear_points))
    
    return contours

def create_mechanical(polygon):
    create_mesh(['1', '1'], polygon,
                7,
                1.3, 1, True,
                show_meshing_result_method="gmsh",
                number_of_showed_class=3,
                is_saving_to_file=True,
                export_filename="tmp.txt")

contours = create_contours()

if __name__ == "__main__":
    create_mechanical(contours)
```

**Особенности:**
- Создает реалистичную шестерню с правильной формой зубьев
- Использует параметры зубчатого зацепления (делительная окружность, головка, ножка)
- Заменяет круги на квадраты для монтажных отверстий (для стабильности)
- **Обновленный алгоритм:** более точное моделирование зубьев шестерни

## Как создать свой собственный пример

### Шаг 1: Определите структуру контуров

Каждый контур должен быть представлен в виде строки:
```
<class_id> x1 y1 x2 y2 ... xn yn
```

Где:
- `class_id` — целое число, идентификатор класса (0, 1, 2, ...)
- `x1 y1, x2 y2, ...` — координаты вершин полигона

### Шаг 2: Создайте функцию-генератор контуров

```python
def create_my_contours():
    """Описание вашего примера"""
    contours = []
    
    # 1. Внешний контур (обычно класс 0)
    contours.append('0 0 0 400 0 400 300 0 300')
    
    # 2. Внутренние элементы
    contours.append('1 50 50 350 50 350 250 50 250')
    
    # 3. Детали
    contours.append('2 100 100 150 100 150 150 100 150')
    
    return contours
```

### Шаг 3: Создайте функцию генерации сетки

```python
from kt_service.ai_tools.mesh_tools.femm_generator import create_mesh

def create_my_mesh(polygon):
    create_mesh(['1', '1'], polygon,
                7,  # Размер элемента сетки
                1.3,  # Порог для слияния коллинеарных точек
                1,  # Толщина "кожи"
                True,  # Показывать внутренние контуры
                show_meshing_result_method="gmsh",
                number_of_showed_class=3,
                is_saving_to_file=True,
                export_filename="my_output.txt")
```

### Шаг 4: Запустите генерацию

```python
if __name__ == "__main__":
    contours = create_my_contours()
    create_my_mesh(contours)
```

## Параметры функции `create_mesh`

```python
create_mesh(
    pixel_spacing,          # Соотношение пиксель/мм [x, y] (например ['1', '1'])
    polygons,               # Список контуров
    lc=7,                   # Размер элемента сетки (меньше = мельче сетка)
    distance_threshold=1.3, # Порог для слияния коллинеарных точек
    skin_width=0,           # Толщина "кожи" (0 - нет, >0 - добавляет внешний слой)
    is_show_inner_contours=False, # Показывать внутренние контуры
    show_meshing_result_method="gmsh", # "gmsh" (GUI) или "no" (без визуализации)
    number_of_showed_class=-1,    # Какой класс скрыть (-1 - все видимы)
    is_saving_to_file=True,       # Сохранять в файл (ЗАМЕНА is_exporting_to_femm)
    export_filename="output.txt"  # Имя файла для экспорта
)
```

## Советы по созданию контуров

### 1. **Порядок точек**
- Точки должны следовать последовательно по контуру
- Первая и последняя точка обычно совпадают (замкнутый контур)

### 2. **Классификация элементов**
- Класс 0 обычно используется для внешнего контура
- Разные классы для разных материалов/свойств
- Классы должны быть целыми числами

### 3. **Оптимизация геометрии**
- Избегайте очень маленьких элементов
- Используйте `distance_threshold` для упрощения контуров
- Для сложных кривых используйте больше точек

### 4. **Пример простого прямоугольника**
```python
# Прямоугольник 100x100 с отверстием
contours = [
    '0 0 0 100 0 100 100 0 100',  # Внешний контур
    '1 20 20 80 20 80 80 20 80'   # Внутреннее отверстие
]
```

### 5. **Пример круга (аппроксимация многоугольником)**
```python
import math

def create_circle_contour(cx, cy, radius, segments=24, class_id=0):
    """Создает контур круга как многоугольник"""
    points = []
    for i in range(segments):
        angle = i * 2 * math.pi / segments
        x = cx + radius * math.cos(angle)
        y = cy + radius * math.sin(angle)
        points.extend([x, y])
    
    # Замыкаем круг
    points.extend([cx + radius, cy])
    
    return f'{class_id} ' + ' '.join(f'{p:.1f}' for p in points)
```

## Обработка ошибок

2. **Нечетное количество координат** — убедитесь, что у каждого контура четное количество чисел (пары x,y)
3. **Некорректный class_id** — должен быть целым числом
4. **Пересекающиеся контуры** — избегайте пересечений между контурами одного класса

## Выходные данные

### 1. **FEMM-формат файла** (экспортируется при `is_saving_to_file=True`)
```
# NODES
1 0.000000 0.000000
2 100.000000 0.000000
...

# TRIANGLES
1 2 3 0
4 5 6 1
...
```

### 2. **Визуализация через Gmsh GUI**
- Интерактивный просмотр сетки
- Навигация и масштабирование

## Запуск примеров

Каждый пример можно запустить независимо:

```bash
python create_architectural_facade.py
python create_generative_art.py
python create_mechanical_assembly.py
```


