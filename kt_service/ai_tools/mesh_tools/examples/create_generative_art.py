from EITSynthAI.mesh_service.femm_generator import test_module


def create_generative_art():
    """Создает абстрактное генеративное искусство"""
    import random
    import math

    contours = []

    random.seed(42)

    contours.append('0 0 0 500 0 500 500 0 500')

    for layer in range(5):
        points = []
        num_points = 20 + layer * 5

        for i in range(num_points):
            t = i / (num_points - 1)
            angle = t * 2 * math.pi

            # Случайные колебания
            noise = random.uniform(-0.1, 0.1) * (5 - layer)

            radius = 150 + layer * 30 + 50 * math.sin(angle * (2 + layer)) + 30 * noise
            x = 250 + radius * math.cos(angle)
            y = 250 + radius * math.sin(angle)

            points.extend([x, y])

        contours.append(f'{layer + 1} ' + ' '.join(f'{p:.1f}' for p in points))

    # Случайные спирали
    for spiral in range(3):
        points = []
        turns = 2 + spiral
        steps = 50

        center_x = 100 + spiral * 150
        center_y = 150 + spiral * 100

        for i in range(steps):
            t = i / steps
            angle = t * 2 * math.pi * turns

            radius = 20 + t * 60
            x = center_x + radius * math.cos(angle)
            y = center_y + radius * math.sin(angle)

            points.extend([x, y])

        contours.append(f'{6 + spiral} ' + ' '.join(f'{p:.1f}' for p in points))

    # Геометрические фигуры
    figures = [
        (250, 400, 40, 6),  # шестиугольник
        (400, 100, 30, 8),  # восьмиугольник
        (100, 100, 35, 5),  # пятиугольник
    ]

    for idx, (cx, cy, radius, sides) in enumerate(figures):
        points = []
        for i in range(sides):
            angle = i * 2 * math.pi / sides
            x = cx + radius * math.cos(angle)
            y = cy + radius * math.sin(angle)
            points.extend([x, y])

        contours.append(f'{9 + idx} ' + ' '.join(f'{p:.1f}' for p in points))

    return contours


test_module(create_generative_art())
