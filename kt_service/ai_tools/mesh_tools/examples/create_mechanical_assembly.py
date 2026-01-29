from EITSynthAI.mesh_service.femm_generator import test_module


def create_mechanical_assembly():
    """Создает сложный механический узел"""
    import math

    contours = []

    # Основная большая шестерня (внешний контур)
    center_x, center_y = 200, 200
    radius_outer = 150
    radius_inner = 100
    teeth = 12

    # Внешний контур шестерни
    gear_points = []
    for i in range(teeth * 2):
        angle = i * math.pi / teeth
        if i % 2 == 0:
            r = radius_outer
        else:
            r = radius_outer - 20
        x = center_x + r * math.cos(angle)
        y = center_y + r * math.sin(angle)
        gear_points.extend([x, y])

    contours.append(f'0 ' + ' '.join(f'{p:.1f}' for p in gear_points))

    # Внутренняя круглая часть
    circle_points = []
    for i in range(24):
        angle = i * 2 * math.pi / 24
        x = center_x + radius_inner * math.cos(angle)
        y = center_y + radius_inner * math.sin(angle)
        circle_points.extend([x, y])

    contours.append(f'1 ' + ' '.join(f'{p:.1f}' for p in circle_points))

    # Шпоночный паз
    contours.append('2 200 80 220 80 220 120 200 120')

    # Монтажные отверстия (4 штуки)
    hole_radius = 15
    for i in range(4):
        angle = i * math.pi / 2
        hole_x = center_x + 80 * math.cos(angle)
        hole_y = center_y + 80 * math.sin(angle)

        hole_points = []
        for j in range(12):
            a = j * 2 * math.pi / 12
            x = hole_x + hole_radius * math.cos(a)
            y = hole_y + hole_radius * math.sin(a)
            hole_points.extend([x, y])

        contours.append(f'3 ' + ' '.join(f'{p:.1f}' for p in hole_points))

    # Декоративные вырезы
    for i in range(6):
        angle = i * math.pi / 3
        start_x = center_x + 110 * math.cos(angle)
        start_y = center_y + 110 * math.sin(angle)
        end_x = center_x + 130 * math.cos(angle)
        end_y = center_y + 130 * math.sin(angle)

        contours.append(f'4 {start_x:.1f} {start_y:.1f} {end_x:.1f} {end_y:.1f}')

    # Внутренний механизм (малая шестерня)
    small_gear_x, small_gear_y = 350, 200
    small_radius = 60

    small_gear_points = []
    for i in range(8 * 2):
        angle = i * math.pi / 8
        if i % 2 == 0:
            r = small_radius
        else:
            r = small_radius - 10
        x = small_gear_x + r * math.cos(angle)
        y = small_gear_y + r * math.sin(angle)
        small_gear_points.extend([x, y])

    contours.append(f'5 ' + ' '.join(f'{p:.1f}' for p in small_gear_points))

    # Соединительная пластина
    contours.append('6 200 200 350 200 340 180 210 180')

    return contours


test_module(create_mechanical_assembly())
