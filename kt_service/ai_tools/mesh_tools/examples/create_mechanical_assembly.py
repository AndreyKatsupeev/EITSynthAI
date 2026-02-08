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

            # Форма зуба - упрощенная
            if step < steps_per_tooth / 3:
                # Подъем к головке зуба
                r = pitch_radius + dedendum + (addendum * (step * 3 / steps_per_tooth))
            elif step < 2 * steps_per_tooth / 3:
                # Верх зуба
                r = pitch_radius + dedendum + addendum
            else:
                # Спуск от зуба
                r = pitch_radius + dedendum + (addendum * (1 - (step - 2 * steps_per_tooth / 3) * 3 / steps_per_tooth))

            x = center_x + r * math.cos(angle)
            y = center_y + r * math.sin(angle)
            gear_points.extend([x, y])

    contours.append(f'0 ' + ' '.join(f'{p:.1f}' for p in gear_points))

    # 2. Втулка (внутренний круг)
    hub_radius = 60
    hub_points = []
    for i in range(24):
        angle = i * 2 * math.pi / 24
        x = center_x + hub_radius * math.cos(angle)
        y = center_y + hub_radius * math.sin(angle)
        hub_points.extend([x, y])

    contours.append(f'1 ' + ' '.join(f'{p:.1f}' for p in hub_points))

    # # 3. Шпоночный паз
    # contours.append('2 200 40 220 40 220 80 200 80')

    # 4. Монтажные отверстия
    for i in range(4):
        angle = i * math.pi / 2
        hole_x = center_x + 90 * math.cos(angle)
        hole_y = center_y + 90 * math.sin(angle)

        # Простой квадрат вместо круга для стабильности
        contours.append(f'3 {hole_x - 10:.1f} {hole_y - 10:.1f} {hole_x + 10:.1f} {hole_y - 10:.1f} '
                        f'{hole_x + 10:.1f} {hole_y + 10:.1f} {hole_x - 10:.1f} {hole_y + 10:.1f}')

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
