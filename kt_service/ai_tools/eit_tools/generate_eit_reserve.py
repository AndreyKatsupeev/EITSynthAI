"""
Генерация датасета ЭИТ с использованием gmsh.
Сохраняет результаты в generation_results/
"""

import numpy as np
import json
import os
from typing import List, Dict, Tuple
from dataclasses import dataclass, field
from shapely.geometry import Polygon, Point
from shapely.ops import unary_union
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import matplotlib
matplotlib.use('Agg')  # для работы в докере без дисплея
import matplotlib.pyplot as plt
import gmsh
from shapely.geometry import Polygon, MultiPolygon, LineString, Point


# ============================================================
# 1. КОНФИГУРАЦИЯ
# ============================================================
@dataclass
class EITConfig:
    # Проводимости (S/m) на частоте ~50 kHz
    conductivity: Dict[str, float] = field(default_factory=lambda: {
        "fat":   0.02,
        "bone":  0.01,
        "muscle": 0.35,
        "lung_exhale": 0.25,
        "lung_inhale": 0.08,
        "background": 0.2,
    })

    # Дыхание
    breath_period_sec: float = 4.0
    breath_fps: float = 10.0
    breath_n_cycles: int = 3

    # Меш (gmsh)
    mesh_characteristic_length: float = 5.0  # характерный размер элемента, мм
    mesh_order: int = 1                      # порядок элементов (1=линейные)

    # Электроды
    n_electrodes: int = 16
    electrode_radius_mm: float = 5.0
    inject_current_A: float = 5e-3

    # Протокол
    drive_pattern: str = "adjacent"

    # Выход
    output_dir: str = "/app/generation_results"
    dataset_name: str = "thorax_breath"


# ============================================================
# 2. ПАРСИНГ ВАШЕГО ФОРМАТА
# ============================================================
def parse_list_crd(list_crd: List[str], pixel_spacing: float) -> Dict[int, List[np.ndarray]]:
    """
    Парсит список строк в словарь {class_id: [polygon_coords_mm, ...]}.
    Класс 4 = тело (внешний контур).
    """
    tissues: Dict[int, List[np.ndarray]] = {}
    for item in list_crd:
        parts = item.strip().split()
        if len(parts) < 7:
            continue
        cls_id = int(parts[0])
        coords = np.array([float(x) for x in parts[1:]]).reshape(-1, 2)
        coords_mm = coords * pixel_spacing
        if cls_id not in tissues:
            tissues[cls_id] = []
        tissues[cls_id].append(coords_mm)
    return tissues

def get_largest_polygon(geom):
    """
    Извлекает самый большой полигон из геометрии.
    Если это Polygon — возвращает его.
    Если MultiPolygon — возвращает полигон с максимальной площадью.
    """
    
    if isinstance(geom, Polygon):
        return geom
    elif isinstance(geom, MultiPolygon):
        # Берём самый большой полигон
        return max(geom.geoms, key=lambda p: p.area)
    else:
        raise ValueError(f"Неожиданный тип геометрии: {type(geom)}")


def build_geometry(tissues: Dict[int, List[np.ndarray]]) -> Dict[int, Polygon]:
    """
    Объединяем полигоны одного класса.
    Если результат — MultiPolygon, берём самый большой полигон.
    """
    from shapely.geometry import Polygon, MultiPolygon
    
    result = {}
    for cls_id, polys in tissues.items():
        valid = [Polygon(p) for p in polys if Polygon(p).is_valid and Polygon(p).area > 1.0]
        if valid:
            merged = unary_union(valid)
            # Если MultiPolygon — берём самый большой
            if isinstance(merged, MultiPolygon):
                merged = get_largest_polygon(merged)
            result[cls_id] = merged
    return result


# ============================================================
# 3. ПОСТРОЕНИЕ МЕША ЧЕРЕЗ GMSH
# ============================================================
def simplify_polygon(poly: Polygon, tolerance: float = 2.0) -> Polygon:
    """
    Упрощает полигон, уменьшая количество точек.
    tolerance: допуск упрощения в мм (чем больше, тем меньше точек)
    """
    simplified = poly.simplify(tolerance, preserve_topology=True)
    # Если упрощение сделало полигон невалидным, возвращаем оригинал
    if not simplified.is_valid or simplified.area < 1.0:
        return poly
    return simplified


def clean_polygon(poly: Polygon, tolerance: float = 3.0) -> Polygon:
    """
    Надёжное упрощение + очистка полигона.
    1. simplify с большим tolerance
    2. buffer(0) — устраняет самопересечения
    3. Проверка валидности
    4. Если MultiPolygon — берём самый большой
    """
    from shapely.geometry import Polygon, MultiPolygon
    
    # Упрощаем
    simplified = poly.simplify(tolerance, preserve_topology=True)
    
    # buffer(0) — стандартный трюк для исправления самопересечений
    cleaned = simplified.buffer(0)
    
    if cleaned.is_empty or cleaned.area < 1.0:
        # Если буфер всё сломал — пробуем без simplify
        cleaned = poly.buffer(0)
    
    if isinstance(cleaned, MultiPolygon):
        cleaned = max(cleaned.geoms, key=lambda p: p.area)
    
    if not isinstance(cleaned, Polygon):
        raise ValueError(f"Не удалось получить Polygon: {type(cleaned)}")
    
    return cleaned


def ensure_ccw(coords: np.ndarray) -> np.ndarray:
    """
    Гарантирует, что контур идёт против часовой стрелки (CCW).
    gmsh требует CCW для внешних контуров и CW для отверстий.
    """
    # Считаем signed area
    x = coords[:, 0]
    y = coords[:, 1]
    signed_area = 0.5 * np.sum(x[:-1] * y[1:] - x[1:] * y[:-1])
    if signed_area < 0:
        # CW → разворачиваем
        coords = coords[::-1]
    return coords


def add_polygon_to_gmsh(poly: Polygon, char_length: float, 
                        max_points: int = 80, is_hole: bool = False):
    """
    Добавляет полигон в gmsh как curve loop.
    Возвращает tag curve loop.
    """
    ext = np.array(poly.exterior.coords)[:-1]  # без замыкающей
    
    # Ограничиваем число точек
    if len(ext) > max_points:
        from shapely.geometry import LineString
        ls = LineString(np.vstack([ext, ext[0]]))
        xs = np.linspace(0, ls.length, max_points, endpoint=False)
        ext = np.array([ls.interpolate(x).coords[0] for x in xs])
    
    # Направление обхода
    ext = ensure_ccw(ext)
    if is_hole:
        ext = ext[::-1]  # для отверстий — CW
    
    # Создаём точки
    point_tags = []
    for x, y in ext:
        pt = gmsh.model.geo.addPoint(float(x), float(y), 0.0, char_length)
        point_tags.append(pt)
    
    # Создаём линии
    line_tags = []
    n = len(point_tags)
    for i in range(n):
        p1 = point_tags[i]
        p2 = point_tags[(i + 1) % n]
        line = gmsh.model.geo.addLine(p1, p2)
        line_tags.append(line)
    
    # Создаём curve loop
    loop = gmsh.model.geo.addCurveLoop(line_tags)
    return loop


def build_mesh_gmsh(
    body_poly: Polygon,
    inclusions: Dict[int, Polygon],
    char_length: float = 10.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, int]]:
    """
    Строит 2D треугольный mesh через gmsh.
    Меш строится ТОЛЬКО по внешнему контуру тела (без отверстий).
    Ткани размечаются после по центроидам элементов.
    Возвращает (points, triangles, element_physicals, phys_mapping).
    """
    from shapely.geometry import Polygon, MultiPolygon, LineString, Point
    
    # Physical tags
    PHYS_BACKGROUND = 1
    PHYS_BONE       = 2
    PHYS_MUSCLE     = 3
    PHYS_LUNG       = 4
    PHYS_FAT        = 5
    
    DICOM_TO_PHYS = {0: PHYS_BONE, 1: PHYS_MUSCLE, 2: PHYS_LUNG, 3: PHYS_FAT}
    
    def ensure_polygon(geom):
        if isinstance(geom, Polygon):
            return geom
        elif isinstance(geom, MultiPolygon):
            return max(geom.geoms, key=lambda p: p.area)
        else:
            raise ValueError(f"Неожиданный тип геометрии: {type(geom)}")
    
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 1)
    gmsh.option.setNumber("General.Verbosity", 3)
    gmsh.model.add("eit_model")
    
    # === 1. Чистим ТОЛЬКО внешний контур тела ===
    body_clean = clean_polygon(body_poly, tolerance=5.0)
    body_clean = ensure_polygon(body_clean)
    print(f"[gmsh] Body: {len(np.array(body_clean.exterior.coords))} точек")
    
    # === 2. Чистим включения (для последующей разметки по центроидам) ===
    incl_clean = {}
    for cls_id, poly in inclusions.items():
        if isinstance(poly, MultiPolygon):
            poly = max(poly.geoms, key=lambda p: p.area)
        poly = ensure_polygon(poly)
        
        cleaned = clean_polygon(poly, tolerance=4.0)
        cleaned = ensure_polygon(cleaned)
        
        if cleaned.area < 20.0:
            print(f"[gmsh] Класс {cls_id}: слишком мал ({cleaned.area:.1f} мм²), пропускаю")
            continue
        
        # Обрезаем по телу
        if not body_clean.contains(cleaned):
            cleaned = cleaned.intersection(body_clean)
            cleaned = ensure_polygon(cleaned)
            if cleaned.is_empty or cleaned.area < 20.0:
                continue
        
        incl_clean[cls_id] = cleaned
        print(f"[gmsh] Класс {cls_id}: площадь {cleaned.area:.1f} мм²")
    
    # === 3. Меш ТОЛЬКО по внешнему контуру (БЕЗ отверстий!) ===
    body_loop = add_polygon_to_gmsh(body_clean, char_length, max_points=150, is_hole=False)
    gmsh.model.geo.synchronize()
    
    body_surf = gmsh.model.geo.addPlaneSurface([body_loop])
    gmsh.model.geo.synchronize()
    
    gmsh.model.addPhysicalGroup(2, [body_surf], tag=PHYS_BACKGROUND)
    gmsh.model.setPhysicalName(2, PHYS_BACKGROUND, "background")
    
    # === 4. Настройки mesh ===
    gmsh.option.setNumber("Mesh.CharacteristicLengthMax", char_length)
    gmsh.option.setNumber("Mesh.CharacteristicLengthMin", char_length / 3.0)
    gmsh.option.setNumber("Mesh.Algorithm", 6)  # Frontal-Delaunay
    gmsh.option.setNumber("Mesh.RecombineAll", 0)
    
    debug_path = "/app/generation_results/debug_mesh.msh"
    gmsh.write(debug_path)
    print(f"[gmsh] Отладочный mesh: {debug_path}")
    
    print("[gmsh] Начинаю генерацию mesh...")
    try:
        gmsh.model.mesh.generate(2)
        print("[gmsh] Mesh сгенерирован успешно")
    except Exception as e:
        print(f"[gmsh] ОШИБКА генерации mesh: {e}")
        gmsh.finalize()
        raise
    
    # === 5. Извлекаем mesh ===
    node_tags, node_coords, _ = gmsh.model.mesh.getNodes()
    node_tags = np.array(node_tags)
    node_coords = np.array(node_coords).reshape(-1, 3)
    tag_to_idx = {int(t): i for i, t in enumerate(node_tags)}
    points = node_coords[:, :2]
    
    elem_types, elem_tags, elem_node_tags = gmsh.model.mesh.getElements(2)
    
    triangles = []
    for etype, etags, enodes in zip(elem_types, elem_tags, elem_node_tags):
        if etype == 2:
            enodes_arr = np.array(enodes).reshape(-1, 3)
            tris = np.array([[tag_to_idx[int(n)] for n in row] for row in enodes_arr])
            triangles.append(tris)
    
    triangles = np.vstack(triangles) if triangles else np.zeros((0, 3), dtype=int)
    
    if len(triangles) == 0:
        gmsh.finalize()
        raise RuntimeError("[gmsh] Не удалось сгенерировать ни одного элемента!")
    
    # === 6. Размечаем ткани по центроидам ===
    centroids = points[triangles].mean(axis=1)  # (M, 2)
    element_physicals = np.full(len(triangles), PHYS_BACKGROUND, dtype=int)
    
    # Приоритет: кость > мышцы > жир > лёгкие (мелкие ткани перекрывают крупные)
    priority_order = [0, 1, 3, 2]  # DICOM классы
    for cls_id in priority_order:
        if cls_id not in incl_clean:
            continue
        poly = incl_clean[cls_id]
        phys_tag = DICOM_TO_PHYS[cls_id]
        for i, c in enumerate(centroids):
            if poly.contains(Point(c[0], c[1])):
                element_physicals[i] = phys_tag
    
    phys_mapping = {
        "background": PHYS_BACKGROUND,
        "bone": PHYS_BONE,
        "muscle": PHYS_MUSCLE,
        "lung": PHYS_LUNG,
        "fat": PHYS_FAT,
    }
    
    gmsh.finalize()
    
    print(f"[gmsh] Итого: {len(points)} узлов, {len(triangles)} элементов")
    unique, counts = np.unique(element_physicals, return_counts=True)
    tag_names = {v: k for k, v in phys_mapping.items()}
    print(f"[gmsh] Physical tags: "
          f"{dict(zip([tag_names.get(t, str(t)) for t in unique.tolist()], counts.tolist()))}")
    
    return points, triangles, element_physicals, phys_mapping


# ============================================================
# 4. НАЗНАЧЕНИЕ ПРОВОДИМОСТЕЙ
# ============================================================
def assign_conductivity(
    elem_physicals: np.ndarray,
    cfg: EITConfig,
    lung_sigma: float,
    phys_mapping: Dict[str, int] = None,
) -> np.ndarray:
    """
    Назначает проводимость каждому элементу по physical tag.
    """
    if phys_mapping is None:
        # Значения по умолчанию
        phys_mapping = {
            "background": 1,
            "bone": 2,
            "muscle": 3,
            "lung": 4,
            "fat": 5,
        }
    
    # physical_tag -> conductivity
    tag_to_sigma = {
        phys_mapping["background"]: cfg.conductivity["background"],
        phys_mapping["bone"]:       cfg.conductivity["bone"],
        phys_mapping["muscle"]:     cfg.conductivity["muscle"],
        phys_mapping["lung"]:       lung_sigma,
        phys_mapping["fat"]:        cfg.conductivity["fat"],
    }
    
    sigma = np.array([tag_to_sigma.get(int(p), cfg.conductivity["background"])
                      for p in elem_physicals])
    return sigma


# ============================================================
# 5. FEM-РЕШАТЕЛЬ
# ============================================================
def assign_conductivity(
    elem_physicals: np.ndarray,
    cfg: EITConfig,
    lung_sigma: float,
    phys_mapping: Dict[str, int] = None,
) -> np.ndarray:
    """Назначает проводимость каждому элементу по physical tag."""
    if phys_mapping is None:
        phys_mapping = {
            "background": 1, "bone": 2, "muscle": 3, "lung": 4, "fat": 5
        }
    
    tag_to_sigma = {
        phys_mapping["background"]: cfg.conductivity["background"],
        phys_mapping["bone"]:       cfg.conductivity["bone"],
        phys_mapping["muscle"]:     cfg.conductivity["muscle"],
        phys_mapping["lung"]:       lung_sigma,
        phys_mapping["fat"]:        cfg.conductivity["fat"],
    }
    
    sigma = np.array([tag_to_sigma.get(int(p), cfg.conductivity["background"])
                      for p in elem_physicals])
    return sigma


def place_electrodes(body_poly: Polygon, n: int) -> np.ndarray:
    """
    ★ ЗДЕСЬ ЗАДАЮТСЯ КООРДИНАТЫ ЭЛЕКТРОДОВ ★
    Расставляет n электродов равномерно по внешнему контуру тела.
    Замените эту функцию на свой алгоритм оптимизации.
    """
    ext = np.array(body_poly.exterior.coords)
    diffs = np.diff(ext, axis=0)
    seg_len = np.linalg.norm(diffs, axis=1)
    cum = np.concatenate([[0], np.cumsum(seg_len)])
    L = cum[-1]
    positions = np.linspace(0, L, n, endpoint=False)

    electrodes = []
    for s in positions:
        idx = np.searchsorted(cum, s, side="right") - 1
        idx = min(idx, len(seg_len) - 1)
        t = (s - cum[idx]) / (seg_len[idx] + 1e-12)
        pt = ext[idx] * (1 - t) + ext[idx + 1] * t
        electrodes.append(pt)
    return np.array(electrodes)


def find_nearest_nodes(points: np.ndarray, electrodes: np.ndarray) -> np.ndarray:
    """Для каждой точки электрода ищем индекс ближайшего узла меша."""
    from scipy.spatial import cKDTree
    tree = cKDTree(points)
    _, idx = tree.query(electrodes)
    return idx


def solve_forward(K, inj_nodes: Tuple[int, int], meas_nodes: np.ndarray, 
                  current: float = 5e-3, ground_node: int = 0) -> np.ndarray:
    """Решает K*V = I с точечными источниками тока."""
    n = K.shape[0]
    I = np.zeros(n)
    I[inj_nodes[0]] += current
    I[inj_nodes[1]] -= current

    keep = np.arange(n) != ground_node
    K_red = K[keep][:, keep]
    I_red = I[keep]

    V_red = spla.spsolve(K_red.tocsc(), I_red)
    V = np.zeros(n)
    V[keep] = V_red
    return V[meas_nodes]


def build_drive_pattern(n_elec: int, pattern: str = "adjacent"):
    """Возвращает список пар (inj+, inj-) и пар (meas+, meas-)."""
    frames = []
    for k in range(n_elec):
        if pattern == "adjacent":
            inj = (k, (k + 1) % n_elec)
            meas_pairs = [((k + 1 + m) % n_elec, (k + 2 + m) % n_elec)
                          for m in range(n_elec - 2)]
        else:
            inj = (k, (k + n_elec // 2) % n_elec)
            meas_pairs = [((k + 1 + m) % n_elec, (k + 2 + m) % n_elec)
                          for m in range(n_elec - 2)]
        frames.append((inj, meas_pairs))
    return frames

def assemble_stiffness(points: np.ndarray, tris: np.ndarray, sigma: np.ndarray):
    """Собирает глобальную матрицу жёсткости K."""
    n_pts = len(points)
    n_tri = len(tris)
    rows, cols, vals = [], [], []

    for e in range(n_tri):
        idx = tris[e]
        xy = points[idx]
        x = xy[:, 0]
        y = xy[:, 1]
        area = 0.5 * abs(
            (x[1] - x[0]) * (y[2] - y[0]) - (x[2] - x[0]) * (y[1] - y[0])
        )
        if area < 1e-12:
            continue
        b = np.array([y[1] - y[2], y[2] - y[0], y[0] - y[1]])
        c = np.array([x[2] - x[1], x[0] - x[2], x[1] - x[0]])
        Ke = sigma[e] * (np.outer(b, b) + np.outer(c, c)) / (4.0 * area)

        for i_loc in range(3):
            for j_loc in range(3):
                rows.append(idx[i_loc])
                cols.append(idx[j_loc])
                vals.append(Ke[i_loc, j_loc])

    K = sp.coo_matrix((vals, (rows, cols)), shape=(n_pts, n_pts)).tocsr()
    return K

# ============================================================
# 6. ГЛАВНАЯ ФУНКЦИЯ
# ============================================================
def generate_eit_dataset(list_crd: List[str], cfg: EITConfig = None):
    if cfg is None:
        cfg = EITConfig()

    os.makedirs(cfg.output_dir, exist_ok=True)

    # --- 6.1. Парсинг и геометрия ---
    pixel_spacing = float(list_crd[0])
    tissues = parse_list_crd(list_crd[2:], pixel_spacing)
    geometries = build_geometry(tissues)

    # Тело — класс 4
    if 4 not in geometries:
        raise ValueError("Класс 4 (контур тела) не найден в данных!")
    body_poly = geometries.pop(4)

    # Остальные классы — включения
    inclusions = geometries  # {0: bone, 1: muscle, 2: lung, 3: fat}

    # Центрируем
    cx, cy = body_poly.centroid.x, body_poly.centroid.y
    from shapely.affinity import translate
    body_poly = translate(body_poly, -cx, -cy)
    inclusions = {k: translate(v, -cx, -cy) for k, v in inclusions.items()}

    # --- 6.2. Меш через gmsh ---
    points, tris, elem_physicals, phys_mapping = build_mesh_gmsh(
        body_poly, inclusions,
        char_length=3.0,  # ← БЫЛО 5.0
    )

    print(f"[EIT] Меш: {len(points)} узлов, {len(tris)} элементов")

    # --- 6.3. Электроды ---
    electrodes_xy = place_electrodes(body_poly, cfg.n_electrodes)
    elec_nodes = find_nearest_nodes(points, electrodes_xy)

    # Сохраняем координаты электродов
    np.savetxt(
        os.path.join(cfg.output_dir, f"{cfg.dataset_name}_electrodes.csv"),
        np.hstack([electrodes_xy, elec_nodes[:, None]]),
        delimiter=",",
        header="x_mm,y_mm,node_id",
        comments="",
    )

    # --- 6.4. Протокол измерений ---
    drive = build_drive_pattern(cfg.n_electrodes, cfg.drive_pattern)

    # --- 6.5. Имитация дыхания ---
    fps = cfg.breath_fps
    dt = 1.0 / fps
    n_frames = int(cfg.breath_period_sec * cfg.breath_fps * cfg.breath_n_cycles)
    sigma_lung_ex = cfg.conductivity["lung_exhale"]
    sigma_lung_in = cfg.conductivity["lung_inhale"]

    t = np.arange(n_frames) * dt
    breath_phase = np.sin(2 * np.pi * t / cfg.breath_period_sec)
    lung_sigma_series = 0.5 * (sigma_lung_ex + sigma_lung_in) \
                        - 0.5 * (sigma_lung_ex - sigma_lung_in) * breath_phase

    sigma_base = assign_conductivity(
        elem_physicals, cfg, 
        lung_sigma=sigma_lung_ex,
        phys_mapping=phys_mapping,
    )
    lung_phys_tag = phys_mapping["lung"]
    lung_elems = np.where(elem_physicals == lung_phys_tag)[0]

    # Сбор данных
    n_meas = cfg.n_electrodes * (cfg.n_electrodes - 2)
    voltages = np.zeros((n_frames, n_meas))

    print(f"[EIT] Электродов: {cfg.n_electrodes}, кадров: {n_frames}")

    for f in range(n_frames):
        sigma = sigma_base.copy()
        if len(lung_elems) > 0:
            sigma[lung_elems] = lung_sigma_series[f]

        K = assemble_stiffness(points, tris, sigma)

        row = 0
        for (inj, meas_pairs) in drive:
            inj_nodes = (elec_nodes[inj[0]], elec_nodes[inj[1]])
            V = solve_forward(K, inj_nodes, elec_nodes, current=cfg.inject_current_A)
            for (m_plus, m_minus) in meas_pairs:
                voltages[f, row] = V[m_plus] - V[m_minus]
                row += 1

        if (f + 1) % 10 == 0 or f == 0:
            print(f"  frame {f+1}/{n_frames}, "
                  f"σ_lung={lung_sigma_series[f]:.3f} S/m, "
                  f"|V|_max={np.abs(voltages[f]).max():.4f} V")

    # --- 6.6. Сохранение ---
    meta = {
        "n_electrodes": cfg.n_electrodes,
        "inject_current_A": cfg.inject_current_A,
        "drive_pattern": cfg.drive_pattern,
        "breath_period_sec": cfg.breath_period_sec,
        "breath_fps": cfg.breath_fps,
        "breath_n_cycles": cfg.breath_n_cycles,
        "pixel_spacing_mm": pixel_spacing,
        "n_nodes": int(len(points)),
        "n_elements": int(len(tris)),
        "conductivities": cfg.conductivity,
        "electrode_nodes": elec_nodes.tolist(),
        "phys_mapping": phys_mapping, 
    }
    with open(os.path.join(cfg.output_dir, f"{cfg.dataset_name}_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    np.save(os.path.join(cfg.output_dir, f"{cfg.dataset_name}_voltages.npy"), voltages)
    np.save(os.path.join(cfg.output_dir, f"{cfg.dataset_name}_breath_phase.npy"), breath_phase)
    np.savez(os.path.join(cfg.output_dir, f"{cfg.dataset_name}_mesh.npz"),
             points=points, tris=tris, elem_physicals=elem_physicals)
    
    print(f"[EIT] ✓ Датасет сохранён в {cfg.output_dir}")

    # Визуализация
    _plot_debug(points, tris, elem_physicals, electrodes_xy, cfg, phys_mapping)
    return voltages, breath_phase


def _plot_debug(points, tris, elem_physicals, electrodes_xy, cfg, phys_mapping):
    """
    Показывает mesh с РАЗНЫМИ ЦВЕТАМИ для каждой ткани.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 7))
    
    # === Левая картинка: ткани разными цветами ===
    ax1 = axes[0]
    
    # Маппинг physical tag -> цвет
    tag_colors = {
        phys_mapping["background"]: "#8B4513",  # коричневый (фон/мышцы)
        phys_mapping["bone"]:       "#FFFFFF",  # белый (кость)
        phys_mapping["muscle"]:     "#FF6347",  # красный (мышцы)
        phys_mapping["lung"]:       "#87CEEB",  # голубой (лёгкие)
        phys_mapping["fat"]:        "#FFD700",  # жёлтый (жир)
    }
    
    tag_names = {
        phys_mapping["background"]: "Фон/мышцы",
        phys_mapping["bone"]:       "Кость",
        phys_mapping["muscle"]:     "Мышцы",
        phys_mapping["lung"]:       "Лёгкие",
        phys_mapping["fat"]:        "Жир",
    }
    
    # Рисуем каждый элемент своим цветом
    from matplotlib.patches import Polygon as MplPolygon
    from matplotlib.collections import PatchCollection
    
    patches = []
    colors = []
    for i, tri in enumerate(tris):
        pts = points[tri]
        patch = MplPolygon(pts, closed=True)
        patches.append(patch)
        phys_tag = elem_physicals[i]
        colors.append(tag_colors.get(phys_tag, "#808080"))
    
    collection = PatchCollection(patches, facecolor=colors, edgecolor='none', alpha=0.8)
    ax1.add_collection(collection)
    
    # Электроды
    ax1.plot(electrodes_xy[:, 0], electrodes_xy[:, 1], 'ro', markersize=10, label='Электроды', zorder=10)
    for i, (x, y) in enumerate(electrodes_xy):
        ax1.text(x, y, str(i), color='white', fontsize=8, ha='center', va='center', fontweight='bold', zorder=11)
    
    ax1.set_xlim(points[:, 0].min() - 20, points[:, 0].max() + 20)
    ax1.set_ylim(points[:, 1].min() - 20, points[:, 1].max() + 20)
    ax1.set_aspect('equal')
    ax1.set_title('Ткани (разные цвета)')
    ax1.legend(loc='upper right')
    
    # Легенда тканей
    from matplotlib.lines import Line2D
    legend_elements = [Line2D([0], [0], marker='s', color='w', 
                             markerfacecolor=tag_colors[tag], markersize=10,
                             label=tag_names[tag])
                      for tag in sorted(tag_colors.keys())]
    ax1.legend(handles=legend_elements, loc='lower right', fontsize=8)
    ax1.grid(True, alpha=0.3)
    
    # === Правая картинка: проводимости (log scale) ===
    ax2 = axes[1]
    
    # Назначаем проводимости
    sigma = np.zeros(len(tris))
    for i, tag in enumerate(elem_physicals):
        if tag == phys_mapping["bone"]:
            sigma[i] = cfg.conductivity["bone"]
        elif tag == phys_mapping["muscle"]:
            sigma[i] = cfg.conductivity["muscle"]
        elif tag == phys_mapping["lung"]:
            sigma[i] = cfg.conductivity["lung_exhale"]
        elif tag == phys_mapping["fat"]:
            sigma[i] = cfg.conductivity["fat"]
        else:
            sigma[i] = cfg.conductivity["background"]
    
    trip = ax2.tripcolor(points[:, 0], points[:, 1], tris,
                        facecolors=np.log10(sigma), cmap="viridis", shading="flat")
    ax2.plot(electrodes_xy[:, 0], electrodes_xy[:, 1], 'ro', markersize=8, label='Электроды')
    plt.colorbar(trip, ax=ax2, label='log10(σ)')
    ax2.set_aspect('equal')
    ax2.set_title('Проводимости (log scale)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(cfg.output_dir, f"{cfg.dataset_name}_mesh_preview.png"), dpi=150)
    plt.close()
    
    # Статистика
    print(f"\n[Preview] Распределение тканей:")
    unique, counts = np.unique(elem_physicals, return_counts=True)
    for tag, count in zip(unique.tolist(), counts.tolist()):
        name = tag_names.get(tag, f"tag_{tag}")
        print(f"  {name:15s}: {count:5d} элементов ({count/len(tris)*100:5.1f}%)")


if __name__ == "__main__":
    sample_list_crd = ['0.616', '0.616', '3 37 119 15 160 0 205 0 297 3 300 4 313 13 345 21 363 49 410 70 434 86 442 130 456 132 461 198 461 200 456 231 451 260 450 287 451 315 456 317 461 372 461 375 456 400 450 441 433 470 398 498 343 506 318 508 301 511 297 511 213 508 211 505 190 498 166 487 141 474 119 461 109 437 97 421 92 409 84 386 76 365 73 345 65 329 63 326 58 180 58 178 63 163 66 136 77 124 76 99 85 87 93 63 101 37 119', '0 289 407 284 407 283 408 279 408 278 409 278 410 277 411 277 412 276 413 275 413 278 413 279 412 281 412 283 410 289 410 289 407', '0 192 406 191 407 190 407 189 408 185 408 184 409 174 409 192 409 192 406', '0 130 406 130 409 139 409 136 409 135 408 133 408 132 407 131 407 130 406', '0 210 409 210 414 213 414 216 417 216 425 224 425 224 422 226 420 229 420 230 419 232 419 233 418 241 418 242 417 243 417 243 416 244 415 249 415 250 414 260 414 261 413 273 413 272 413 271 412 263 412 262 411 247 411 246 410 245 410 244 409 245 408 245 407 246 406 245 405 241 405 240 404 237 404 236 403 233 403 232 402 228 402 227 403 226 403 225 404 224 404 223 405 222 405 221 406 217 406 214 409 210 409', '0 249 402 250 402 251 403 259 403 260 402 261 402 257 402 256 401 252 401 251 402 249 402', '0 271 394 271 398 273 400 280 400 282 402 284 402 285 403 289 403 289 402 290 401 291 401 292 402 292 406 294 408 294 409 296 411 296 412 297 413 297 417 329 417 329 412 325 412 324 411 323 411 322 410 321 410 319 408 317 408 314 405 314 402 294 402 293 401 293 394 286 394 286 401 285 402 284 401 283 401 282 400 282 394 271 394', '0 78 358 78 366 81 366 84 369 84 370 85 371 85 372 89 376 89 377 92 380 92 381 94 383 95 383 98 386 99 386 101 388 101 389 102 390 103 390 105 392 105 393 106 394 108 394 112 398 113 398 116 401 116 405 121 405 121 397 116 397 115 396 115 395 114 394 114 393 113 393 112 392 111 392 110 391 110 390 109 389 108 389 106 387 106 386 105 385 104 385 102 383 102 382 101 381 100 381 98 379 98 378 96 376 95 376 94 375 94 374 90 370 90 369 86 365 85 365 82 362 82 358 78 358', '0 429 342 426 342 426 345 425 346 425 348 421 352 421 354 420 355 420 356 418 358 418 359 417 360 417 362 413 366 413 367 411 369 410 369 409 370 409 371 408 372 407 372 406 373 405 373 405 374 402 377 401 377 401 378 398 381 397 381 396 382 396 383 395 384 394 384 392 386 392 387 391 388 390 388 389 389 386 389 386 393 397 393 397 389 398 388 398 387 399 386 399 385 402 382 403 382 414 371 415 371 417 369 417 368 419 366 419 365 421 363 422 363 422 362 423 361 423 360 425 358 429 358 429 342', '0 58 314 58 326 61 326 62 327 63 327 64 328 64 330 65 331 65 332 66 333 66 337 77 337 77 328 74 328 71 325 71 323 69 321 68 321 66 319 66 314 58 314', '0 449 290 445 290 445 294 443 296 442 296 441 297 441 300 439 302 439 303 437 305 434 305 434 317 440 317 440 314 443 311 443 310 446 307 449 307 449 290', '0 331 252 331 256 332 257 332 260 333 261 333 263 333 252 331 252', '0 54 250 54 253 53 254 50 254 50 269 53 269 54 270 54 271 55 272 55 276 56 277 56 278 59 281 59 285 61 285 62 286 62 308 64 308 64 305 63 304 63 303 62 302 62 292 63 291 63 286 64 285 65 285 65 255 62 255 61 254 61 253 60 252 60 250 54 250', '0 64 236 62 236 62 251 62 240 63 239 63 237 64 236', '0 453 234 449 234 449 238 446 241 442 241 442 269 447 269 447 266 451 262 451 260 454 257 457 257 457 238 454 238 453 237 453 234', '0 313 234 313 236 315 236 316 235 322 235 323 234 326 234 328 236 328 234 313 234', '0 231 231 231 233 237 233 236 232 233 232 232 231 231 231', '0 168 226 166 226 166 253 167 253 167 246 168 245 168 235 167 234 167 229 168 228 168 226', '0 198 194 198 202 200 202 200 200 199 199 199 196 198 195 198 194', '0 450 174 450 184 454 184 455 185 455 186 457 188 458 188 461 191 462 191 464 193 464 194 465 195 465 196 466 197 466 199 467 200 467 203 468 204 468 207 469 208 469 210 472 213 472 215 473 216 473 217 476 220 476 221 480 225 480 229 489 229 489 211 485 211 484 210 484 209 483 208 483 206 481 204 480 204 479 203 478 203 472 197 472 196 471 195 471 192 470 191 470 189 468 187 468 186 467 185 467 184 463 180 462 180 459 177 459 174 450 174', '0 65 166 53 166 53 169 48 174 48 175 46 177 45 177 45 178 44 179 44 180 40 184 38 184 34 188 33 188 32 189 32 190 30 192 29 192 28 193 28 195 27 196 27 197 25 199 22 199 22 209 33 209 33 205 34 204 35 204 36 203 38 203 39 202 40 202 46 196 46 194 47 193 47 188 48 187 48 185 50 183 50 182 54 178 55 178 56 177 57 177 58 176 59 176 60 175 65 175 65 166', '0 426 162 426 181 429 181 432 184 432 190 433 191 433 195 434 196 434 197 435 198 435 199 436 200 436 204 437 205 437 207 438 208 438 213 439 213 440 214 441 213 445 213 445 184 442 184 440 182 440 178 439 177 439 175 435 171 435 168 434 167 434 166 433 165 433 162 426 162', '0 93 154 87 154 87 158 86 159 85 159 85 160 84 161 84 162 81 165 81 166 80 167 80 169 79 170 79 171 77 173 77 174 76 175 76 178 75 179 75 180 74 181 74 182 73 183 73 184 72 185 72 190 71 191 71 192 69 194 66 194 66 221 70 221 70 218 73 215 74 215 74 213 75 212 75 210 76 209 76 208 77 207 77 206 78 205 78 198 79 197 79 196 82 193 82 186 83 185 83 184 86 181 86 170 87 169 87 164 88 163 88 162 90 160 93 160 93 154', '0 161 106 161 109 162 108 170 108 171 109 171 106 161 106', '0 269 90 260 90 260 100 259 101 259 104 258 106 251 112 246 113 242 116 234 116 233 117 230 117 229 118 222 118 222 133 229 133 229 125 230 124 235 124 237 126 239 126 240 127 246 127 250 124 258 120 263 120 265 122 268 123 273 128 274 130 274 133 282 133 282 129 283 128 287 127 290 125 292 126 292 135 290 136 286 141 279 141 278 140 277 141 274 141 266 148 263 148 262 149 260 148 254 148 250 144 249 144 246 140 234 140 234 169 237 169 240 172 242 180 245 183 245 185 249 188 249 193 271 193 271 189 278 182 279 178 281 176 281 174 282 173 282 169 283 168 283 153 290 148 290 146 291 145 291 136 294 134 294 132 298 128 301 128 301 118 296 118 293 121 288 121 287 120 285 120 282 117 277 116 274 113 271 112 269 110 269 108 270 107 270 103 271 102 271 100 270 99 270 95 269 94 269 90', '1 438 435 437 436 434 436 431 439 429 439 428 440 425 440 424 441 423 441 422 442 421 442 420 443 419 443 418 444 415 444 414 445 413 445 411 447 409 447 408 448 405 448 404 449 403 449 402 450 401 450 400 451 398 451 397 452 393 452 392 453 389 453 387 455 385 455 384 456 378 456 377 457 375 457 374 458 373 458 373 461 372 462 350 462 372 462 373 461 380 461 381 460 385 460 386 459 390 459 391 458 393 458 394 457 397 457 398 456 401 456 402 455 404 455 405 454 407 454 408 453 410 453 411 452 412 452 413 451 414 451 415 450 417 450 418 449 419 449 420 448 421 448 422 447 423 447 424 446 425 446 427 444 428 444 433 439 434 439 438 435', '1 211 415 212 416 212 417 215 420 215 425 215 417 213 415 211 415', '1 141 410 142 410 143 411 146 411 147 412 155 412 156 413 160 413 161 412 169 412 170 411 171 411 172 410 141 410', '1 227 402 219 402 218 403 213 403 214 403 216 405 213 408 212 408 214 408 217 405 221 405 222 404 223 404 224 403 225 403 226 402 227 402', '1 296 413 295 412 295 411 293 409 293 408 291 406 291 402 290 402 290 403 289 404 285 404 284 403 282 403 280 401 273 401 270 398 270 396 269 397 268 397 265 400 259 400 258 401 257 401 261 401 262 402 259 404 251 404 248 402 247 403 238 403 237 402 233 402 236 402 237 403 240 403 241 404 245 404 247 406 245 409 246 409 247 410 262 410 263 411 271 411 274 413 273 414 261 414 260 415 250 415 249 416 244 416 244 417 241 419 233 419 232 420 230 420 229 421 226 421 225 422 225 423 226 423 227 422 229 422 232 420 237 420 238 419 241 419 242 418 244 418 245 417 248 417 249 416 256 416 257 415 261 415 262 414 267 414 268 415 271 415 272 414 280 414 283 412 285 412 286 411 287 411 283 411 281 413 279 413 278 414 275 414 274 413 275 412 276 412 276 411 277 410 277 409 279 407 283 407 284 406 289 406 290 407 290 409 292 411 292 412 293 413 293 414 295 414 296 415 296 413', '1 283 394 283 400 284 400 285 401 285 394 286 393 293 393 294 394 294 401 314 401 315 402 315 405 317 407 319 407 321 409 322 409 323 410 324 410 325 411 329 411 330 412 330 417 329 418 307 418 309 418 310 419 314 419 315 418 334 418 335 417 342 417 342 416 341 415 341 414 340 413 338 413 337 412 334 412 330 408 329 408 326 405 325 405 324 404 317 404 313 400 312 400 311 399 310 399 309 398 310 397 311 397 312 396 313 396 301 396 300 395 298 395 297 394 296 394 295 393 292 393 291 392 289 392 288 393 286 393 285 394 283 394', '1 239 378 241 380 242 380 245 383 242 380 241 380 239 378', '1 274 371 273 372 272 372 272 373 269 376 268 376 268 377 263 382 263 383 260 386 260 387 259 388 259 389 259 388 260 387 260 386 263 383 263 382 274 371', '1 281 367 280 367 279 368 278 368 276 370 278 368 279 368 280 367 281 367', '1 83 355 83 362 85 364 86 364 91 369 91 370 95 374 95 375 96 375 99 378 99 379 100 380 101 380 103 382 103 383 104 384 105 384 107 386 107 387 108 388 109 388 111 390 111 391 112 391 113 392 114 392 115 393 115 394 116 395 116 396 118 396 117 396 116 395 116 394 114 392 113 392 99 378 99 377 98 376 97 376 94 373 94 372 93 372 92 371 92 370 91 369 91 368 90 367 89 367 88 366 88 365 87 364 87 363 85 361 85 360 83 358 83 355', '1 295 354 294 355 293 355 291 357 293 355 294 355 295 354', '1 421 351 420 352 420 354 419 355 419 356 417 358 417 359 416 360 416 361 412 365 412 366 406 372 405 372 404 373 404 374 402 376 401 376 397 380 396 380 395 381 395 382 393 384 392 384 391 385 390 385 387 388 386 388 389 388 390 387 391 387 391 386 394 383 395 383 395 382 397 380 398 380 400 378 400 377 401 376 402 376 404 374 404 373 405 372 406 372 407 371 408 371 408 370 410 368 411 368 412 367 412 366 416 362 416 360 417 359 417 358 419 356 419 355 420 354 420 352 421 351', '1 69 348 67 350 67 351 66 352 66 354 68 356 68 361 71 364 71 365 72 366 72 367 75 370 75 371 76 372 76 374 77 375 78 375 80 377 80 378 85 383 86 383 88 385 88 386 89 387 91 387 92 388 94 388 95 389 97 389 98 390 99 390 100 391 101 391 104 394 105 394 109 398 110 398 114 402 114 403 115 403 115 401 113 399 112 399 108 395 106 395 104 393 104 392 103 391 102 391 100 389 100 388 99 387 98 387 95 384 94 384 91 381 91 380 88 377 88 376 84 372 84 371 83 370 83 369 81 367 78 367 77 366 77 359 75 357 75 356 74 355 74 354 71 351 71 350 70 349 70 348 69 348', '1 431 344 430 344 430 358 429 359 425 359 424 360 424 361 423 362 423 363 422 364 421 364 420 365 420 366 418 368 418 369 415 372 414 372 403 383 402 383 400 385 400 386 399 387 399 388 398 389 398 390 406 382 407 382 408 381 409 381 409 380 412 377 413 377 413 376 415 374 416 374 420 370 421 370 422 369 422 368 424 366 425 366 426 365 426 364 427 363 427 362 428 361 428 360 430 358 430 357 434 353 434 351 433 350 433 346 431 344', '1 425 344 425 345 424 346 424 348 423 349 424 348 424 346 425 345 425 344', '1 72 338 73 338 74 339 75 339 76 340 76 339 75 338 72 338', '1 61 327 63 329 63 330 64 331 64 332 65 333 64 332 64 331 63 330 63 328 62 328 61 327', '1 67 315 67 319 68 320 69 320 70 321 67 318 67 315', '1 448 308 446 308 444 310 444 311 441 314 443 312 444 312 445 311 445 310 446 309 447 309 448 308', '1 444 290 444 292 443 293 443 294 440 297 440 300 440 297 442 295 443 295 444 294 444 290', '1 192 287 192 285 194 283 195 283 196 282 200 282 201 281 202 281 203 282 206 282 210 278 210 276 209 275 208 275 207 276 206 276 204 278 204 279 202 281 200 281 199 280 198 280 197 279 195 279 193 277 193 274 193 276 192 277 192 287', '1 294 272 282 272 281 273 281 274 280 275 280 277 279 278 279 279 280 280 280 285 278 287 274 287 270 284 263 284 262 283 261 283 258 280 257 280 252 275 251 275 250 274 248 274 246 272 246 274 247 275 247 283 244 287 244 288 243 289 243 293 241 295 239 295 235 291 234 291 233 290 233 288 234 287 234 285 233 284 232 284 231 283 226 283 225 284 224 284 224 285 223 286 223 290 219 294 219 295 216 300 216 302 214 304 213 304 209 308 208 308 208 309 207 310 207 312 205 315 205 317 204 318 204 324 203 325 203 329 202 330 202 332 201 333 201 338 202 339 203 342 205 344 206 344 207 345 207 346 213 352 214 352 216 355 221 356 229 363 232 363 233 364 240 364 243 362 243 356 242 355 242 353 241 352 241 333 245 330 250 332 251 333 251 334 253 336 254 336 255 337 255 338 258 339 261 343 264 343 265 344 267 344 268 345 269 345 270 346 272 346 273 347 284 347 288 344 293 343 299 337 299 336 301 333 302 333 303 334 303 336 303 335 305 332 305 329 304 328 304 323 303 322 303 320 304 319 304 317 303 316 303 312 304 311 304 305 305 304 305 290 304 289 304 288 305 287 303 284 303 281 302 280 301 280 299 278 299 277 298 276 297 276 294 272', '1 51 270 52 271 52 275 48 279 48 285 49 286 49 288 50 289 50 290 51 290 52 291 52 292 53 293 53 297 54 298 55 298 55 297 56 296 56 292 55 291 55 287 56 286 56 284 57 283 58 283 58 281 55 278 55 277 54 276 54 272 53 271 53 270 51 270', '1 466 243 464 243 462 245 458 245 458 257 457 258 454 258 452 260 452 262 450 264 451 264 452 263 452 262 454 260 457 263 461 263 462 262 463 262 464 261 464 259 465 258 465 256 464 255 464 250 465 249 465 248 467 246 467 245 466 244 466 243', '1 247 249 245 247 241 247 240 246 239 246 235 242 235 239 234 238 233 238 233 240 232 241 232 242 229 245 227 245 226 246 225 245 224 245 223 244 222 244 221 243 212 243 210 245 209 245 207 243 204 243 203 242 203 241 204 240 204 239 204 240 203 241 203 242 204 243 205 243 207 245 207 253 204 256 204 257 203 258 203 259 202 260 204 258 204 257 206 255 208 257 208 258 209 259 209 262 208 263 207 263 206 264 205 264 204 265 204 269 205 270 206 270 208 272 210 272 210 271 211 270 211 269 216 264 217 264 221 260 222 260 223 259 236 259 237 258 238 258 240 256 241 256 242 255 243 255 244 254 245 254 246 253 246 252 247 251 247 249', '1 448 237 446 239 445 238 445 237 445 240 446 240 448 238 448 237', '1 330 238 329 237 328 237 326 235 323 235 322 236 316 236 315 237 312 237 309 239 309 241 308 242 308 245 306 247 305 246 294 246 294 248 293 249 292 249 291 248 289 248 287 246 287 242 286 241 286 237 285 237 283 239 282 239 281 240 266 240 265 239 264 239 265 239 266 240 269 240 271 242 269 244 269 246 270 246 273 248 283 248 284 247 286 247 287 248 289 248 292 250 291 251 291 252 289 254 285 254 282 252 280 252 279 251 277 251 276 252 274 252 272 254 271 254 270 255 265 255 264 256 260 256 258 258 258 259 257 260 257 263 258 263 261 265 263 265 264 266 267 266 268 265 270 265 271 264 287 264 288 265 290 265 291 264 299 264 300 265 301 265 307 271 308 271 311 273 311 275 312 276 312 279 311 280 310 280 314 280 316 283 319 284 322 287 323 287 324 288 324 290 324 289 325 288 326 289 326 288 324 286 324 282 325 281 325 280 323 278 323 275 325 272 328 272 329 271 330 272 331 272 332 273 332 274 330 276 329 276 327 278 327 279 327 278 332 274 332 271 333 270 333 264 332 263 332 261 331 260 331 257 330 256 330 238', '1 503 232 502 231 497 231 496 232 495 232 494 233 494 235 493 236 493 237 494 238 498 238 499 237 501 237 502 236 502 235 503 234 503 232', '1 248 232 247 231 247 230 246 229 246 228 245 229 241 229 240 230 240 233 239 234 244 234 244 232 243 231 243 230 244 229 245 229 246 230 246 232 248 232', '1 223 229 224 230 227 230 228 229 230 229 228 229 227 228 224 228 223 229', '1 220 224 220 226 219 227 218 227 217 228 216 228 214 230 219 230 220 229 220 224', '1 70 223 68 225 68 227 67 228 67 231 67 228 68 227 68 225 70 223', '1 178 222 175 225 169 225 169 228 168 229 168 234 169 235 169 245 168 246 168 253 169 254 170 254 171 253 172 253 174 251 176 251 177 250 178 250 178 249 181 246 180 245 181 244 189 244 188 243 188 242 186 240 185 240 182 237 182 226 181 225 181 224 180 224 178 222', '1 75 213 75 215 74 216 73 216 71 218 71 221 71 219 72 218 72 217 73 216 74 216 75 215 75 213', '1 80 196 80 197 79 198 79 203 79 198 80 197 80 196', '1 452 185 454 187 454 188 455 189 455 190 459 194 459 195 460 196 460 197 461 198 460 199 460 200 459 201 459 202 458 203 458 205 457 206 455 206 454 207 451 207 449 209 450 210 450 211 451 212 451 216 452 217 452 218 457 223 459 223 460 224 462 224 463 225 463 227 465 229 467 229 468 230 469 230 471 228 471 227 472 226 472 225 473 224 475 224 476 225 477 225 479 227 479 225 475 221 475 220 472 217 472 216 471 215 471 213 468 210 468 208 467 207 467 204 466 203 466 200 465 199 465 197 464 196 464 195 463 194 463 193 462 192 461 192 458 189 457 189 454 186 454 185 452 185', '1 85 183 84 184 84 185 83 186 83 191 83 186 84 185 84 184 85 183', '1 73 182 71 184 71 190 70 191 70 192 69 193 68 193 69 193 70 192 70 191 71 190 71 185 72 184 72 183 73 182', '1 63 176 60 176 59 177 58 177 57 178 56 178 55 179 54 179 51 182 51 183 49 185 49 187 48 188 48 193 47 194 47 196 40 203 39 203 38 204 36 204 35 205 34 205 34 208 35 208 36 207 39 207 40 208 41 208 42 209 42 218 41 219 41 221 40 222 40 230 43 233 43 234 44 235 44 241 45 242 45 250 46 251 47 251 49 253 49 254 48 255 48 260 47 261 47 262 48 263 48 265 49 266 49 268 49 254 50 253 53 253 53 251 52 251 49 248 49 246 51 244 51 242 52 241 52 240 54 238 54 237 55 236 55 232 56 231 56 221 55 220 55 219 53 217 53 216 52 215 52 213 51 212 51 211 49 209 49 207 50 206 51 206 52 207 65 207 65 196 59 196 58 195 50 195 49 194 49 191 50 190 51 190 52 189 54 189 57 192 58 191 58 181 59 180 60 180 60 179 63 176', '1 78 171 77 172 76 172 76 174 75 175 75 178 74 179 74 180 74 179 75 178 75 175 76 174 76 173 78 171', '1 198 193 199 194 199 195 200 196 200 199 201 200 201 202 203 204 203 205 204 206 204 207 205 208 205 209 207 211 208 211 209 212 210 212 213 215 214 215 218 219 219 219 221 221 221 219 222 218 235 218 236 219 241 219 244 216 244 215 246 213 246 212 247 211 247 210 254 203 253 203 252 202 252 201 251 200 251 197 253 195 261 195 262 196 263 196 264 197 266 195 269 195 270 194 269 195 268 194 249 194 248 193 248 188 247 187 246 187 244 185 244 183 241 180 241 179 240 178 240 174 239 173 239 172 237 170 236 170 236 172 238 174 238 176 239 177 239 181 238 182 237 182 236 181 236 180 235 179 234 179 231 176 229 176 228 175 225 175 222 172 222 170 221 171 220 171 219 172 217 172 216 173 214 173 213 174 212 174 210 176 208 176 207 177 206 177 206 178 201 183 201 184 200 185 200 189 199 190 199 192 198 193', '1 457 172 456 173 459 173 460 174 460 177 462 179 463 179 468 184 468 185 469 186 469 187 471 189 471 191 472 192 472 195 473 196 473 197 478 202 479 202 480 203 481 203 484 206 484 208 485 209 485 210 489 210 490 211 490 229 489 230 484 230 490 230 491 229 492 229 492 228 496 224 497 224 498 223 499 223 499 220 500 219 500 212 503 209 503 207 504 206 504 200 503 200 502 199 501 199 500 198 500 195 499 194 499 190 498 189 495 189 493 191 492 191 491 192 488 192 487 191 486 191 484 189 484 188 483 187 483 185 482 184 482 183 480 181 480 179 479 178 479 174 476 171 476 170 474 168 470 168 466 172 457 172', '1 95 158 95 159 93 161 90 161 89 162 89 163 88 164 88 169 87 170 87 179 87 171 88 170 88 168 89 167 89 166 88 165 89 164 89 162 90 161 91 162 91 163 95 159 95 158', '1 58 162 57 162 56 161 54 161 53 160 52 160 51 159 49 159 48 158 42 158 41 159 40 159 38 161 38 162 36 164 36 169 35 170 35 173 33 175 32 175 31 176 29 176 28 177 27 177 27 178 25 180 20 180 19 179 18 179 15 182 15 183 14 184 13 184 13 185 12 186 12 190 11 191 11 198 8 201 8 204 7 205 7 213 8 214 8 224 9 225 9 226 10 227 10 228 11 229 11 231 12 232 12 235 13 236 14 236 17 239 19 239 20 238 20 237 19 237 16 234 16 232 15 231 15 228 16 227 16 224 18 222 19 222 19 217 20 216 20 215 21 214 22 214 23 215 24 215 25 216 28 216 29 215 30 215 31 214 31 213 32 212 32 210 22 210 21 209 21 199 22 198 25 198 26 197 26 196 27 195 27 193 29 191 30 191 31 190 31 189 33 187 34 187 38 183 40 183 43 180 43 179 44 178 44 177 45 176 46 176 47 175 47 174 52 169 52 166 53 165 63 165 62 165 61 164 60 164 58 162', '1 34 157 30 157 29 158 29 160 28 161 26 161 25 162 25 167 26 167 26 166 27 165 27 164 33 158 34 158 34 157', '1 226 147 227 148 227 149 228 150 228 162 227 163 227 164 227 163 228 162 228 149 227 148 227 147 226 147', '1 429 145 429 151 431 153 431 154 432 155 432 159 430 161 429 161 433 161 434 162 434 165 435 166 435 167 436 168 436 171 438 173 437 172 437 170 436 169 438 167 439 167 440 168 442 168 445 171 446 171 447 170 448 170 449 169 449 168 451 166 451 165 452 164 453 164 455 162 455 159 454 158 454 157 453 156 452 156 445 149 444 149 443 148 441 148 438 145 429 145', '1 94 137 92 137 91 136 80 136 78 134 77 134 77 136 74 139 71 139 71 140 70 141 70 142 69 143 68 143 68 144 67 145 67 146 64 149 64 154 65 155 66 155 69 158 69 162 71 164 74 164 76 162 76 161 78 159 79 159 80 158 80 157 81 156 85 156 86 157 85 158 84 158 83 159 83 161 82 162 82 163 81 164 80 164 80 166 79 167 79 169 79 167 80 166 80 165 83 162 83 161 84 160 84 159 85 158 86 158 86 154 87 153 91 153 90 152 90 148 91 147 91 145 92 144 92 143 93 142 93 141 94 140 94 137', '1 219 132 219 133 220 134 220 135 223 138 223 139 224 140 224 143 224 139 220 135 220 134 219 133 219 132', '1 305 131 303 133 303 134 301 136 303 134 303 133 305 131', '1 215 128 216 129 216 130 217 131 216 130 216 129 215 128', '1 387 122 388 123 389 123 390 124 389 123 388 123 387 122', '1 127 122 126 123 125 123 124 124 125 123 126 123 127 122', '1 230 125 230 133 229 134 228 134 229 135 231 133 231 127 233 125 235 127 236 127 237 128 242 128 243 129 245 129 246 130 246 134 245 135 244 135 241 138 238 138 237 139 235 139 246 139 248 141 248 142 249 143 250 143 254 147 260 147 261 148 262 148 263 147 266 147 274 140 277 140 278 139 279 140 286 140 290 135 291 135 292 136 292 145 291 146 291 148 288 151 287 151 284 153 284 168 283 169 283 173 282 174 282 176 280 178 280 180 279 181 279 182 272 189 272 192 273 191 274 191 275 190 275 189 279 185 279 184 282 179 282 177 283 176 283 173 284 172 284 169 286 166 286 164 287 163 287 161 288 160 288 159 287 159 287 160 286 161 285 160 285 155 284 154 286 152 287 152 290 150 291 151 291 155 291 153 292 152 292 151 294 149 293 150 292 149 292 141 293 140 293 138 295 136 295 135 296 134 296 132 299 129 298 129 295 132 295 134 293 136 292 136 291 135 291 126 290 126 287 128 285 128 284 129 283 129 283 133 282 134 274 134 273 133 273 130 272 129 272 128 268 124 265 123 263 121 258 121 257 122 256 122 255 123 250 125 248 127 247 127 246 128 240 128 239 127 237 127 235 125 230 125', '1 205 120 206 120 210 124 211 124 212 125 212 126 213 127 212 126 212 125 211 124 210 124 206 120 205 120', '1 155 110 151 110 150 111 148 111 147 112 148 112 149 111 155 111 155 110', '1 426 93 427 93 428 94 430 94 431 95 434 95 435 96 437 96 438 97 439 97 440 98 441 98 442 99 444 99 445 100 446 100 447 101 448 101 450 103 452 103 453 104 454 104 456 106 457 106 458 107 459 107 460 108 461 108 464 111 465 111 466 112 467 112 468 113 466 111 466 110 464 108 464 107 461 104 461 103 458 100 457 100 456 99 454 99 453 98 451 98 450 97 447 97 446 96 442 96 441 95 438 95 437 94 429 94 428 93 426 93', '1 267 89 269 89 270 90 270 94 271 95 271 99 272 100 272 102 271 103 271 107 270 108 270 110 271 111 272 111 273 112 274 112 277 115 279 115 280 116 282 116 285 119 287 119 288 120 292 120 291 119 290 119 288 117 289 116 294 116 295 115 296 115 299 112 308 112 309 111 313 111 314 112 316 112 317 113 322 113 323 112 329 112 330 111 335 111 338 108 343 108 344 107 353 107 354 106 360 106 361 105 361 104 360 103 360 100 359 99 359 97 361 95 372 95 373 94 372 93 371 93 370 92 369 92 366 89 365 89 364 88 360 88 359 87 358 87 356 85 353 85 352 84 344 84 343 83 339 83 338 82 337 82 336 81 334 81 333 80 322 80 321 79 308 79 307 78 305 78 304 79 291 79 290 80 279 80 275 84 275 85 274 86 273 86 271 88 270 88 269 89 267 89', '1 141 89 141 90 142 91 152 91 152 90 153 89 154 89 155 88 158 88 160 90 159 91 158 91 154 95 153 95 151 97 150 97 151 98 152 98 153 99 155 99 156 100 161 100 163 102 164 102 165 103 172 103 173 104 181 104 183 106 184 106 185 107 194 107 196 109 196 110 197 111 202 111 205 108 207 108 208 107 210 107 213 104 221 104 223 106 223 107 224 108 225 108 229 112 231 112 233 114 231 116 233 116 234 115 242 115 243 114 244 114 246 112 248 112 249 111 251 111 253 109 254 109 257 106 257 105 258 104 258 101 259 100 259 91 257 91 255 89 255 88 253 86 251 86 250 85 248 85 244 81 242 81 241 80 231 80 230 79 229 79 228 80 216 80 215 79 208 79 207 78 201 78 200 77 194 77 193 78 191 78 190 79 185 79 184 80 172 80 171 81 167 81 166 82 165 82 162 84 153 84 152 85 149 85 147 87 146 87 145 88 142 88 141 89', '1 359 69 360 70 361 70 362 71 364 71 365 72 368 72 369 73 372 73 373 74 374 74 373 74 372 73 371 73 370 72 369 72 368 71 365 71 364 70 360 70 359 69', '2 105 144 89 168 76 215 63 240 62 254 66 255 63 302 72 325 78 328 76 338 84 358 93 371 122 397 122 401 140 409 173 409 219 401 247 402 265 399 271 393 289 391 332 400 362 400 393 383 415 361 428 340 434 318 433 305 438 303 444 288 444 270 441 269 444 226 441 216 437 213 431 184 425 181 423 159 410 141 386 121 366 112 342 112 318 121 297 143 276 190 255 203 241 220 222 219 222 227 238 233 241 228 246 227 253 236 266 239 281 239 286 236 288 246 292 248 294 245 306 246 313 233 328 233 331 238 331 251 334 252 333 274 325 282 327 293 314 281 309 281 305 288 307 326 301 347 258 391 253 391 237 377 205 359 184 330 184 308 191 289 191 277 206 253 206 245 202 242 204 235 220 223 197 202 197 193 206 176 226 165 227 150 222 138 211 125 198 116 162 109 129 121 105 144', '4 219 62 218 63 217 63 216 63 215 63 214 63 213 63 212 63 211 63 210 63 209 63 208 63 207 63 206 63 205 63 204 63 203 64 202 64 201 64 200 64 199 64 198 64 197 64 196 64 195 64 194 64 193 64 192 64 191 64 190 64 189 64 188 64 187 64 186 64 185 65 184 65 183 65 182 65 181 66 180 66 179 66 178 66 177 66 176 66 175 67 174 67 173 67 172 67 171 67 170 68 169 68 168 68 167 68 166 68 165 69 164 69 163 69 162 69 161 70 160 70 159 70 158 71 157 71 156 71 155 72 154 72 153 73 152 73 151 74 150 74 149 75 148 75 147 76 146 76 145 77 144 77 143 77 142 77 141 77 140 78 139 78 138 78 137 78 136 78 135 78 134 78 133 78 132 78 131 78 130 78 129 78 128 78 127 79 126 79 125 79 124 80 123 80 122 81 121 81 120 81 119 82 118 82 117 82 116 83 115 83 114 84 113 84 112 84 111 84 110 85 109 85 108 85 107 86 106 86 105 86 104 87 103 87 102 88 101 88 100 89 99 89 98 90 97 91 96 91 95 92 94 93 93 94 92 94 91 95 90 95 89 96 88 96 87 96 86 97 85 97 84 98 83 98 82 98 81 98 80 99 79 99 78 99 77 100 76 100 75 100 74 101 73 101 72 102 71 102 70 102 69 103 68 103 67 104 66 104 65 105 64 105 63 106 62 106 61 107 60 107 59 108 58 109 57 109 56 110 55 110 54 111 53 111 52 112 51 112 50 113 49 114 48 114 47 115 46 115 45 116 44 117 43 117 42 118 41 119 40 119 39 120 38 121 38 122 37 123 36 124 36 125 35 126 35 127 34 128 33 129 33 130 32 131 32 132 31 133 31 134 30 135 30 136 29 137 29 138 28 139 28 140 27 141 27 142 26 143 26 144 25 145 25 146 24 147 24 148 23 149 23 150 22 151 22 152 21 153 21 154 21 155 20 156 20 157 19 158 19 159 18 160 18 161 18 162 17 163 17 164 17 165 16 166 16 167 15 168 15 169 15 170 14 171 14 172 14 173 13 174 13 175 13 176 12 177 12 178 12 179 11 180 11 181 11 182 10 183 10 184 10 185 10 186 9 187 9 188 9 189 9 190 8 191 8 192 8 193 7 194 7 195 7 196 7 197 7 198 6 199 6 200 6 201 6 202 5 203 5 204 5 205 5 206 5 207 4 208 4 209 4 210 4 211 4 212 4 213 3 214 3 215 3 216 3 217 3 218 3 219 2 220 2 221 2 222 2 223 2 224 2 225 2 226 2 227 1 228 1 229 1 230 1 231 1 232 1 233 1 234 1 235 1 236 1 237 1 238 1 239 0 240 0 241 0 242 0 243 0 244 0 245 0 246 0 247 0 248 0 249 0 250 0 251 0 252 0 253 0 254 0 255 0 256 0 257 0 258 0 259 0 260 0 261 0 262 0 263 0 264 0 265 0 266 0 267 0 268 0 269 0 270 0 271 1 272 1 273 1 274 1 275 1 276 1 277 1 278 1 279 1 280 1 281 1 282 1 283 2 284 2 285 2 286 2 287 2 288 2 289 2 290 2 291 3 292 3 293 3 294 3 295 3 296 3 297 4 298 4 299 4 300 4 301 4 302 4 303 5 304 5 305 5 306 5 307 5 308 6 309 6 310 6 311 6 312 7 313 7 314 7 315 7 316 7 317 8 318 8 319 8 320 9 321 9 322 9 323 9 324 10 325 10 326 10 327 10 328 11 329 11 330 11 331 12 332 12 333 12 334 13 335 13 336 13 337 14 338 14 339 14 340 15 341 15 342 15 343 16 344 16 345 17 346 17 347 17 348 18 349 18 350 18 351 19 352 19 353 20 354 20 355 21 356 21 357 21 358 22 359 22 360 23 361 23 362 24 363 24 364 25 365 25 366 26 367 26 368 27 369 27 370 28 371 28 372 29 373 29 374 30 375 30 376 31 377 31 378 32 379 32 380 33 381 33 382 34 383 35 384 35 385 36 386 36 387 37 388 38 389 38 390 39 391 39 392 40 393 41 394 41 395 42 396 43 397 43 398 44 399 45 400 45 401 46 402 47 403 47 404 48 405 49 406 50 407 50 408 51 409 52 410 53 411 53 412 54 413 55 414 56 415 57 416 57 417 58 418 59 419 60 420 61 421 62 422 62 423 63 424 64 425 65 426 66 427 67 428 68 429 69 430 70 431 71 432 72 432 73 433 74 433 75 433 76 434 77 435 78 435 79 435 80 436 81 436 82 437 83 437 84 437 85 438 86 438 87 439 88 439 89 439 90 440 91 440 92 440 93 441 94 441 95 441 96 442 97 442 98 443 99 443 100 444 101 444 102 444 103 445 104 445 105 445 106 446 107 446 108 447 109 447 110 447 111 448 112 448 113 448 114 448 115 449 116 449 117 449 118 450 119 450 120 450 121 451 122 451 123 451 124 452 125 452 126 452 127 452 128 453 129 453 130 453 131 454 132 454 133 454 134 454 135 455 136 455 137 455 138 455 139 455 140 456 141 456 142 456 143 456 144 456 145 457 146 457 147 457 148 457 149 457 150 457 151 458 152 458 153 458 154 458 155 458 156 458 157 458 158 458 159 458 160 458 161 459 162 459 163 459 164 459 165 459 166 459 167 459 168 459 169 459 170 459 171 459 172 459 173 459 174 459 175 459 176 459 177 458 178 458 179 458 180 458 181 458 182 458 183 457 184 457 185 457 186 457 187 457 188 456 189 456 190 456 191 456 192 455 193 455 194 455 195 455 196 455 197 455 198 454 199 454 200 454 201 454 202 454 203 453 204 453 205 453 206 453 207 453 208 452 209 452 210 452 211 452 212 452 213 452 214 451 215 451 216 451 217 451 218 451 219 451 220 450 221 450 222 450 223 450 224 450 225 450 226 450 227 450 228 450 229 450 230 450 231 449 232 449 233 449 234 449 235 449 236 449 237 449 238 449 239 449 240 448 241 448 242 448 243 448 244 448 245 448 246 448 247 448 248 448 249 448 250 448 251 448 252 448 253 448 254 449 255 449 256 449 257 449 258 449 259 449 260 449 261 449 262 449 263 449 264 449 265 449 266 449 267 450 268 450 269 450 270 450 271 450 272 450 273 450 274 450 275 450 276 450 277 450 278 451 279 451 280 451 281 451 282 451 283 451 284 451 285 451 286 451 287 451 288 452 289 452 290 452 291 452 292 452 293 452 294 453 295 453 296 453 297 453 298 453 299 453 300 454 301 454 302 454 303 454 304 454 305 454 306 454 307 454 308 455 309 455 310 455 311 455 312 455 313 455 314 456 315 456 316 456 317 456 318 456 319 456 320 457 321 457 322 457 323 457 324 457 325 457 326 458 327 458 328 458 329 458 330 458 331 459 332 459 333 459 334 459 335 459 336 460 337 460 338 460 339 460 340 460 341 460 342 461 343 461 344 461 345 461 346 461 347 461 348 461 349 461 350 462 351 462 352 462 353 462 354 462 355 462 356 462 357 462 358 462 359 462 360 462 361 462 362 462 363 462 364 462 365 462 366 462 367 462 368 462 369 462 370 462 371 462 372 462 373 461 374 461 375 461 376 461 377 461 378 461 379 461 380 461 381 460 382 460 383 460 384 460 385 460 386 459 387 459 388 459 389 459 390 459 391 458 392 458 393 458 394 457 395 457 396 457 397 457 398 456 399 456 400 456 401 456 402 455 403 455 404 455 405 454 406 454 407 454 408 453 409 453 410 453 411 452 412 452 413 451 414 451 415 450 416 450 417 450 418 449 419 449 420 448 421 448 422 447 423 447 424 446 425 446 426 445 427 444 428 444 429 443 430 442 431 441 432 440 433 439 434 439 435 438 436 437 437 436 438 435 439 434 439 433 440 432 441 431 442 430 443 429 444 428 445 427 446 426 447 425 448 424 449 423 449 422 450 421 451 420 452 419 453 418 454 417 454 416 455 415 456 414 457 413 458 412 458 411 459 410 460 409 461 408 461 407 462 406 463 405 464 404 464 403 465 402 466 401 466 400 467 399 468 398 468 397 469 396 470 395 470 394 471 393 472 392 472 391 473 390 473 389 474 388 475 387 475 386 476 385 476 384 477 383 478 382 478 381 479 380 479 379 480 378 480 377 481 376 481 375 482 374 482 373 483 372 483 371 484 370 484 369 485 368 485 367 486 366 486 365 487 364 487 363 488 362 488 361 489 360 489 359 490 358 490 357 490 356 491 355 491 354 492 353 492 352 493 351 493 350 493 349 494 348 494 347 494 346 495 345 495 344 496 343 496 342 496 341 497 340 497 339 497 338 498 337 498 336 498 335 499 334 499 333 499 332 500 331 500 330 500 329 501 328 501 327 501 326 501 325 502 324 502 323 502 322 502 321 503 320 503 319 503 318 504 317 504 316 504 315 504 314 504 313 505 312 505 311 505 310 505 309 506 308 506 307 506 306 506 305 506 304 507 303 507 302 507 301 507 300 507 299 507 298 508 297 508 296 508 295 508 294 508 293 508 292 509 291 509 290 509 289 509 288 509 287 509 286 509 285 509 284 510 283 510 282 510 281 510 280 510 279 510 278 510 277 510 276 510 275 510 274 510 273 510 272 511 271 511 270 511 269 511 268 511 267 511 266 511 265 511 264 511 263 511 262 511 261 511 260 511 259 511 258 511 257 511 256 511 255 511 254 511 253 511 252 511 251 511 250 511 249 511 248 511 247 511 246 511 245 511 244 511 243 511 242 511 241 511 240 510 239 510 238 510 237 510 236 510 235 510 234 510 233 510 232 510 231 510 230 510 229 510 228 509 227 509 226 509 225 509 224 509 223 509 222 509 221 509 220 508 219 508 218 508 217 508 216 508 215 508 214 507 213 507 212 507 211 507 210 507 209 507 208 506 207 506 206 506 205 506 204 506 203 505 202 505 201 505 200 505 199 504 198 504 197 504 196 504 195 504 194 503 193 503 192 503 191 502 190 502 189 502 188 502 187 501 186 501 185 501 184 501 183 500 182 500 181 500 180 499 179 499 178 499 177 498 176 498 175 498 174 497 173 497 172 497 171 496 170 496 169 496 168 495 167 495 166 494 165 494 164 494 163 493 162 493 161 493 160 492 159 492 158 491 157 491 156 490 155 490 154 490 153 489 152 489 151 488 150 488 149 487 148 487 147 486 146 486 145 485 144 485 143 484 142 484 141 483 140 483 139 482 138 482 137 481 136 481 135 480 134 480 133 479 132 479 131 478 130 478 129 477 128 476 127 476 126 475 125 475 124 474 123 473 122 473 121 472 120 472 119 471 118 470 117 470 116 469 115 468 114 468 113 467 112 466 111 466 110 465 109 464 108 464 107 463 106 462 105 461 104 461 103 460 102 459 101 458 100 457 100 456 99 455 99 454 99 453 98 452 98 451 98 450 97 449 97 448 97 447 97 446 96 445 96 444 96 443 96 442 96 441 95 440 95 439 95 438 95 437 94 436 94 435 94 434 94 433 94 432 94 431 94 430 94 429 94 428 93 427 93 426 93 425 93 424 93 423 92 422 92 421 92 420 92 419 91 418 91 417 91 416 91 415 90 414 90 413 90 412 89 411 89 410 88 409 88 408 88 407 87 406 87 405 86 404 86 403 85 402 85 401 84 400 84 399 83 398 83 397 82 396 82 395 82 394 81 393 81 392 81 391 81 390 80 389 80 388 80 387 79 386 79 385 79 384 78 383 78 382 78 381 77 380 77 379 76 378 76 377 76 376 75 375 75 374 74 373 74 372 73 371 73 370 72 369 72 368 71 367 71 366 71 365 71 364 70 363 70 362 70 361 70 360 70 359 69 358 69 357 69 356 69 355 69 354 69 353 69 352 69 351 69 350 69 349 69 348 69 347 69 346 69 345 69 344 68 343 68 342 68 341 68 340 67 339 67 338 67 337 66 336 66 335 66 334 65 333 65 332 65 331 65 330 64 329 64 328 64 327 64 326 64 325 64 324 64 323 64 322 64 321 64 320 64 319 64 318 64 317 64 316 64 315 64 314 64 313 63 312 63 311 63 310 63 309 63 308 63 307 63 306 63 305 63 304 63 303 62 302 62 301 62 300 62 299 62 298 62 297 62 296 62 295 62 294 62 293 62 292 62 291 62 290 62 289 62 288 62 287 62 286 63 285 63 284 63 283 63 282 63 281 63 280 63 279 63 278 64 277 64 276 64 275 63 274 63 273 63 272 63 271 63 270 63 269 63 268 63 267 62 266 62 265 62 264 62 263 62 262 62 261 62 260 62 259 62 258 62 257 62 256 62 255 62 254 62 253 62 252 62 251 62 250 62 249 62 248 62 247 62 246 62 245 62 244 62 243 62 242 62 241 62 240 62 239 62 238 62 237 62 236 62 235 62 234 62 233 62 232 62 231 63 230 63 229 63 228 63 227 62 226 62 225 62 224 62 223 62 222 62 221 62 220 62']

    generate_eit_dataset(sample_list_crd)