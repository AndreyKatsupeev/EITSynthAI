"""
Генерация датасета ЭИТ с использованием gmsh.
"""

import json
import os
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Union

import gmsh
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from matplotlib.collections import PatchCollection
from matplotlib.lines import Line2D
from matplotlib.patches import Polygon as MplPolygon
from shapely.affinity import translate
from shapely.geometry import GeometryCollection, LineString, MultiPolygon, Point, Polygon
from shapely.ops import unary_union
from shapely.prepared import prep
import random

GeometryLike = Union[Polygon, MultiPolygon]


# ============================================================
# 1. КОНФИГУРАЦИЯ
# ============================================================
@dataclass
class EITConfig:
    conductivity: Dict[str, float] = field(default_factory=lambda: {
        "fat": 0.02,
        "bone": 0.01,
        "muscle": 0.35,
        "lung_exhale": 0.25,
        "lung_inhale": 0.08,
        "background": 0.20,
    })

    breath_period_sec: float = 4.0
    breath_fps: float = 10.0
    breath_n_cycles: int = 3

    mesh_characteristic_length: float = 3.5
    mesh_order: int = 1
    mesh_min_length_factor: float = 0.35

    n_electrodes: int = 16
    electrode_radius_mm: float = 5.0
    inject_current_A: float = 5e-3

    drive_pattern: str = "opposite"
    overlap_threshold_default: float = 0.30
    overlap_threshold_lung: float = 0.15

    output_dir: str = "/app/generation_results"
    dataset_name: str = "thorax_breath"

    debug_plot_each_class: bool = True
    debug_plot_mesh: bool = True
    save_ground_truth_series: bool = True


# ============================================================
# 2. ПАРСИНГ И ГЕОМЕТРИЯ
# ============================================================
def parse_list_crd(list_crd: List[str], pixel_spacing: float) -> Dict[int, List[np.ndarray]]:
    tissues: Dict[int, List[np.ndarray]] = {}
    for item in list_crd:
        parts = item.strip().split()
        if len(parts) < 7:
            continue
        cls_id = int(parts[0])
        coords = np.array([float(x) for x in parts[1:]], dtype=float).reshape(-1, 2)
        tissues.setdefault(cls_id, []).append(coords * pixel_spacing)
    return tissues



def _clean_polygon_from_coords(coords: np.ndarray) -> Union[Polygon, MultiPolygon, None]:
    try:
        pts = [(float(x), float(y)) for x, y in coords]
        if len(pts) < 3:
            return None
        if pts[0] != pts[-1]:
            pts.append(pts[0])
        poly = Polygon(pts).buffer(0)
        if poly.is_empty or poly.area <= 0:
            return None
        if isinstance(poly, (Polygon, MultiPolygon)):
            return poly
    except Exception:
        pass
    return None



def _collect_polygon_parts(geom) -> List[Polygon]:
    parts: List[Polygon] = []
    if geom is None or geom.is_empty:
        return parts
    if isinstance(geom, Polygon):
        if geom.area > 0:
            parts.append(geom)
    elif isinstance(geom, MultiPolygon):
        parts.extend([g for g in geom.geoms if g.area > 0])
    elif isinstance(geom, GeometryCollection):
        for g in geom.geoms:
            parts.extend(_collect_polygon_parts(g))
    return parts



def _normalize_geometry(geom) -> Union[GeometryLike, None]:
    if geom is None:
        return None
    geom = geom.buffer(0)
    if geom.is_empty:
        return None

    parts = _collect_polygon_parts(geom)
    if not parts:
        return None

    merged = unary_union(parts).buffer(0)
    if isinstance(merged, (Polygon, MultiPolygon)) and not merged.is_empty:
        return merged

    parts2 = _collect_polygon_parts(merged)
    if not parts2:
        return None

    merged2 = unary_union(parts2).buffer(0)
    if isinstance(merged2, (Polygon, MultiPolygon)) and not merged2.is_empty:
        return merged2
    return None



def build_class_geometry(polygons_data: Dict[int, List[np.ndarray]]) -> Dict[int, GeometryLike]:
    result: Dict[int, GeometryLike] = {}
    for class_id, coord_lists in polygons_data.items():
        polys = []
        for coords in coord_lists:
            poly = _clean_polygon_from_coords(coords)
            if poly is not None:
                polys.append(poly)
        if not polys:
            continue
        merged = _normalize_geometry(unary_union(polys))
        if merged is not None:
            result[class_id] = merged
    return result



def _largest_polygon(geom: GeometryLike) -> Polygon:
    if isinstance(geom, Polygon):
        return geom
    if isinstance(geom, MultiPolygon):
        return max(geom.geoms, key=lambda p: p.area)
    raise TypeError("Ожидался Polygon или MultiPolygon")



def add_polygon_to_gmsh(poly: Polygon, char_length: float, max_points: int = 500) -> int:
    ext = np.array(poly.exterior.coords[:-1], dtype=float)
    if len(ext) > max_points:
        ls = LineString(np.vstack([ext, ext[0]]))
        xs = np.linspace(0, ls.length, max_points, endpoint=False)
        ext = np.array([ls.interpolate(x).coords[0] for x in xs], dtype=float)

    point_tags = [gmsh.model.geo.addPoint(float(x), float(y), 0.0, char_length) for x, y in ext]
    line_tags = []
    for i in range(len(point_tags)):
        line_tags.append(gmsh.model.geo.addLine(point_tags[i], point_tags[(i + 1) % len(point_tags)]))
    return gmsh.model.geo.addCurveLoop(line_tags)


# ============================================================
# 3. MESH И КЛАССИФИКАЦИЯ
# ============================================================
def build_mesh_gmsh(
    body_poly: Polygon,
    inclusions: Dict[int, GeometryLike],
    char_length: float,
    min_length_factor: float,
    mesh_order: int,
    overlap_threshold_default: float,
    overlap_threshold_lung: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, int]]:
    PHYS_BACKGROUND = 1
    PHYS_BONE = 2
    PHYS_MUSCLE = 3
    PHYS_LUNG = 4
    PHYS_FAT = 5

    DICOM_TO_PHYS = {0: PHYS_BONE, 1: PHYS_MUSCLE, 2: PHYS_LUNG, 3: PHYS_FAT}

    gmsh.initialize()
    try:
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.option.setNumber("General.Verbosity", 0)
        gmsh.model.add("eit_model")

        print("[gmsh] Создание геометрии...")
        print(f"  Тело (класс 4): площадь {body_poly.area:.1f} мм²")

        outer_loop = add_polygon_to_gmsh(body_poly, char_length, max_points=500)
        gmsh.model.geo.addPlaneSurface([outer_loop])
        gmsh.model.geo.synchronize()

        print("[gmsh] Генерация mesh...")
        gmsh.option.setNumber("Mesh.CharacteristicLengthMax", char_length)
        gmsh.option.setNumber("Mesh.CharacteristicLengthMin", max(char_length * min_length_factor, 0.5))
        gmsh.option.setNumber("Mesh.Algorithm", 6)
        gmsh.option.setNumber("Mesh.ElementOrder", mesh_order)
        gmsh.model.mesh.generate(2)

        node_tags, node_coords, _ = gmsh.model.mesh.getNodes()
        node_tags = np.array(node_tags, dtype=int)
        node_coords = np.array(node_coords, dtype=float).reshape(-1, 3)
        tag_to_idx = {int(t): i for i, t in enumerate(node_tags)}
        points = node_coords[:, :2]

        elem_types, _, elem_node_tags = gmsh.model.mesh.getElements(2)
        triangles = []
        for etype, enodes in zip(elem_types, elem_node_tags):
            if etype == 2:
                arr = np.array(enodes, dtype=int).reshape(-1, 3)
                triangles.append(np.array([[tag_to_idx[int(n)] for n in row] for row in arr], dtype=int))
        triangles = np.vstack(triangles) if triangles else np.zeros((0, 3), dtype=int)
        if len(triangles) == 0:
            raise RuntimeError("[gmsh] Нет элементов!")

        print("[gmsh] Классификация элементов...")
        triangle_polys = [Polygon(points[tri]).buffer(0) for tri in triangles]
        triangle_areas = np.array([tp.area if not tp.is_empty else 0.0 for tp in triangle_polys])
        centroids = points[triangles].mean(axis=1)

        element_physicals = np.full(len(triangles), PHYS_BACKGROUND, dtype=int)

        # Более естественный приоритет для твоей анатомии:
        # кость > мышцы > лёгкие > жир
        priority_order = [0, 1, 2, 3]
        prepared = {cls_id: prep(geom) for cls_id, geom in inclusions.items()}
        cls_names = {0: "кость", 1: "мышцы", 2: "лёгкие", 3: "жир"}

        for cls_id in priority_order:
            if cls_id not in inclusions:
                print(f"  Класс {cls_id}: нет в данных")
                continue

            geom = inclusions[cls_id]
            geom_prep = prepared[cls_id]
            phys_tag = DICOM_TO_PHYS[cls_id]
            threshold = overlap_threshold_lung if cls_id == 2 else overlap_threshold_default

            assigned_centroid = 0
            assigned_overlap = 0

            for i, c in enumerate(centroids):
                if element_physicals[i] != PHYS_BACKGROUND:
                    continue
                if geom_prep.contains(Point(float(c[0]), float(c[1]))):
                    element_physicals[i] = phys_tag
                    assigned_centroid += 1

            remaining = np.where(element_physicals == PHYS_BACKGROUND)[0]
            for i in remaining:
                tri_area = triangle_areas[i]
                if tri_area <= 1e-12:
                    continue
                try:
                    inter_area = triangle_polys[i].intersection(geom).area
                    if inter_area / tri_area >= threshold:
                        element_physicals[i] = phys_tag
                        assigned_overlap += 1
                except Exception:
                    continue

            print(
                f"  Класс {cls_id} ({cls_names.get(cls_id, '?')}): {assigned_centroid + assigned_overlap} элементов "
                f"(центроид={assigned_centroid}, overlap={assigned_overlap}, thr={threshold:.2f})"
            )

        phys_mapping = {
            "background": PHYS_BACKGROUND,
            "bone": PHYS_BONE,
            "muscle": PHYS_MUSCLE,
            "lung": PHYS_LUNG,
            "fat": PHYS_FAT,
        }

        print(f"\n[gmsh] Итого: {len(points)} узлов, {len(triangles)} элементов")
        unique, counts = np.unique(element_physicals, return_counts=True)
        tag_names = {v: k for k, v in phys_mapping.items()}
        for tag, count in zip(unique.tolist(), counts.tolist()):
            print(f"  {tag_names.get(tag, str(tag)):15s}: {count:5d} элементов ({count / len(triangles) * 100:5.1f}%)")

        return points, triangles, element_physicals, phys_mapping

    finally:
        gmsh.finalize()


# ============================================================
# 4. FEM
# ============================================================
def assign_conductivity(elem_physicals: np.ndarray, cfg: EITConfig, lung_sigma: float, phys_mapping: Dict[str, int]) -> np.ndarray:
    tag_to_sigma = {
        phys_mapping["background"]: cfg.conductivity["background"],
        phys_mapping["bone"]: cfg.conductivity["bone"],
        phys_mapping["muscle"]: cfg.conductivity["muscle"],
        phys_mapping["lung"]: lung_sigma,
        phys_mapping["fat"]: cfg.conductivity["fat"],
    }
    return np.array([tag_to_sigma.get(int(tag), cfg.conductivity["background"]) for tag in elem_physicals], dtype=float)

# =============================================================================
# ГЕНЕТИЧЕСКИЙ ОПТИМИЗАТОР (Встраивается в generate_eit.py)
# =============================================================================

# Параметры ГА (захардкожены по запросу)
GA_GENERATIONS = 200
GA_POPULATION_SIZE = 100
GA_MUTATION_RATE = 0.01
GA_ELITE_COUNT = 20
ELECTRODE_SIZE_MM = 15.0  # Диаметр электрода

class EITElectrodeOptimizer:
    """
    Класс для оптимизации размещения электродов с использованием генетического алгоритма.
    Адаптирован для работы с Shapely Polygon и dict inclusions.
    """
    def __init__(self, body_poly, inclusions: dict, n_electrodes: int, electrode_size_mm: float):
        self.body_poly = body_poly
        self.inclusions = inclusions
        
        # Контур тела в виде numpy массива
        ext_coords = np.array(body_poly.exterior.coords, dtype=float)
        self.contour_coords = ext_coords[:-1] # Убираем дублирующую последнюю точку
        self.contour_length = body_poly.length
        
        self.n_electrodes = n_electrodes
        self.electrode_size = electrode_size_mm
        
        # Извлечение центров тканей для фитнес-функции
        # Класс 2 - лёгкие (target)
        self.lung_centers = []
        if 2 in self.inclusions:
            geom = self.inclusions[2]
            if isinstance(geom, Polygon):
                self.lung_centers.append(np.array([geom.centroid.x, geom.centroid.y]))
            elif isinstance(geom, MultiPolygon):
                for g in geom.geoms:
                    self.lung_centers.append(np.array([g.centroid.x, g.centroid.y]))
        
        # Класс 0 - кости (penalty)
        self.bone_geoms = []
        if 0 in self.inclusions:
            geom = self.inclusions[0]
            if isinstance(geom, (Polygon, MultiPolygon)):
                if isinstance(geom, Polygon): self.bone_geoms.append(geom)
                else: self.bone_geoms.extend(list(geom.geoms))
                
        print(f"[INFO] Оптимизатор: Контур {self.contour_length:.1f}мм, Лёгкие: {len(self.lung_centers)}, Кости: {len(self.bone_geoms)}")

    def _get_point_on_contour(self, t: float) -> np.ndarray:
        """Возвращает координаты точки на контуре по параметру t [0.0, 1.0]"""
        t = t % 1.0
        target_dist = t * self.contour_length
        # Вычисляем длины сегментов контура
        diffs = np.diff(self.contour_coords, axis=0)
        seg_lens = np.linalg.norm(diffs, axis=1)
        cum = np.concatenate([[0.0], np.cumsum(seg_lens)])
        
        # Поиск индекса сегмента
        idx = np.searchsorted(cum, target_dist, side="right") - 1
        idx = min(max(idx, 0), len(seg_lens) - 1)
        
        # Интерполяция
        t_local = (target_dist - cum[idx]) / (seg_lens[idx] + 1e-12)
        return self.contour_coords[idx] * (1.0 - t_local) + self.contour_coords[idx + 1] * t_local

    def _check_overlap(self, positions: List[float]) -> bool:
        """Проверка наложения электродов"""
        if len(positions) < 2: return False
        
        # Преобразуем t в расстояния
        dists_mm = sorted([p * self.contour_length for p in positions])
        n = len(dists_mm)
        
        for i in range(n):
            d1 = dists_mm[i]
            d2 = dists_mm[(i + 1) % n]
            
            if i == n - 1:
                # Расстояние через разрыв контура
                distance = self.contour_length - dists_mm[-1] + dists_mm[0]
            else:
                distance = d2 - d1
            
            if distance < self.electrode_size:
                return True
        return False

    def _dist_point_to_line(self, p, a, b):
        """Расстояние от точки p до отрезка ab"""
        ap = p - a
        ab = b - a
        ab_sq = np.dot(ab, ab)
        if ab_sq == 0: return np.linalg.norm(p - a)
        proj = np.dot(ap, ab) / ab_sq
        proj = np.clip(proj, 0.0, 1.0)
        closest = a + proj * ab
        return np.linalg.norm(p - closest)

    def _calculate_fitness(self, positions: List[float]) -> float:
        """Функция пригодности (Fitness)"""
        if not self.lung_centers:
            # Если нет легких, стремимся просто к равномерности
            return self._calc_uniformity(positions)
            
        electrodes = [self._get_point_on_contour(t) for t in positions]
        lung_center = np.mean(self.lung_centers, axis=0) # Центр масс всех легких
        
        total_score = 0.0
        n_pairs = 0
        
        # Оценка пар электродов
        for i in range(len(electrodes)):
            for j in range(i + 1, len(electrodes)):
                e1, e2 = electrodes[i], electrodes[j]
                mid = (e1 + e2) / 2.0
                
                # 1. Притяжение к легким (чем ближе середина пары к легкому, тем лучше)
                dist_lung = np.linalg.norm(mid - lung_center)
                proximity = 1.0 / (1.0 + dist_lung / 100.0)
                
                # 2. Штраф за кости (если линия между электродами проходит близко к кости)
                bone_penalty = 0.0
                for bone in self.bone_geoms:
                    # Проверяем расстояние до центра кости для простоты, 
                    # либо до границы, если полигон кости маленький.
                    # В оригинале было dist_to_line для центра кости.
                    dist_bone = self._dist_point_to_line(np.array([bone.centroid.x, bone.centroid.y]), e1, e2)
                    if dist_bone < 20.0:
                        bone_penalty += 0.5 * (1.0 - dist_bone / 20.0)
                
                # 3. Длина пути (оптимально ~1/4 периметра)
                path_len = np.linalg.norm(e1 - e2)
                optimal_len = self.contour_length / 4.0
                len_score = np.exp(-((path_len - optimal_len)**2) / (2 * 50**2))
                
                pair_score = proximity * (1.0 - min(bone_penalty, 1.0)) * len_score
                total_score += pair_score
                n_pairs += 1
                
        if n_pairs > 0: total_score /= n_pairs
        
        # Регуляризация на равномерность (вес 30%)
        uniformity = self._calc_uniformity(positions)
        return 0.7 * total_score + 0.3 * uniformity

    def _calc_uniformity(self, positions: List[float]) -> float:
        distances = []
        n = len(positions)
        for i in range(n):
            t1, t2 = positions[i], positions[(i + 1) % n]
            dist = (t2 - t1) if t2 >= t1 else (1.0 - t1 + t2)
            distances.append(dist)
        
        ideal = 1.0 / n
        score = 1.0 - np.std(distances) / ideal
        return max(0.0, min(1.0, score))

    def optimize(self, verbose=True) -> List[np.ndarray]:
        print(f"[GA] Запуск оптимизации ({GA_GENERATIONS} поколений)...")
        
        # Инициализация популяции (параметры t: 0.0..1.0)
        population = []
        for _ in range(GA_POPULATION_SIZE):
            for _ in range(100): # Попытки создать валидного индивида
                pos = sorted([random.random() for _ in range(self.n_electrodes)])
                if not self._check_overlap(pos):
                    population.append(pos)
                    break
            else:
                population.append(pos) # Fallback

        best_fitness_ever = -1.0
        best_pos_ever = []
        
        for gen in range(GA_GENERATIONS):
            # Оценка
            fitnesses = [self._calculate_fitness(ind) for ind in population]
            max_fit = max(fitnesses)
            best_idx = fitnesses.index(max_fit)
            
            if max_fit > best_fitness_ever:
                best_fitness_ever = max_fit
                best_pos_ever = population[best_idx].copy()
            
            if verbose and gen % 50 == 0:
                print(f"  Gen {gen}: Best Fit {best_fitness_ever:.4f}")
            
            # Отбор (Элиты)
            sorted_idx = np.argsort(fitnesses)[::-1]
            new_pop = [population[i].copy() for i in sorted_idx[:GA_ELITE_COUNT]]
            
            # Мутация и скрещивание
            while len(new_pop) < GA_POPULATION_SIZE:
                parent = population[random.randint(0, min(19, len(population)-1))].copy() # Берем из топ-20
                child = parent.copy()
                
                # Мутация одного гена
                idx_mut = random.randint(0, self.n_electrodes - 1)
                child[idx_mut] += random.gauss(0, GA_MUTATION_RATE)
                child[idx_mut] = child[idx_mut] % 1.0
                child = sorted(child)
                
                if not self._check_overlap(child):
                    new_pop.append(child)
                else:
                    new_pop.append(parent)
            
            population = new_pop

        # Возврат координат
        final_coords = [self._get_point_on_contour(t) for t in best_pos_ever]
        print(f"[GA] Оптимизация завершена. Fitness: {best_fitness_ever:.4f}")
        return np.array(final_coords)


# =============================================================================
# ФУНКЦИИ РАССТАНОВКИ
# =============================================================================

def place_electrodes_uniform(body_poly: Polygon, n: int) -> np.ndarray:
    """Старый метод равномерного размещения"""
    ext = np.array(body_poly.exterior.coords, dtype=float)
    diffs = np.diff(ext, axis=0)
    seg_len = np.linalg.norm(diffs, axis=1)
    cum = np.concatenate([[0.0], np.cumsum(seg_len)])
    total_len = cum[-1]
    positions = np.linspace(0.0, total_len, n, endpoint=False)

    electrodes = []
    for s in positions:
        idx = np.searchsorted(cum, s, side="right") - 1
        idx = min(max(idx, 0), len(seg_len) - 1)
        t = (s - cum[idx]) / (seg_len[idx] + 1e-12)
        pt = ext[idx] * (1.0 - t) + ext[idx + 1] * t
        electrodes.append(pt)
    return np.array(electrodes, dtype=float)


def find_nearest_nodes(points: np.ndarray, electrodes: np.ndarray) -> np.ndarray:
    from scipy.spatial import cKDTree
    tree = cKDTree(points)
    _, idx = tree.query(electrodes)
    return np.asarray(idx, dtype=int)

def place_electrodes(body_poly: Polygon, n: int) -> np.ndarray:
    ext = np.array(body_poly.exterior.coords, dtype=float)
    diffs = np.diff(ext, axis=0)
    seg_len = np.linalg.norm(diffs, axis=1)
    cum = np.concatenate([[0.0], np.cumsum(seg_len)])
    total_len = cum[-1]
    positions = np.linspace(0.0, total_len, n, endpoint=False)

    electrodes = []
    for s in positions:
        idx = np.searchsorted(cum, s, side="right") - 1
        idx = min(max(idx, 0), len(seg_len) - 1)
        t = (s - cum[idx]) / (seg_len[idx] + 1e-12)
        pt = ext[idx] * (1.0 - t) + ext[idx + 1] * t
        electrodes.append(pt)
    return np.array(electrodes, dtype=float)



# def place_electrodes(
#     body_poly: Polygon, 
#     n: int, 
#     min_dist: float = 5, 
#     max_attempts: int = 10000
# ) -> np.ndarray:
#     """
#     Размещает n электродов случайно вдоль периметра полигона.
    
#     :param body_poly: Shapely Polygon, вдоль границы которого размещаются электроды.
#     :param n: Количество электродов.
#     :param min_dist: Минимальное евклидово расстояние между любыми двумя электродами.
#     :param max_attempts: Максимальное количество попыток генерации (защита от бесконечного цикла).
#     :return: np.ndarray формы (n, 2) с координатами (x, y).
#     """
#     if n == 0:
#         return np.empty((0, 2), dtype=float)

#     ext = np.array(body_poly.exterior.coords, dtype=float)
#     diffs = np.diff(ext, axis=0)
#     seg_len = np.linalg.norm(diffs, axis=1)
#     cum = np.concatenate([[0.0], np.cumsum(seg_len)])
#     total_len = cum[-1]

#     # Вспомогательная функция для перевода 1D расстояния вдоль периметра в 2D точку
#     def get_point_at_dist(s: float) -> np.ndarray:
#         idx = np.searchsorted(cum, s, side="right") - 1
#         idx = min(max(idx, 0), len(seg_len) - 1)
#         t = (s - cum[idx]) / (seg_len[idx] + 1e-12)
#         return ext[idx] * (1.0 - t) + ext[idx + 1] * t

#     electrodes = []
#     attempts = 0

#     while len(electrodes) < n:
#         attempts += 1
#         if attempts > max_attempts:
#             raise ValueError(
#                 f"Не удалось разместить {n} электродов с min_dist={min_dist:.4f}. "
#                 f"Удалось разместить только {len(electrodes)}. "
#                 f"Попробуйте уменьшить n, увеличить min_dist или max_attempts."
#             )

#         # 1. Генерируем случайную позицию вдоль периметра
#         s = np.random.uniform(0.0, total_len)
#         pt = get_point_at_dist(s)

#         # 2. Если это первый электрод, просто добавляем его
#         if not electrodes:
#             electrodes.append(pt)
#             continue

#         # 3. Проверяем евклидово расстояние до всех уже размещенных электродов
#         pts_array = np.array(electrodes)
#         dists = np.linalg.norm(pts_array - pt, axis=1)

#         # Если точка достаточно далеко от всех остальных, принимаем её
#         if np.all(dists >= min_dist):
#             electrodes.append(pt)

#     return np.array(electrodes, dtype=float)

def find_nearest_nodes(points: np.ndarray, electrodes: np.ndarray) -> np.ndarray:
    from scipy.spatial import cKDTree
    tree = cKDTree(points)
    _, idx = tree.query(electrodes)
    return np.asarray(idx, dtype=int)



def assemble_stiffness(points: np.ndarray, tris: np.ndarray, sigma: np.ndarray) -> sp.csr_matrix:
    n_pts = len(points)
    rows, cols, vals = [], [], []

    for e, idx in enumerate(tris):
        xy = points[idx]
        x = xy[:, 0]
        y = xy[:, 1]
        area = 0.5 * abs((x[1] - x[0]) * (y[2] - y[0]) - (x[2] - x[0]) * (y[1] - y[0]))
        if area < 1e-12:
            continue

        b = np.array([y[1] - y[2], y[2] - y[0], y[0] - y[1]], dtype=float)
        c = np.array([x[2] - x[1], x[0] - x[2], x[1] - x[0]], dtype=float)
        ke = sigma[e] * (np.outer(b, b) + np.outer(c, c)) / (4.0 * area)

        for i_loc in range(3):
            for j_loc in range(3):
                rows.append(int(idx[i_loc]))
                cols.append(int(idx[j_loc]))
                vals.append(float(ke[i_loc, j_loc]))

    return sp.coo_matrix((vals, (rows, cols)), shape=(n_pts, n_pts)).tocsr()



def solve_forward(k_global: sp.csr_matrix, inj_nodes: Tuple[int, int], meas_nodes: np.ndarray, current: float = 5e-3) -> np.ndarray:
    """
    Решение прямой задачи с фиксацией среднего потенциала через один удалённый узел.
    Это не complete electrode model, но даёт более стабильную симуляцию.
    """
    n = k_global.shape[0]
    current_vec = np.zeros(n, dtype=float)
    current_vec[inj_nodes[0]] += current
    current_vec[inj_nodes[1]] -= current

    # Берем ground как узел, максимально удалённый от активной пары
    ground_node = int(np.setdiff1d(np.arange(n), np.array(inj_nodes, dtype=int))[0])

    keep = np.arange(n) != ground_node
    k_red = k_global[keep][:, keep]
    i_red = current_vec[keep]
    v_red = spla.spsolve(k_red.tocsc(), i_red)

    v = np.zeros(n, dtype=float)
    v[keep] = v_red
    v -= np.mean(v[meas_nodes])
    return v[meas_nodes]



def build_drive_pattern(n_elec: int, pattern: str = "opposite"):
    frames = []
    for k in range(n_elec):
        if pattern == "adjacent":
            inj = (k, (k + 1) % n_elec)
        elif pattern == "opposite":
            inj = (k, (k + n_elec // 2) % n_elec)
        elif pattern == "skip1":
            inj = (k, (k + 2) % n_elec)
        else:
            raise ValueError(f"Неизвестный drive pattern: {pattern}")

        meas_pairs = [((k + 1 + m) % n_elec, (k + 2 + m) % n_elec) for m in range(n_elec - 2)]
        frames.append((inj, meas_pairs))
    return frames


# ============================================================
# 5. ВИЗУАЛИЗАЦИЯ
# ============================================================
def _conductivity_by_tag(elem_physicals: np.ndarray, cfg: EITConfig, phys_mapping: Dict[str, int]) -> np.ndarray:
    sigma = np.zeros(len(elem_physicals), dtype=float)
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
    return sigma



def _plot_debug(points, tris, elem_physicals, electrodes_final, cfg, phys_mapping,
                electrodes_standard=None, electrodes_optimized=None, lung_elems=None):
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    ax1, ax2 = axes

    tag_colors = {
        phys_mapping["background"]: "#CD853F",
        phys_mapping["bone"]:       "#F5F5F5",
        phys_mapping["muscle"]:     "#DC143C",
        phys_mapping["lung"]:       "#00BFFF",
        phys_mapping["fat"]:        "#FFD700",
    }
    tag_names = {
        phys_mapping["background"]: "Фон/тело",
        phys_mapping["bone"]:       "Кость",
        phys_mapping["muscle"]:     "Мышцы",
        phys_mapping["lung"]:       "Лёгкие",
        phys_mapping["fat"]:        "Жир",
    }

    # Две независимые коллекции патчей
    patches, colors = [], []
    for i, tri in enumerate(tris):
        patches.append(MplPolygon(points[tri], closed=True))
        colors.append(tag_colors.get(int(elem_physicals[i]), "#808080"))

    ax1.add_collection(PatchCollection(patches, facecolor=colors, edgecolor="none", alpha=0.95))
    ax2.add_collection(PatchCollection(patches, facecolor=colors, edgecolor="none", alpha=0.95))

    # --- Электроды ---
    show_both = (electrodes_standard is not None and electrodes_optimized is not None)

    if show_both:
        # Исходное (равномерное) — красное
        ax1.plot(electrodes_standard[:, 0], electrodes_standard[:, 1], "ro",
                 markersize=10, markeredgecolor="black", markeredgewidth=1.5,
                 label="Исходное положение", zorder=10)
        ax2.plot(electrodes_standard[:, 0], electrodes_standard[:, 1], "ro",
                 markersize=8, markeredgecolor="white", markeredgewidth=1.5,
                 label="Исходное положение", zorder=10)

        # Оптимальное — зелёное
        ax1.plot(electrodes_optimized[:, 0], electrodes_optimized[:, 1], "go",
                 markersize=10, markeredgecolor="black", markeredgewidth=1.5,
                 label="Оптимальное положение", zorder=11)
        ax2.plot(electrodes_optimized[:, 0], electrodes_optimized[:, 1], "go",
                 markersize=8, markeredgecolor="white", markeredgewidth=1.5,
                 label="Оптимальное положение", zorder=11)
    else:
        # Только один набор (равномерный режим)
        ax1.plot(electrodes_final[:, 0], electrodes_final[:, 1], "go",
                 markersize=10, markeredgecolor="black", markeredgewidth=1.5,
                 label="Электроды", zorder=10)
        ax2.plot(electrodes_final[:, 0], electrodes_final[:, 1], "go",
                 markersize=8, markeredgecolor="white", markeredgewidth=1.5,
                 label="Электроды", zorder=10)

    # Настройки левого графика
    ax1.set_aspect("equal")
    ax1.set_title("Ткани (разные цвета)", fontsize=14, fontweight="bold")
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(points[:, 0].min() - 20, points[:, 0].max() + 20)
    ax1.set_ylim(points[:, 1].min() - 20, points[:, 1].max() + 20)

    # Легенда тканей
    tissue_legend = [
        Line2D([0], [0], marker="s", color="w",
               markerfacecolor=tag_colors[tag], markersize=10,
               label=tag_names[tag])
        for tag in sorted(tag_colors.keys())
    ]
    ax1.legend(handles=tissue_legend, loc="upper right", fontsize=9)

    # Правый график — проводимости
    sigma = _conductivity_by_tag(elem_physicals, cfg, phys_mapping)
    trip = ax2.tripcolor(
        points[:, 0], points[:, 1], tris,
        facecolors=np.log10(np.clip(sigma, 1e-8, None)),
        cmap="viridis", shading="flat",
    )
    if lung_elems is not None and len(lung_elems) > 0:
        lung_points = points[tris[lung_elems]].mean(axis=1)
        ax2.scatter(lung_points[:, 0], lung_points[:, 1],
                    c="red", s=5, alpha=0.35, label="Лёгкие", zorder=5)

    plt.colorbar(trip, ax=ax2, label="log10(σ)")
    ax2.set_aspect("equal")
    ax2.set_title("Проводимости (log scale)", fontsize=14, fontweight="bold")
    ax2.legend(loc="upper right", fontsize=9)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(cfg.output_dir,
                             f"{cfg.dataset_name}_mesh_preview.png"),
                dpi=220, bbox_inches="tight")
    plt.close(fig)



def _plot_class_masks(points, tris, elem_physicals, electrodes_final, cfg, phys_mapping,
                      electrodes_standard=None, electrodes_optimized=None):
    if not cfg.debug_plot_each_class:
        return

    class_info = [
        ("bone",   "Кость"),
        ("muscle", "Мышцы"),
        ("lung",   "Лёгкие"),
        ("fat",    "Жир"),
    ]
    show_both = (electrodes_standard is not None and electrodes_optimized is not None)

    for key, title in class_info:
        tag = phys_mapping[key]
        idx = np.where(elem_physicals == tag)[0]

        fig, axes = plt.subplots(1, 2, figsize=(16, 7))
        ax1, ax2 = axes

        # --- Левый график: геометрия класса ---
        ax1.triplot(points[:, 0], points[:, 1], tris,
                    color="lightgray", lw=0.2, alpha=0.6)

        if show_both:
            ax1.plot(electrodes_standard[:, 0], electrodes_standard[:, 1], "ro",
                     markersize=12, markeredgecolor="black", markeredgewidth=1.5,
                     label="Исходное положение", zorder=10)
            ax1.plot(electrodes_optimized[:, 0], electrodes_optimized[:, 1], "go",
                     markersize=12, markeredgecolor="black", markeredgewidth=1.5,
                     label="Оптимальное положение", zorder=11)
        else:
            ax1.plot(electrodes_final[:, 0], electrodes_final[:, 1], "go",
                     markersize=12, markeredgecolor="black", markeredgewidth=1.5,
                     label="Электроды", zorder=10)

        if len(idx) == 0:
            ax1.text(0.5, 0.5,
                     f"НЕТ ЭЛЕМЕНТОВ\nкласса {title}",
                     transform=ax1.transAxes,
                     ha="center", va="center",
                     color="red", fontsize=20, fontweight="bold")
        else:
            pts = points[tris[idx]].mean(axis=1)
            ax1.scatter(pts[:, 0], pts[:, 1],
                        c="red", s=10, alpha=0.6, label="Центроиды")

        ax1.set_aspect("equal")
        ax1.set_title(f"Только: {title}", fontsize=14, fontweight="bold")
        ax1.grid(True, alpha=0.3)
        ax1.legend(loc="upper left", fontsize=9)

        # --- Правый график: проводимость ---
        sigma = np.full(len(tris), cfg.conductivity["background"], dtype=float)
        if key == "bone":
            sigma[idx] = cfg.conductivity["bone"]
        elif key == "muscle":
            sigma[idx] = cfg.conductivity["muscle"]
        elif key == "lung":
            sigma[idx] = cfg.conductivity["lung_exhale"]
        elif key == "fat":
            sigma[idx] = cfg.conductivity["fat"]

        trip = ax2.tripcolor(
            points[:, 0], points[:, 1], tris,
            facecolors=np.log10(np.clip(sigma, 1e-8, None)),
            cmap="viridis", shading="flat",
        )

        if show_both:
            ax2.plot(electrodes_standard[:, 0], electrodes_standard[:, 1], "ro",
                     markersize=10, markeredgecolor="white", markeredgewidth=1.5,
                     label="Исходное положение", zorder=10)
            ax2.plot(electrodes_optimized[:, 0], electrodes_optimized[:, 1], "go",
                     markersize=10, markeredgecolor="white", markeredgewidth=1.5,
                     label="Оптимальное положение", zorder=11)
        else:
            ax2.plot(electrodes_final[:, 0], electrodes_final[:, 1], "go",
                     markersize=10, markeredgecolor="white", markeredgewidth=1.5,
                     label="Электроды", zorder=10)

        plt.colorbar(trip, ax=ax2, label="log10(σ)")
        ax2.set_aspect("equal")
        ax2.set_title(f"Проводимость: {title}", fontsize=14, fontweight="bold")
        ax2.grid(True, alpha=0.3)
        ax2.legend(loc="upper right", fontsize=9)

        plt.tight_layout()
        plt.savefig(os.path.join(cfg.output_dir,
                                 f"{cfg.dataset_name}_debug_{key}.png"),
                    dpi=220, bbox_inches="tight")
        plt.close(fig)


# ============================================================
# 6. ГЛАВНАЯ ФУНЦИЯ
# ============================================================
def generate_eit_dataset(list_crd: List[str], cfg: EITConfig = None, use_optimization: bool = True):
    """
    Основная функция генерации датасета ЭИТ.

    :param list_crd: список строк с описанием тканей (первая строка — pixel_spacing).
    :param cfg: конфигурация генерации.
    :param use_optimization: если True — электроды оптимизируются ГА;
                             равномерное размещение всё равно считается и рисуется для сравнения,
                             но в FEM-расчётах участвуют только оптимизированные позиции.
                             Если False — используется только равномерное размещение.
    """
    if cfg is None:
        cfg = EITConfig()
    os.makedirs(cfg.output_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # 1. Парсинг и геометрия
    # ------------------------------------------------------------------
    pixel_spacing = float(list_crd[0])
    tissues = parse_list_crd(list_crd[2:], pixel_spacing)

    print(f"\n[Main] Найдено классов: {sorted(tissues.keys())}")
    for cls_id, polys in tissues.items():
        total_area = 0.0
        for p in polys:
            poly = _clean_polygon_from_coords(p)
            if poly is not None:
                total_area += poly.area
        print(f"  Класс {cls_id}: {len(polys)} полигонов, общая площадь {total_area:.1f} мм²")

    if 4 not in tissues:
        raise ValueError("Класс 4 (контур тела) не найден!")

    body_geoms = build_class_geometry({4: tissues[4]})
    if 4 not in body_geoms:
        raise ValueError("Не удалось создать корректный полигон тела")

    body_poly = _largest_polygon(body_geoms[4]).buffer(0)
    if body_poly.is_empty or body_poly.area <= 0:
        raise ValueError("Полигон тела пустой или некорректный")

    raw_geoms = build_class_geometry({k: v for k, v in tissues.items() if k in [0, 1, 2, 3]})
    inclusions: Dict[int, GeometryLike] = {}
    for cls_id, geom in raw_geoms.items():
        clipped = _normalize_geometry(geom.intersection(body_poly))
        if clipped is not None and clipped.area > 10.0:
            inclusions[cls_id] = clipped

    print("\n[Main] Геометрии включений после clip/intersection:")
    for cls_id, geom in inclusions.items():
        print(f"  class {cls_id}: area={geom.area:.1f}, type={geom.geom_type}")

    # Центрирование
    cx, cy = body_poly.centroid.x, body_poly.centroid.y
    body_poly = translate(body_poly, xoff=-cx, yoff=-cy)
    inclusions = {k: translate(v, xoff=-cx, yoff=-cy) for k, v in inclusions.items()}

    # ------------------------------------------------------------------
    # 2. Построение mesh
    # ------------------------------------------------------------------
    points, tris, elem_physicals, phys_mapping = build_mesh_gmsh(
        body_poly=body_poly,
        inclusions=inclusions,
        char_length=cfg.mesh_characteristic_length,
        min_length_factor=cfg.mesh_min_length_factor,
        mesh_order=cfg.mesh_order,
        overlap_threshold_default=cfg.overlap_threshold_default,
        overlap_threshold_lung=cfg.overlap_threshold_lung,
    )

    # ------------------------------------------------------------------
    # 3. Расстановка электродов
    # ------------------------------------------------------------------
    print("\n[Main] Расстановка электродов...")

    # Всегда считаем равномерное размещение (для визуализации и как fallback)
    electrodes_standard = place_electrodes(body_poly, cfg.n_electrodes)

    if use_optimization:
        print("[Main] Запуск оптимизации размещения электродов (ГА)...")
        optimizer = EITElectrodeOptimizer(
            body_poly=body_poly,
            inclusions=inclusions,
            n_electrodes=cfg.n_electrodes,
            electrode_size_mm=15.0,  # диаметр электрода, мм
        )
        electrodes_optimized = optimizer.optimize()
        electrodes_final = electrodes_optimized
        print("[Main] Электроды оптимизированы. В расчётах участвуют оптимизированные позиции.")
    else:
        electrodes_optimized = None
        electrodes_final = electrodes_standard
        print("[Main] Используется равномерное размещение.")

    # Поиск узлов mesh, ближайших к финальным электродам
    elec_nodes = find_nearest_nodes(points, electrodes_final)

    np.savetxt(
        os.path.join(cfg.output_dir, f"{cfg.dataset_name}_electrodes.csv"),
        np.hstack([electrodes_final, elec_nodes[:, None]]),
        delimiter=",",
        header="x_mm,y_mm,node_id",
        comments="",
    )

    # ------------------------------------------------------------------
    # 4. Подготовка прямой задачи
    # ------------------------------------------------------------------
    drive = build_drive_pattern(cfg.n_electrodes, cfg.drive_pattern)

    dt = 1.0 / cfg.breath_fps
    n_frames = int(cfg.breath_period_sec * cfg.breath_fps * cfg.breath_n_cycles)
    sigma_lung_ex = cfg.conductivity["lung_exhale"]
    sigma_lung_in = cfg.conductivity["lung_inhale"]

    t = np.arange(n_frames, dtype=float) * dt
    breath_phase = np.sin(2.0 * np.pi * t / cfg.breath_period_sec)
    lung_sigma_series = (
        0.5 * (sigma_lung_ex + sigma_lung_in)
        - 0.5 * (sigma_lung_ex - sigma_lung_in) * breath_phase
    )

    sigma_base = assign_conductivity(elem_physicals, cfg, sigma_lung_ex, phys_mapping)
    lung_mask = elem_physicals == phys_mapping["lung"]
    lung_elems = np.where(lung_mask)[0]

    print(f"\n[Main] Элементов лёгких: {len(lung_elems)} из {len(tris)} "
          f"({len(lung_elems) / len(tris) * 100:.1f}%)")

    n_meas = cfg.n_electrodes * (cfg.n_electrodes - 2)
    voltages = np.zeros((n_frames, n_meas), dtype=float)

    if cfg.save_ground_truth_series:
        sigma_series = np.zeros((n_frames, len(tris)), dtype=np.float32)
        delta_sigma_series = np.zeros((n_frames, len(tris)), dtype=np.float32)
    else:
        sigma_series = None
        delta_sigma_series = None

    # ------------------------------------------------------------------
    # 5. Цикл по кадрам (FEM)
    # ------------------------------------------------------------------
    print(f"\n[Main] Генерация {n_frames} кадров...")
    for f in range(n_frames):
        sigma = sigma_base.copy()
        sigma[lung_mask] = lung_sigma_series[f]

        if sigma_series is not None:
            sigma_series[f] = sigma.astype(np.float32)
            delta_sigma_series[f] = (sigma - sigma_base).astype(np.float32)

        k_global = assemble_stiffness(points, tris, sigma)

        row = 0
        for inj, meas_pairs in drive:
            inj_nodes = (int(elec_nodes[inj[0]]), int(elec_nodes[inj[1]]))
            v = solve_forward(k_global, inj_nodes, elec_nodes, current=cfg.inject_current_A)
            for m_plus, m_minus in meas_pairs:
                voltages[f, row] = v[m_plus] - v[m_minus]
                row += 1

        if (f + 1) % 10 == 0 or f == 0:
            print(f"  frame {f + 1}/{n_frames}, σ_lung={lung_sigma_series[f]:.4f} S/m")

    # ------------------------------------------------------------------
    # 6. Сохранение результатов
    # ------------------------------------------------------------------
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
        "n_lung_elements": int(len(lung_elems)),
        "lung_sigma_series_min": float(np.min(lung_sigma_series)),
        "lung_sigma_series_max": float(np.max(lung_sigma_series)),
        "use_optimization": bool(use_optimization),
    }

    with open(os.path.join(cfg.output_dir, f"{cfg.dataset_name}_meta.json"),
              "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    np.save(os.path.join(cfg.output_dir, f"{cfg.dataset_name}_voltages.npy"), voltages)
    np.save(os.path.join(cfg.output_dir, f"{cfg.dataset_name}_breath_phase.npy"), breath_phase)
    np.save(os.path.join(cfg.output_dir, f"{cfg.dataset_name}_lung_sigma_series.npy"), lung_sigma_series)
    np.savez(os.path.join(cfg.output_dir, f"{cfg.dataset_name}_mesh.npz"),
             points=points, tris=tris, elem_physicals=elem_physicals)

    if sigma_series is not None:
        np.save(os.path.join(cfg.output_dir, f"{cfg.dataset_name}_sigma_series.npy"), sigma_series)
        np.save(os.path.join(cfg.output_dir, f"{cfg.dataset_name}_delta_sigma_series.npy"), delta_sigma_series)

    print(f"\n[Main] ✓ Датасет сохранён в {cfg.output_dir}")

    # ------------------------------------------------------------------
    # 7. Визуализация
    # ------------------------------------------------------------------
    if cfg.debug_plot_mesh:
        _plot_debug(
            points, tris, elem_physicals, electrodes_final, cfg, phys_mapping,
            electrodes_standard=electrodes_standard,
            electrodes_optimized=electrodes_optimized,
            lung_elems=lung_elems,
        )
        _plot_class_masks(
            points, tris, elem_physicals, electrodes_final, cfg, phys_mapping,
            electrodes_standard=electrodes_standard,
            electrodes_optimized=electrodes_optimized,
        )

    return voltages, breath_phase
