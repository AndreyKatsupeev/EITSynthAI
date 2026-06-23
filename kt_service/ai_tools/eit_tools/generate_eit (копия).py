"""
Генерация датасета ЭИТ с использованием gmsh.
Переписанная версия с корректной обработкой всех классов,
MultiPolygon-геометрии и дыхательной модуляции только в лёгких.
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

    drive_pattern: str = "adjacent"

    output_dir: str = "/app/generation_results"
    dataset_name: str = "thorax_breath"

    debug_plot_each_class: bool = True
    debug_plot_mesh: bool = True


# ============================================================
# 2. ПАРСИНГ И ГЕОМЕТРИЯ
# ============================================================
def parse_list_crd(list_crd: List[str], pixel_spacing: float) -> Dict[int, List[np.ndarray]]:
    """Парсит полигоны из формата list_crd."""
    tissues: Dict[int, List[np.ndarray]] = {}

    for item in list_crd:
        parts = item.strip().split()
        if len(parts) < 7:
            continue

        cls_id = int(parts[0])
        coords = np.array([float(x) for x in parts[1:]], dtype=float).reshape(-1, 2)
        coords_mm = coords * pixel_spacing
        tissues.setdefault(cls_id, []).append(coords_mm)

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
        return None
    except Exception:
        return None



def _collect_polygon_parts(geom) -> List[Polygon]:
    parts: List[Polygon] = []
    if geom is None or geom.is_empty:
        return parts

    if isinstance(geom, Polygon):
        if geom.area > 0:
            parts.append(geom)
    elif isinstance(geom, MultiPolygon):
        for g in geom.geoms:
            if g.area > 0:
                parts.append(g)
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
    if merged.is_empty:
        return None
    if isinstance(merged, (Polygon, MultiPolygon)):
        return merged

    parts2 = _collect_polygon_parts(merged)
    if not parts2:
        return None
    merged2 = unary_union(parts2).buffer(0)
    if isinstance(merged2, (Polygon, MultiPolygon)) and not merged2.is_empty:
        return merged2
    return None



def build_class_geometry(polygons_data: Dict[int, List[np.ndarray]]) -> Dict[int, GeometryLike]:
    """Создает полную геометрию класса, не теряя MultiPolygon-компоненты."""
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



def add_polygon_to_gmsh(poly: Polygon, char_length: float, max_points: int = 400) -> int:
    """Добавляет внешний контур полигона в GMSH."""
    ext = np.array(poly.exterior.coords[:-1], dtype=float)

    if len(ext) > max_points:
        ls = LineString(np.vstack([ext, ext[0]]))
        xs = np.linspace(0, ls.length, max_points, endpoint=False)
        ext = np.array([ls.interpolate(x).coords[0] for x in xs], dtype=float)

    point_tags = [gmsh.model.geo.addPoint(float(x), float(y), 0.0, char_length) for x, y in ext]

    line_tags = []
    n = len(point_tags)
    for i in range(n):
        p1 = point_tags[i]
        p2 = point_tags[(i + 1) % n]
        line_tags.append(gmsh.model.geo.addLine(p1, p2))

    return gmsh.model.geo.addCurveLoop(line_tags)


# ============================================================
# 3. MESH И КЛАССИФИКАЦИЯ
# ============================================================
def build_mesh_gmsh(
    body_poly: Polygon,
    inclusions: Dict[int, GeometryLike],
    char_length: float = 3.5,
    min_length_factor: float = 0.35,
    mesh_order: int = 1,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, int]]:
    """
    Строит mesh только по внешнему контуру тела.
    Внутренние классы размечаются после генерации сетки.
    """
    PHYS_BACKGROUND = 1
    PHYS_BONE = 2
    PHYS_MUSCLE = 3
    PHYS_LUNG = 4
    PHYS_FAT = 5

    DICOM_TO_PHYS = {
        0: PHYS_BONE,
        1: PHYS_MUSCLE,
        2: PHYS_LUNG,
        3: PHYS_FAT,
    }

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

        elem_types, elem_tags, elem_node_tags = gmsh.model.mesh.getElements(2)
        triangles = []
        for etype, enodes in zip(elem_types, elem_node_tags):
            if etype == 2:
                enodes_arr = np.array(enodes, dtype=int).reshape(-1, 3)
                tris = np.array([[tag_to_idx[int(n)] for n in row] for row in enodes_arr], dtype=int)
                triangles.append(tris)

        triangles = np.vstack(triangles) if triangles else np.zeros((0, 3), dtype=int)
        if len(triangles) == 0:
            raise RuntimeError("[gmsh] Нет элементов!")

        print("[gmsh] Классификация элементов...")
        centroids = points[triangles].mean(axis=1)
        triangle_polys = [Polygon(points[tri]).buffer(0) for tri in triangles]
        triangle_areas = np.array([tp.area if not tp.is_empty else 0.0 for tp in triangle_polys])

        element_physicals = np.full(len(triangles), PHYS_BACKGROUND, dtype=int)

        priority_order = [0, 2, 1, 3]   # bone > lung > muscle > fat
        prepared = {cls_id: prep(geom) for cls_id, geom in inclusions.items()}

        cls_names = {0: "кость", 1: "мышцы", 2: "лёгкие", 3: "жир"}

        for cls_id in priority_order:
            if cls_id not in inclusions:
                print(f"  Класс {cls_id}: нет в данных")
                continue

            geom = inclusions[cls_id]
            geom_prep = prepared[cls_id]
            phys_tag = DICOM_TO_PHYS[cls_id]
            assigned_centroid = 0
            assigned_overlap = 0

            for i, c in enumerate(centroids):
                if element_physicals[i] != PHYS_BACKGROUND:
                    continue
                if geom_prep.contains(Point(float(c[0]), float(c[1]))):
                    element_physicals[i] = phys_tag
                    assigned_centroid += 1

            boundary_candidates = np.where(element_physicals == PHYS_BACKGROUND)[0]
            for i in boundary_candidates:
                tri_area = triangle_areas[i]
                if tri_area <= 1e-12:
                    continue
                try:
                    inter_area = triangle_polys[i].intersection(geom).area
                    threshold = 0.15 if cls_id == 2 else 0.30
                    if inter_area / tri_area >= threshold:
                        element_physicals[i] = phys_tag
                        assigned_overlap += 1
                except Exception:
                    continue

            total_assigned = assigned_centroid + assigned_overlap
            print(
                f"  Класс {cls_id} ({cls_names.get(cls_id, '?')}): "
                f"{total_assigned} элементов "
                f"(центроид={assigned_centroid}, overlap={assigned_overlap})"
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
def assign_conductivity(
    elem_physicals: np.ndarray,
    cfg: EITConfig,
    lung_sigma: float,
    phys_mapping: Dict[str, int] = None,
) -> np.ndarray:
    if phys_mapping is None:
        phys_mapping = {
            "background": 1,
            "bone": 2,
            "muscle": 3,
            "lung": 4,
            "fat": 5,
        }

    tag_to_sigma = {
        phys_mapping["background"]: cfg.conductivity["background"],
        phys_mapping["bone"]: cfg.conductivity["bone"],
        phys_mapping["muscle"]: cfg.conductivity["muscle"],
        phys_mapping["lung"]: lung_sigma,
        phys_mapping["fat"]: cfg.conductivity["fat"],
    }

    sigma = np.array(
        [tag_to_sigma.get(int(tag), cfg.conductivity["background"]) for tag in elem_physicals],
        dtype=float,
    )
    return sigma



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
        area = 0.5 * abs(
            (x[1] - x[0]) * (y[2] - y[0]) - (x[2] - x[0]) * (y[1] - y[0])
        )
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

    k_global = sp.coo_matrix((vals, (rows, cols)), shape=(n_pts, n_pts)).tocsr()
    return k_global



def solve_forward(
    k_global: sp.csr_matrix,
    inj_nodes: Tuple[int, int],
    meas_nodes: np.ndarray,
    current: float = 5e-3,
    ground_node: int = 0,
) -> np.ndarray:
    n = k_global.shape[0]
    current_vec = np.zeros(n, dtype=float)
    current_vec[inj_nodes[0]] += current
    current_vec[inj_nodes[1]] -= current

    keep = np.arange(n) != ground_node
    k_red = k_global[keep][:, keep]
    i_red = current_vec[keep]

    v_red = spla.spsolve(k_red.tocsc(), i_red)
    v = np.zeros(n, dtype=float)
    v[keep] = v_red
    return v[meas_nodes]



def build_drive_pattern(n_elec: int, pattern: str = "adjacent"):
    frames = []
    for k in range(n_elec):
        if pattern == "adjacent":
            inj = (k, (k + 1) % n_elec)
        elif pattern == "opposite":
            inj = (k, (k + n_elec // 2) % n_elec)
        else:
            raise ValueError(f"Неизвестный drive pattern: {pattern}")

        meas_pairs = [
            ((k + 1 + m) % n_elec, (k + 2 + m) % n_elec)
            for m in range(n_elec - 2)
        ]
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



def _plot_debug(points, tris, elem_physicals, electrodes_xy, cfg, phys_mapping, lung_elems=None):
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    ax1, ax2 = axes

    tag_colors = {
        phys_mapping["background"]: "#CD853F",
        phys_mapping["bone"]: "#F5F5F5",
        phys_mapping["muscle"]: "#DC143C",
        phys_mapping["lung"]: "#00BFFF",
        phys_mapping["fat"]: "#FFD700",
    }
    tag_names = {
        phys_mapping["background"]: "Фон/тело",
        phys_mapping["bone"]: "Кость",
        phys_mapping["muscle"]: "Мышцы",
        phys_mapping["lung"]: "Лёгкие",
        phys_mapping["fat"]: "Жир",
    }

    patches = []
    colors = []
    for i, tri in enumerate(tris):
        patch = MplPolygon(points[tri], closed=True)
        patches.append(patch)
        colors.append(tag_colors.get(int(elem_physicals[i]), "#808080"))

    collection = PatchCollection(patches, facecolor=colors, edgecolor="none", alpha=0.95)
    ax1.add_collection(collection)
    ax1.plot(
        electrodes_xy[:, 0], electrodes_xy[:, 1],
        "go", markersize=10, markeredgecolor="black", markeredgewidth=1.5, zorder=10
    )
    for i, (x, y) in enumerate(electrodes_xy):
        ax1.text(x, y, str(i), color="white", fontsize=8, ha="center", va="center", fontweight="bold", zorder=11)

    ax1.set_aspect("equal")
    ax1.set_title("Ткани (разные цвета)", fontsize=14, fontweight="bold")
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(points[:, 0].min() - 20, points[:, 0].max() + 20)
    ax1.set_ylim(points[:, 1].min() - 20, points[:, 1].max() + 20)

    legend_elements = [
        Line2D([0], [0], marker="s", color="w", markerfacecolor=tag_colors[tag], markersize=10, label=tag_names[tag])
        for tag in sorted(tag_colors.keys())
    ]
    ax1.legend(handles=legend_elements, loc="upper right", fontsize=10)

    sigma = _conductivity_by_tag(elem_physicals, cfg, phys_mapping)
    trip = ax2.tripcolor(
        points[:, 0], points[:, 1], tris,
        facecolors=np.log10(np.clip(sigma, 1e-8, None)),
        cmap="viridis",
        shading="flat",
    )

    if lung_elems is not None and len(lung_elems) > 0:
        lung_points = points[tris[lung_elems]].mean(axis=1)
        ax2.scatter(lung_points[:, 0], lung_points[:, 1], c="red", s=5, alpha=0.35, label="Лёгкие", zorder=5)

    ax2.plot(
        electrodes_xy[:, 0], electrodes_xy[:, 1],
        "go", markersize=8, markeredgecolor="white", markeredgewidth=1.5, label="Электроды"
    )
    plt.colorbar(trip, ax=ax2, label="log10(σ)")
    ax2.set_aspect("equal")
    ax2.set_title("Проводимости (log scale)", fontsize=14, fontweight="bold")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    out_path = os.path.join(cfg.output_dir, f"{cfg.dataset_name}_mesh_preview.png")
    plt.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)

    print(f"\n[Preview] Распределение тканей:")
    unique, counts = np.unique(elem_physicals, return_counts=True)
    for tag, count in zip(unique.tolist(), counts.tolist()):
        print(f"  {tag_names.get(tag, f'tag_{tag}'):15s}: {count:5d} элементов ({count / len(tris) * 100:5.1f}%)")



def _plot_class_masks(points, tris, elem_physicals, electrodes_xy, cfg, phys_mapping):
    if not cfg.debug_plot_each_class:
        return

    class_info = [
        ("bone", "Кость"),
        ("muscle", "Мышцы"),
        ("lung", "Лёгкие"),
        ("fat", "Жир"),
    ]

    for key, title in class_info:
        tag = phys_mapping[key]
        idx = np.where(elem_physicals == tag)[0]

        fig, axes = plt.subplots(1, 2, figsize=(16, 7))
        ax1, ax2 = axes

        ax1.triplot(points[:, 0], points[:, 1], tris, color="lightgray", lw=0.2, alpha=0.6)
        ax1.plot(electrodes_xy[:, 0], electrodes_xy[:, 1], 'go', markersize=12, markeredgecolor='black', markeredgewidth=1.5)

        if len(idx) == 0:
            ax1.text(0.5, 0.5, f"НЕТ ЭЛЕМЕНТОВ\nкласса {title}", transform=ax1.transAxes,
                     ha="center", va="center", color="red", fontsize=20, fontweight="bold")
        else:
            pts = points[tris[idx]].mean(axis=1)
            ax1.scatter(pts[:, 0], pts[:, 1], c="red", s=10, alpha=0.6, label="Центроиды")

        ax1.set_aspect("equal")
        ax1.set_title(f"Только: {title}", fontsize=14, fontweight="bold")
        ax1.grid(True, alpha=0.3)
        ax1.legend(loc="upper left")

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
            cmap="viridis",
            shading="flat"
        )
        ax2.plot(electrodes_xy[:, 0], electrodes_xy[:, 1], 'go', markersize=10, label='Электроды')
        plt.colorbar(trip, ax=ax2, label='log10(σ)')
        ax2.set_aspect('equal')
        ax2.set_title(f'Проводимость: {title}', fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        ax2.legend(loc="upper right")

        plt.tight_layout()
        plt.savefig(os.path.join(cfg.output_dir, f"{cfg.dataset_name}_debug_{key}.png"), dpi=220, bbox_inches='tight')
        plt.close(fig)


# ============================================================
# 6. ГЛАВНАЯ ФУНКЦИЯ
# ============================================================
def generate_eit_dataset(list_crd: List[str], cfg: EITConfig = None):
    if cfg is None:
        cfg = EITConfig()

    os.makedirs(cfg.output_dir, exist_ok=True)

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

    cx, cy = body_poly.centroid.x, body_poly.centroid.y
    body_poly = translate(body_poly, xoff=-cx, yoff=-cy)
    inclusions = {k: translate(v, xoff=-cx, yoff=-cy) for k, v in inclusions.items()}

    points, tris, elem_physicals, phys_mapping = build_mesh_gmsh(
        body_poly=body_poly,
        inclusions=inclusions,
        char_length=cfg.mesh_characteristic_length,
        min_length_factor=cfg.mesh_min_length_factor,
        mesh_order=cfg.mesh_order,
    )

    electrodes_xy = place_electrodes(body_poly, cfg.n_electrodes)
    elec_nodes = find_nearest_nodes(points, electrodes_xy)

    np.savetxt(
        os.path.join(cfg.output_dir, f"{cfg.dataset_name}_electrodes.csv"),
        np.hstack([electrodes_xy, elec_nodes[:, None]]),
        delimiter=",",
        header="x_mm,y_mm,node_id",
        comments="",
    )

    drive = build_drive_pattern(cfg.n_electrodes, cfg.drive_pattern)

    fps = cfg.breath_fps
    dt = 1.0 / fps
    n_frames = int(cfg.breath_period_sec * cfg.breath_fps * cfg.breath_n_cycles)

    sigma_lung_ex = cfg.conductivity["lung_exhale"]
    sigma_lung_in = cfg.conductivity["lung_inhale"]

    t = np.arange(n_frames, dtype=float) * dt
    breath_phase = np.sin(2.0 * np.pi * t / cfg.breath_period_sec)
    lung_sigma_series = 0.5 * (sigma_lung_ex + sigma_lung_in) - 0.5 * (sigma_lung_ex - sigma_lung_in) * breath_phase

    sigma_base = assign_conductivity(
        elem_physicals=elem_physicals,
        cfg=cfg,
        lung_sigma=sigma_lung_ex,
        phys_mapping=phys_mapping,
    )

    lung_phys_tag = phys_mapping["lung"]
    lung_mask = elem_physicals == lung_phys_tag
    lung_elems = np.where(lung_mask)[0]

    print(f"\n[Main] Элементов лёгких: {len(lung_elems)} из {len(tris)} ({len(lung_elems) / len(tris) * 100:.1f}%)")

    n_meas = cfg.n_electrodes * (cfg.n_electrodes - 2)
    voltages = np.zeros((n_frames, n_meas), dtype=float)

    print(f"\n[Main] Генерация {n_frames} кадров...")
    for f in range(n_frames):
        sigma = sigma_base.copy()
        sigma[lung_mask] = lung_sigma_series[f]

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
    }

    with open(os.path.join(cfg.output_dir, f"{cfg.dataset_name}_meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    np.save(os.path.join(cfg.output_dir, f"{cfg.dataset_name}_voltages.npy"), voltages)
    np.save(os.path.join(cfg.output_dir, f"{cfg.dataset_name}_breath_phase.npy"), breath_phase)
    np.save(os.path.join(cfg.output_dir, f"{cfg.dataset_name}_lung_sigma_series.npy"), lung_sigma_series)
    np.savez(
        os.path.join(cfg.output_dir, f"{cfg.dataset_name}_mesh.npz"),
        points=points,
        tris=tris,
        elem_physicals=elem_physicals,
    )

    print(f"\n[Main] ✓ Датасет сохранён в {cfg.output_dir}")

    if cfg.debug_plot_mesh:
        _plot_debug(points, tris, elem_physicals, electrodes_xy, cfg, phys_mapping, lung_elems)
        _plot_class_masks(points, tris, elem_physicals, electrodes_xy, cfg, phys_mapping)

    return voltages, breath_phase

