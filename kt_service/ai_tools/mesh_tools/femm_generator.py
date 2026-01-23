# Finite element mesh generator for FEMM
import gmsh
import numpy as np
from shapely.geometry import Polygon, Point
import shapely.errors
import cv2
import multiprocessing
import time


def divide_triangles_into_groups(contours, outer_contour_class, outer_contour, skin_width):
    """
        Divides 2D triangular mesh elements into classes based on intersection with given contours.

        Parameters:
        -----------
        contours : list of lists
            A list of contours. Each contour is a list where:
            - The first element is the class ID (an integer).
            - The remaining elements represent the coordinates of the contour's points
              in the format [x1, y1, x2, y2, ..., xn, yn].

        outer_contour_class: integer
        number of class which includes all elements not divided into another segments

        outer_contour: list
        Contains the data of outer contour in the same format as contours parameter

        skin_width: float, optional (default=7)
        Width of skin to be added. If value is 0, no skin will be added
        If value is -1, innermost finite elements will be considered as skin
        Otherwise outer contour with specified width will be added

        Returns:
        --------
        class_groups : dict
            A dictionary mapping class IDs to lists of element tags (triangles) that belong to that class.
            For example: {0: [12, 15, 19], 1: [23, 25, 28]}

        Notes:
        ------
        - This function uses the Gmsh Python API and assumes a 2D triangular mesh is already generated.
        - The classification is performed by computing the intersection area between each triangle and each contour.
          The triangle is assigned to the class of the contour with which it has the maximum intersection area.
        - Contours with fewer than 4 points (less than 9 values including class ID) are discarded before processing.
        - For each class, a corresponding physical group is created in the Gmsh model.
        """
    # Filter out short contours with fewer than 4 points (i.e., less than 9 values)
    k = -1
    start_time = time.time()
    for i in range(len(contours)):
        k += 1
        if k <= len(contours) - 1 and len(contours[k]) < 9:
            contours.pop(k)
            k -= 1
    end_time = time.time()
    print(f"Noise removal time: {end_time - start_time:.6f} seconds")
    # Retrieve all 2D elements (triangles)
    elem_types, elem_tags, elem_node_tags = gmsh.model.mesh.getElements(2)
    # Get all mesh nodes and their coordinates
    node_tags, node_coords, _ = gmsh.model.mesh.getNodes()
    node_dict = {tag: (node_coords[i * 3], node_coords[i * 3 + 1]) for i, tag in enumerate(node_tags)}
    class_groups = {}
    # Loop over all elements
    for elem_type, tags, nodes in zip(elem_types, elem_tags, elem_node_tags):
        if elem_type == 2:  # Element type 2 corresponds to triangles
            task_args = [
                (i, nodes, node_dict, tags, contours, outer_contour_class, outer_contour, skin_width)
                for i in range(len(tags))
            ]
            # Parallel execution
            with multiprocessing.Pool() as pool:
                results = pool.starmap(process_triangle, task_args)
            for best_class, tag in results:
                if best_class not in class_groups:
                    class_groups[best_class] = []
                class_groups[best_class].append(tag)
        # Create physical groups in Gmsh for each class
        for class_id, elements in class_groups.items():
            group = gmsh.model.addPhysicalGroup(2, elements)
            gmsh.model.setPhysicalName(2, group, f"Class_{class_id}")
    return class_groups

def process_triangle(i, nodes, node_dict, tags, contours, outer_contour_class, outer_contour, skin_width):
    """
            Determines the most appropriate class for a triangle based on maximum area of intersection with given contours.

            For the triangle at index `i`, this function:
            - Extracts its vertex coordinates using the `nodes` and `node_dict`.
            - Constructs a triangle polygon.
            - Compares it with each contour polygon to find the one with the largest intersection area.
            - Assigns the triangle to the class of the best-matching contour.

            Parameters:
            -----------
            i (int): Index of the triangle (0-based).
            nodes (List[int]): Flat list of node indices, with 3 indices per triangle.
            node_dict (Dict[int, Tuple[float, float]]): Mapping from node index to (x, y) coordinate.
            tags (List[int]): List of triangle identifiers (e.g., element tags) indexed by triangle.
            contours (List[List[float]]): List of contours, where each contour starts with a class label
            followed by flat (x1, y1, x2, y2, ...) coordinates.
            outer_contour_class (Any): Default class to assign if no intersection is found.
            outer_contour: list
            Contains the data of outer contour in the same format as contours parameter
            skin_width: float, optional (default=7)
            Width of skin to be added. If value is 0, no skin will be added
            If value is -1, innermost finite elements will be considered as skin
            Otherwise outer contour with specified width will be added

            Returns:
            --------
            Tuple[Any, int]: A tuple containing:
            - The class label assigned to the triangle.
            - The original tag of the triangle from `tags[i]`.
            """
    # Get triangle vertex coordinates
    n1, n2, n3 = nodes[i * 3: (i + 1) * 3]
    triangle = [node_dict[n1], node_dict[n2], node_dict[n3]]
    if skin_width==-1:
        tri_poly = Polygon(triangle)
        outer_contour_points = [(outer_contour[k], outer_contour[k + 1]) for k in range(1, len(outer_contour), 2)]
        outer_poly=Polygon(outer_contour_points)
        eps = 1e-9
        for x, y in triangle:
            if outer_poly.boundary.distance(Point(x, y)) < eps:
                return 4, tags[i]
    max_intersection = 0
    best_class = outer_contour_class
    # Compare triangle with all contours
    for j in range(len(contours)):
        contour_class = contours[j][0]
        contour_points = [(contours[j][k], contours[j][k + 1]) for k in range(1, len(contours[j]), 2)]
        contour_poly = Polygon(contour_points)
        try:
            intersection = Polygon(triangle).intersection(contour_poly).area
            if intersection > max_intersection:
                max_intersection = intersection
                best_class = contour_class
        except:
            pass
    return best_class, tags[i]


def export_mesh_for_femm(filename, class_groups, isSaveToFile = False):
    """
    Export a 2D triangular mesh from Gmsh to a simple text-based format suitable for use in FEMM or other tools.

    Parameters:
    -----------
    filename : str
        Path to the output file (e.g. "mesh_for_femm.txt")

    class_groups : dict
        Dictionary mapping class IDs to lists of triangle element tags.
        Example: {0: [1, 2, 3], 1: [4, 5, 6]}

    isSaveToFile: bool
        Is result need saving to file

    Returns:
    --------
    Dictionary['NODES, TRIANGLES, CLASSES']: A dictionary containing information for femm

    Notes:
    ------
    - This function assumes that Gmsh has already generated a 2D triangular mesh.
    - Each triangle is written with its three vertex indices and the associated class ID.
    - The output format is simple and human-readable:
        - First: list of nodes (with IDs and coordinates)
        - Then: list of triangles (with node IDs and class label)
    """
    # Nodes
    node_tags, node_coords, _ = gmsh.model.mesh.getNodes()
    node_dict = {
        tag: (node_coords[i * 3], node_coords[i * 3 + 1])
        for i, tag in enumerate(node_tags)
    }

    # Elements
    elem_types, elem_tags, elem_node_tags = gmsh.model.mesh.getElements(2)

    # Combining triangles list with their classes
    triangle_data = []
    dictionary_for_femm={'NODES' : [], 'TRIANGLES': [], 'CLASS': []}
    used_nodes = set()
    for elem_type, tags, nodes in zip(elem_types, elem_tags, elem_node_tags):
        if elem_type == 2:  # Triangles
            for i in range(len(tags)):
                elem_tag = tags[i]
                n1, n2, n3 = nodes[i * 3:(i + 1) * 3]
                used_nodes.update([n1, n2, n3])  # adding nodes to set
                class_id = None
                for cid, tag_list in class_groups.items():
                    if elem_tag in tag_list:
                        class_id = cid
                        break
                triangle_data.append((n1, n2, n3, class_id))

    # Map: node tag → local index
    # Must save only used nodes
    tag_to_index = {tag: i+1 for i, tag in enumerate(sorted(used_nodes))}

    for tag in sorted(used_nodes):
        x, y = node_dict[tag]
        dictionary_for_femm['NODES'].append([float(x), float(y)])
    for n1, n2, n3, class_id in triangle_data:
        dictionary_for_femm['TRIANGLES'].append([int(tag_to_index[n1])-1, int(tag_to_index[n2])-1, int(tag_to_index[n3])-1])
        dictionary_for_femm['CLASS'].append(int(float(class_id)))

    if isSaveToFile is True:
        # Saving into file
        with open(filename, 'w') as f:
            f.write("# NODES\n")
            for tag in sorted(used_nodes):
                x, y = node_dict[tag]
                idx = tag_to_index[tag]
                f.write(f"{idx} {x:.12f} {y:.12f}\n")
            f.write("\n# TRIANGLES\n")
            for n1, n2, n3, class_id in triangle_data:
                f.write(f"{tag_to_index[n1]} {tag_to_index[n2]} {tag_to_index[n3]} {class_id}\n")
        print(f"Mesh exported to: {filename}, finite elements count - {len(triangle_data)}")
    return dictionary_for_femm

def show_class(class_groups, class_for_showing=-1):
    """
        Hides all elements in the specified class within the Gmsh GUI visualization window.

        Parameters:
        -----------
        class_groups : dict
            A dictionary mapping class IDs to lists of element tags.
            Example: {0: [101, 102], 1: [103, 104]}

        class_for_showing : int, optional (default: -1)
            The class ID to hide in the visualization.
            All other classes will remain visible. If set to -1, no class will be hidden.

        Behavior:
        ---------
        This function uses Gmsh's visibility API to hide the mesh elements associated
        with the given class ID. It then launches the Gmsh interactive GUI (via `gmsh.fltk.run`)
        so the user can visually inspect the remaining visible mesh elements.

        Note:
        -----
        - The function does not modify the mesh or the physical groups — it only affects visibility in the GUI.
        - Use this function for visual verification of how elements are grouped into classes.
        """
    for class_id, elements in class_groups.items():
        if class_id == class_for_showing:
            gmsh.model.mesh.setVisibility(elements, False)
    gmsh.fltk.run()


def get_image(class_groups, image_size=(1000, 1000), margin=10):
    """
    Generates a color-coded image representing finite element classes using OpenCV.

    Each class of elements is drawn in a unique color, with triangle (or polygon) borders rendered in black.

    Parameters:
    -----------
    class_groups : dict
        A dictionary mapping class IDs to lists of element tags.
        Example: {0: [101, 102], 1: [103, 104], ...}

    image_size : tuple of int, optional
        Size of the output image in pixels, as (width, height). Default is (1000, 1000).

    margin : int, optional
        Margin size (in pixels) around the rendered mesh. Default is 10.

    Returns:
    --------
    numpy.ndarray
        A NumPy array representing the RGB image (dtype=np.uint8) with elements
        filled in class-specific colors and outlined in black.

    Notes:
    ------
    - This function automatically retrieves all mesh nodes using `gmsh.model.mesh.getNodes()`.
    - It uses `gmsh.model.mesh.getElement()` to retrieve node connectivity for each element tag.
    - Elements are rendered using OpenCV functions: `cv2.fillPoly` for fill, `cv2.polylines` for borders.
    - If an element tag is invalid or not found in the current mesh, it will be silently skipped.
    """
    width, height = image_size
    img = np.zeros((height, width, 3), dtype=np.uint8)

    node_tags, node_coords, _ = gmsh.model.mesh.getNodes()
    # Dict {nodeTag: (x, y)}
    node_coordinates = {
        tag: (node_coords[i], node_coords[i + 1])
        for tag, i in zip(node_tags, range(0, len(node_coords), 3))
    }

    all_coordinates = np.array(list(node_coordinates.values()))
    min_x, min_y = all_coordinates.min(axis=0)
    max_x, max_y = all_coordinates.max(axis=0)

    def to_pixel(x, y):
        px = int((x - min_x) / (max_x - min_x) * (width - 2 * margin) + margin)
        py = int((max_y - y) / (max_y - min_y) * (height - 2 * margin) + margin)
        return px, py

    # Colors for classes
    class_colors = [(255, 255, 255), (0, 0, 255), (255, 255, 0), (0, 255, 255)]

    for class_id, element_tags in class_groups.items():
        for tag in element_tags:
            try:
                element_type, node_tag_list, _, _ = gmsh.model.mesh.getElement(tag)
                num_nodes = gmsh.model.mesh.getElementProperties(element_type)[3]
                # Dividing nodeTagList in elements with needed nodes count
                for i in range(0, len(node_tag_list), num_nodes):
                    nodes = node_tag_list[i:i + num_nodes]
                    pts = [to_pixel(*node_coordinates[n]) for n in nodes]
                    pts_np = np.array(pts, dtype=np.int32)
                    cv2.fillPoly(img, [np.array(pts, dtype=np.int32)], class_colors[int(class_id)])
                    cv2.polylines(img, [pts_np], isClosed=True, color=(0, 0, 0), thickness=1)
            except:
                pass  # sometimes getElement cannot find tag — leave it
    return img


def create_mesh(pixel_spacing, polygons, lc=7, distance_threshold=1.3, skin_width = 0, is_show_inner_contours=False,
                show_meshing_result_method="opencv",
                number_of_showed_class=-1, is_saving_to_file=False, export_filename=None):
    """
    Creates a 2D triangular mesh from contour data and optionally exports or visualizes it.

    Parameters:
    -----------
    pixel_spacing: [0.682, 0.682] - ratio between pixels and millimeters on x and y axis
    polygons: List containing contour data.
            Each line should represent a contour in the format:
            <class_id> x1 y1 x2 y2 ... xn yn. Example:
            ['3 93 390 93 391 94 392 94 393 95 394 98 394 98 393 97 393 94 390 93 390',
               '3 93 390 93 391 94 392 94 393 95 394 98 394 98 393 97 393 94 390 93 390', ...]

    lc : float, optional (default=7)
        Mesh size parameter passed to Gmsh. Controls the fineness of the mesh.

    distance_threshold : float, optional (default=1.3)
        Distance threshold used when merging collinear segments in the contour.

    skin_width: float, optional (default=7)
        Width of skin to be added. If value is 0, no skin will be added
        If value is -1, innermost finite elements will be considered as skin
        Otherwise outer contour with specified width will be added

    is_show_inner_contours : bool, optional (default=False)
        If True, all inner contours (not just the outer boundary) will be shown and meshed.
        If False, only the outermost contour is used for geometry creation.

   show_meshing_result_method : string, optional (default="no")
        If method=="gmsh", launches the Gmsh GUI (`gmsh.fltk.run()`) and displays the mesh
        for the specified class (see `number_of_showed_class`).
        If method=="opencv", creates a numpy array containing image for further display

    number_of_showed_class : int, optional (default=-1)
        Specifies which class to display in the Gmsh GUI when `iShowMeshingResult` is True.
        If set to -1, no class is explicitly hidden.

    is_saving_to_file : bool, optional (default=False)
        If True, exports the mesh with class information to a format compatible with FEMM.

    export_filename : str or None, optional (default=None)
        Path to save the FEMM-compatible mesh file.
        Required if `isExportingToFemm` is True.

    Returns:
    --------
    mesh_image: numpy.array
    A NumPy array representing the RGB image (dtype=np.uint8) with elements
        filled in class-specific colors and outlined in black
    mesh_data: data for femm generation of synthetic datasets

    Side Effects:
    -------------
    - Initializes and finalizes the Gmsh API.
    - Creates a 2D mesh using Gmsh.
    - Optionally launches the GUI for visual inspection.
    - Optionally exports the mesh to FEMM.

    Notes:
    ------
    - The outermost contour is automatically detected based on largest area.
    - Inner contours are only included if `isShowInnerContours=True`.
    - Triangle elements are grouped by class using geometric intersection with contours.
    - Mesh elements are exported to FEMM with physical group labels.

    Dependencies:
    -------------
    - Requires functions: `largest_segment_area_index`, `merge_collinear_segments`,
      `divide_triangles_into_groups`, `show_class`, `export_mesh_for_femm`.
    - Uses the `gmsh` Python API.
    """
    mesh_image = np.array([512, 512, 3])
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)
    contours = []
    outer_contour = None
    outer_contour_class = None
    outer_segment_area = None
    #Not counting the polygon not defining the class data
    for polygon in polygons:
        if polygon[0]=='4':
            polygons.remove(polygon)
            break
    outer_segment = largest_segment_area_index(polygons)
    if skin_width>0:
        outer_segment, polygons = add_skin(outer_segment, polygons, skin_width)
    for k in range(len(polygons)):
        geometry_points = []
        geometry_lines = []
        area = list(map(float, polygons[k].strip().split(' ')))
        if k != outer_segment:
            contours.append(area)
        area = [area[0]] + merge_collinear_segments(area[1:], distance_threshold)
        if k == outer_segment or is_show_inner_contours:
            for i in range(1, len(area) - 1):
                if i % 2:
                    geometry_points.append(gmsh.model.geo.add_point(area[i], area[i + 1], 0, lc))
            for i in range(-1, len(geometry_points) - 1):
                if geometry_points[i] != geometry_points[i + 1]:
                    geometry_lines.append(gmsh.model.geo.add_line(geometry_points[i], geometry_points[i + 1]))
            if k == outer_segment:
                outer_contour_class = int(area[0])
                outer_segment_area = area
                outer_contour = gmsh.model.geo.add_curve_loop(geometry_lines)

    gmsh.model.geo.add_plane_surface([outer_contour])
    gmsh.model.geo.removeAllDuplicates()
    gmsh.model.geo.synchronize()

    # Generate mesh:
    gmsh.model.mesh.generate(2)

    class_groups = divide_triangles_into_groups(contours, outer_contour_class, outer_segment_area, skin_width)
    img = None
    if show_meshing_result_method == "gmsh":
        show_class(class_groups, number_of_showed_class)
    elif show_meshing_result_method == "opencv":
        img = get_image(class_groups)
    mesh_data = export_mesh_for_femm(export_filename, class_groups, is_saving_to_file)
    # It finalizes the Gmsh API
    gmsh.finalize()
    return img, mesh_data


def largest_segment_area_index(polygons):
    """
        Determines the index (line number) of the contour with the largest area in the input file.

        Parameters:
        -----------
        polygons: List containing contour data.
            Each line should represent a contour in the format:
            <class_id> x1 y1 x2 y2 ... xn yn. Example:
            ['3 93 390 93 391 94 392 94 393 95 394 98 394 98 393 97 393 94 390 93 390',
               '3 93 390 93 391 94 392 94 393 95 394 98 394 98 393 97 393 94 390 93 390', ...]

        Returns:
        --------
        int
            The index (starting from 0) of the line representing the largest area polygon.
            Returns -1 if no valid polygons are found.

        Notes:
        ------
        - The function ignores lines with an odd number of coordinates (invalid pairs).
        - Polygons are closed automatically if the first and last points do not match.
        - Lines that cause parsing errors or contain invalid geometries are skipped.
        - Uses Shapely to calculate polygon areas.

        Exceptions:
        -----------
        - Lines that fail due to incorrect float conversion or topological issues in Shapely
          are safely skipped.
        """
    max_area = 0.0
    max_index = -1
    for idx, line in enumerate(polygons, start=0):
        parts = line.strip().split()
        coords = parts[1:]

        if len(coords) % 2 != 0:
            continue  # Skip lines with incorrect number of coordinates

        try:
            points = [(float(coords[i]), float(coords[i + 1])) for i in range(0, len(coords), 2)]

            # Close the polygon if it's not already closed
            if points[0] != points[-1]:
                points.append(points[0])

            polygon = Polygon(points)
            area = polygon.area

            if area > max_area:
                max_area = area
                max_index = idx

        except (ValueError, shapely.errors.TopologicalError):
            continue  # Skip lines with invalid float values or problematic geometry

    return max_index


def merge_collinear_segments(contour, distance_threshold=1.3):
    """
    Merges nearly collinear segments in a contour by removing intermediate points
    that lie close to a straight line and do not significantly alter the shape.

    Parameters:
    -----------
    contour : list of float
        A flat list of x and y coordinates representing the contour.
        Example: [x1, y1, x2, y2, ..., xn, yn]

    distance_threshold : float, optional (default=1.3)
        The maximum perpendicular distance from a point to a line
        for it to be considered approximately collinear.

    Returns:
    --------
    list of float
        A simplified contour with unnecessary points removed.

    Notes:
    ------
    - If the contour has fewer than 3 points (6 coordinates), it is returned unchanged.
    - A point is removed if it lies close to the line segment formed by two previous points.
    - This helps to reduce the number of points in long straight or gently curving segments
      without significantly affecting the overall shape.
    - Relies on the helper function `point_line_distance(px, py, x1, y1, x2, y2)` to
      calculate the perpendicular distance from a point to a line segment.

    Example:
    --------
    Input:  [0, 0, 1, 0, 2, 0.1, 3, 0]
    Output: [0, 0, 1, 0, 3, 0]
    """
    if len(contour) < 6:
        return contour  # Not enough points to merge

    merged_contour = contour[:2]  # Start with the first point

    for i in range(2, len(contour) - 2, 2):
        # Check if the point lies close to the line formed by the last two segments
        if len(merged_contour) >= 4:
            x1, y1 = merged_contour[-4], merged_contour[-3]
            x2, y2 = merged_contour[-2], merged_contour[-1]
            px, py = contour[i], contour[i + 1]
            if point_line_distance(px, py, x1, y1, x2, y2) < distance_threshold:
                continue  # Skip the point if it is close to the line

        merged_contour.extend(contour[i:i + 2])  # Add the new segment point

    merged_contour.extend(contour[-2:])  # Always include the last point
    return merged_contour


def point_line_distance(px, py, x1, y1, x2, y2):
    """
        Calculates the perpendicular distance from a point (px, py) to the line
        defined by two points (x1, y1) and (x2, y2).

        Parameters:
        -----------
        px, py : float
            Coordinates of the point from which the distance is measured.

        x1, y1 : float
            Coordinates of the first point on the line.

        x2, y2 : float
            Coordinates of the second point on the line.

        Returns:
        --------
        float
            The shortest (perpendicular) distance from the point (px, py) to the line.

        Notes:
        ------
        - If the two line points (x1, y1) and (x2, y2) are identical,
          the function returns the Euclidean distance from the point to this single point.
        - This function uses the standard formula for point-to-line distance in 2D space.
        """
    if (x1, y1) == (x2, y2):
        return np.linalg.norm([px - x1, py - y1])
    return abs((y2 - y1) * px - (x2 - x1) * py + x2 * y1 - y2 * x1) / np.linalg.norm([x2 - x1, y2 - y1])


def add_skin(outer_segment, polygons, skin_width):
    """
    Create an additional contour around the outer contour (offset by skin_width).

    :param outer_segment: int, index of the outer contour in polygons
    :param polygons: list[str], format "<class_id> x1 y1 x2 y2 ... xn yn"
    :param skin_width: float, offset distance
    :return: new_outer_segment, polygons
    """
    # Parse the outer contour
    parts = polygons[outer_segment].split()
    class_id = 4 #skin
    coords = list(map(float, parts[1:]))
    points = [(coords[i], coords[i + 1]) for i in range(0, len(coords), 2)]

    # Create a shapely polygon
    poly = Polygon(points)

    # Offset outward (buffer)
    outer_poly = poly.buffer(skin_width)

    # Extract exterior coordinates of the new polygon
    new_points = list(outer_poly.exterior.coords)

    # Convert back to the polygons format
    new_line = str(class_id) + " " + " ".join(f"{x:.6f} {y:.6f}" for x, y in new_points)

    # Append the new contour
    polygons.append(new_line)

    # Return the index of the new contour
    return len(polygons) - 1, polygons

def test_module():
    test_list = [
        '3 33 124 0 202 0 299 28 378 74 438 154 463 261 452 381 459 441 434 482 378 511 301 511 212 486 140 452 105 332 58 176 58 33 124',
        '0 249 417 237 417 235 419 234 419 233 420 226 420 226 425 238 425 238 420 239 419 240 419 241 418 246 418 247 417 249 417',
        '0 147 414 147 417 170 417 170 414 168 416 166 416 165 417 156 417 155 416 151 416 150 415 149 415 148 414 147 414',
        '0 215 410 215 414 216 415 216 416 221 416 221 410 215 410',
        '0 200 410 200 416 202 416 203 417 201 419 198 417 194 417 194 422 200 422 201 423 207 423 208 422 208 420 207 419 207 417 211 413 211 410 200 410',
        '0 247 406 252 406 253 407 260 407 261 406 262 406 258 406 257 405 254 405 253 406 247 406',
        '0 226 407 234 407 235 406 244 406 237 406 236 405 230 405 229 406 226 406 226 407',
        '0 386 401 382 401 379 398 370 401 366 408 349 407 345 411 342 411 340 407 336 405 325 406 317 404 310 407 306 407 306 413 310 413 313 416 314 421 349 421 349 417 351 415 376 410 383 406 386 401',
        '0 267 398 267 402 269 403 278 403 280 404 284 409 284 410 282 412 280 413 277 413 274 417 283 417 284 418 290 418 291 417 293 417 293 406 289 406 287 405 285 401 285 398 267 398',
        '0 94 378 94 391 98 391 109 402 111 402 114 406 117 406 121 410 123 410 124 413 129 416 129 421 137 421 137 414 133 414 130 410 125 410 117 401 115 401 102 388 98 383 98 378 94 378',
        '0 421 362 415 362 415 365 409 375 407 377 406 377 405 379 403 381 402 381 401 383 399 385 398 385 394 390 390 390 392 392 392 395 393 396 393 395 397 393 399 390 401 390 402 389 402 387 404 386 414 376 417 371 421 371 421 362',
        '0 62 326 62 341 66 341 68 343 68 344 73 348 73 349 74 350 74 353 81 353 81 342 77 342 74 339 74 337 71 334 71 331 69 328 69 326 62 326',
        '0 449 302 441 302 441 307 440 308 440 309 436 313 434 313 434 329 438 329 438 326 442 322 442 320 443 319 443 318 446 315 449 315 449 302',
        '0 62 298 62 304 65 304 65 303 64 302 64 301 63 300 63 299 62 298',
        '0 290 274 290 284 293 284 294 285 294 287 299 287 301 289 301 293 302 293 302 288 305 285 304 284 304 278 303 278 302 277 302 274 290 274',
        '0 54 266 54 294 59 294 60 295 62 295 63 294 63 292 64 291 65 288 64 287 64 284 63 283 63 279 62 278 62 271 61 270 61 266 54 266',
        '0 449 237 447 237 448 238 448 244 449 245 448 247 448 253 445 256 442 256 442 281 445 281 445 278 448 275 450 275 455 269 455 260 456 259 456 256 455 255 455 250 450 250 449 249 449 237',
        '0 182 204 181 205 179 205 181 208 181 215 177 219 177 220 174 223 172 223 171 222 166 222 166 244 173 243 174 239 176 237 176 229 177 228 177 226 178 224 181 221 182 221 182 220 184 218 184 217 182 215 182 204',
        '0 458 182 458 198 461 198 464 200 465 205 468 208 469 213 472 217 480 224 482 224 485 222 489 222 489 214 486 214 483 210 483 206 480 204 474 197 471 192 471 190 467 186 467 182 458 182',
        '0 438 182 438 224 446 224 447 220 448 219 448 211 447 210 447 198 444 194 444 189 442 186 442 182 438 182',
        '0 81 182 73 182 72 190 69 193 68 196 68 202 65 205 62 205 62 237 69 237 70 225 71 222 75 218 75 207 76 204 78 202 81 202 81 182',
        '0 59 174 46 174 46 177 39 184 38 184 29 194 26 194 26 209 35 209 35 205 38 203 41 203 46 198 47 196 47 193 50 190 50 189 53 186 54 186 54 185 57 182 58 182 58 180 59 179 59 174',
        '0 385 114 365 100 317 105 277 148 241 144 234 132 248 131 263 118 282 137 287 121 267 106 268 90 260 90 254 109 234 117 233 131 209 103 191 97 159 96 134 108 181 107 218 123 228 136 241 182 259 197 278 183 295 140 311 123 342 110 385 114',
        '1 432 440 405 452 401 452 381 460 361 461 347 464 373 464 395 460 413 454 423 449 432 440',
        '1 308 414 312 418 312 421 313 421 313 418 312 417 312 416 310 414 308 414',
        '1 182 411 181 411 180 412 176 412 180 412 181 411 182 411',
        '1 287 419 273 417 283 410 278 404 266 402 275 410 234 417 229 411 236 407 225 408 225 405 263 408 264 405 258 403 263 406 253 408 215 404 196 408 212 410 207 424 193 422 194 416 202 417 198 412 183 417 177 426 199 429 213 426 220 419 225 422 237 416 250 417 239 421 264 416 287 419',
        '1 286 401 287 402 287 404 288 404 289 405 290 405 289 405 287 403 287 402 286 401',
        '1 299 406 300 407 300 409 305 412 305 407 306 406 310 406 311 405 313 405 314 404 316 404 317 403 320 403 321 404 324 404 325 405 327 405 328 404 331 404 331 402 330 402 329 401 317 401 316 400 312 400 311 399 305 399 304 398 303 398 302 399 299 406',
        '1 283 395 280 395 279 396 274 396 275 396 276 397 282 397 283 396 283 395',
        '1 99 382 103 388 105 389 106 391 108 392 115 400 117 400 119 403 124 407 125 409 130 409 133 413 135 413 136 412 141 412 136 412 135 411 133 411 131 409 121 404 118 400 116 400 100 384 99 382',
        '1 76 373 77 378 85 386 96 394 100 395 107 403 115 407 123 415 128 418 128 416 125 415 123 413 123 411 121 411 117 407 114 407 111 403 109 403 98 392 93 391 93 378 94 377 92 375 89 379 85 380 78 372 76 373',
        '1 87 364 87 365 88 366 88 367 90 369 88 367 88 366 87 365 87 364',
        '1 426 362 417 359 416 361 422 362 422 371 417 372 393 397 389 390 394 389 414 365 387 392 368 400 356 401 362 403 352 402 351 406 340 406 341 409 345 410 349 406 366 407 370 400 379 397 387 401 383 407 350 418 360 418 382 410 426 366 426 362',
        '1 83 357 83 358 84 359 84 360 84 359 83 358 83 357',
        '1 64 342 65 343 65 345 66 346 69 346 71 348 72 348 71 347 70 347 67 344 67 343 66 342 64 342',
        '1 74 336 75 337 75 339 77 341 76 340 76 339 75 338 75 337 74 336',
        '1 433 320 432 321 432 329 431 330 431 331 431 330 432 329 432 328 433 327 433 320',
        '1 446 316 444 318 444 319 443 320 445 318 445 317 446 316',
        '1 444 282 444 292 443 293 443 300 442 301 443 300 443 295 444 294 444 282',
        '1 325 267 328 267 329 268 330 268 332 270 334 270 337 273 337 274 338 274 339 275 340 275 341 276 342 276 343 275 344 275 345 274 345 273 343 271 343 270 341 268 340 268 339 267 339 266 338 265 334 265 333 264 332 264 331 263 330 264 330 265 328 267 325 267',
        '1 316 256 302 255 302 259 294 260 292 270 280 273 279 282 274 287 251 281 252 294 241 308 230 303 210 302 197 308 196 314 203 315 200 337 208 350 230 367 249 366 244 354 246 332 265 343 285 344 293 339 302 322 304 293 309 280 306 277 310 274 318 282 321 275 316 256',
        '1 456 251 456 255 457 256 457 259 456 260 456 269 452 273 452 274 450 276 448 276 447 277 448 276 451 276 454 273 455 273 456 272 456 262 457 261 457 253 456 252 456 251',
        '1 442 225 443 226 443 228 444 229 444 233 445 234 445 235 446 236 445 235 445 234 444 233 444 228 443 227 443 226 444 225 445 225 442 225',
        '1 72 222 72 224 71 225 71 232 70 233 70 234 70 233 71 232 71 225 72 224 72 222',
        '1 339 211 338 212 337 211 333 216 332 220 329 223 328 227 326 229 322 231 317 231 316 230 318 233 318 240 319 241 314 244 312 242 312 240 312 246 314 249 314 252 316 250 319 250 321 248 327 253 327 256 327 252 330 249 332 242 331 240 331 230 334 225 335 220 338 217 339 211',
        '1 181 199 180 200 179 200 178 201 178 204 181 204 181 203 182 202 182 200 181 199',
        '1 457 184 456 194 454 196 449 197 445 194 448 198 449 219 447 224 450 219 456 226 457 234 460 237 460 248 465 255 470 253 470 249 473 243 472 226 474 224 483 225 487 223 480 225 471 217 464 205 463 200 457 198 457 184',
        '1 438 180 436 180 434 182 433 181 435 184 435 202 436 203 436 210 437 211 437 213 437 182 438 181 439 181 438 180',
        '1 461 181 468 182 468 186 472 190 473 194 484 206 485 212 490 214 490 221 493 219 495 221 496 227 493 234 495 235 501 233 502 220 500 217 496 215 496 210 499 206 499 201 495 195 494 187 491 185 489 186 488 194 485 197 478 191 473 181 469 179 461 181',
        '1 431 173 431 176 432 177 432 179 432 177 431 176 431 173',
        '1 45 169 38 180 21 177 13 194 11 222 18 238 15 226 20 208 30 210 25 209 25 194 46 173 60 174 59 182 36 207 41 207 41 226 51 229 50 240 39 238 40 267 61 339 57 302 61 296 53 294 53 266 61 265 52 263 61 218 54 218 48 205 65 204 72 185 64 194 59 189 60 174 45 169',
        '1 225 164 224 165 224 166 222 168 224 166 224 165 225 164',
        '1 424 161 424 162 427 165 427 166 428 167 428 168 428 167 427 166 427 165 424 162 424 161',
        '1 442 160 441 161 441 166 442 167 443 167 446 170 447 170 449 172 449 173 450 173 450 166 451 165 451 164 453 162 452 162 451 161 449 161 448 160 447 160 446 159 445 159 444 160 442 160',
        '1 49 152 48 153 43 153 42 154 42 155 40 157 37 157 37 158 36 159 36 160 35 161 33 161 33 162 37 162 38 161 38 160 41 157 41 156 42 155 43 155 44 154 48 154 49 153 49 152',
        '1 81 150 79 150 78 149 75 149 74 148 67 148 61 154 61 155 60 156 60 161 62 163 64 163 65 164 66 164 68 167 71 167 72 166 73 166 74 165 74 164 81 157 81 155 82 154 82 152 81 151 81 150',
        '1 62 144 61 145 57 145 57 147 59 147 60 146 62 146 62 144',
        '1 126 123 123 124 123 125 122 126 118 126 118 129 117 130 116 130 114 132 114 134 113 135 112 135 111 136 111 137 107 140 107 141 103 145 103 146 99 150 99 151 99 150 104 145 104 144 107 142 107 141 115 133 115 132 116 131 117 131 125 123 126 123',
        '1 326 117 299 135 279 183 253 202 258 194 240 182 223 129 195 112 226 134 225 142 219 131 236 172 231 176 220 169 200 182 200 203 220 219 211 229 223 234 214 243 208 235 213 246 192 265 197 275 216 277 225 264 239 268 255 261 263 246 247 236 245 213 237 215 236 207 250 207 270 195 296 141 326 117',
        '1 338 113 376 113 377 112 378 113 379 113 378 113 377 112 374 112 373 111 342 111 341 112 339 112 338 113',
        '1 133 109 133 112 137 112 138 111 141 111 141 110 140 110 139 111 134 111 133 110 133 109',
        '1 143 109 144 109 145 108 149 108 150 109 184 109 182 109 181 108 171 108 170 107 161 107 160 106 156 106 155 107 152 107 151 108 145 108 144 109 143 109',
        '1 422 91 426 92 428 94 432 96 434 96 439 99 442 99 450 104 452 104 470 116 461 103 457 101 449 99 446 97 443 97 442 96 439 96 432 93 428 93 427 92 422 91',
        '1 235 133 241 143 277 147 313 106 338 99 365 99 384 110 386 114 382 118 387 113 362 95 365 89 358 85 321 77 287 77 272 84 268 106 288 121 283 140 277 144 272 142 272 135 277 134 263 119 256 120 248 132 235 133',
        '1 135 107 141 102 159 95 191 96 209 102 228 123 233 125 233 130 233 117 254 108 259 102 259 92 249 81 235 77 225 80 215 76 184 76 148 85 161 89 154 96 141 99 135 107',
        '2 380 114 336 114 314 124 297 141 280 185 245 214 246 231 292 241 308 255 312 230 326 228 340 210 328 258 346 274 325 268 318 283 308 282 304 346 264 386 265 396 351 405 398 383 428 339 442 300 447 238 437 224 430 171 380 114',
        '2 142 110 100 150 82 183 82 202 62 256 67 325 82 342 92 374 133 410 165 416 215 403 257 404 242 385 194 363 176 334 188 272 212 246 205 230 219 222 199 203 197 190 227 161 223 137 194 112 142 110']

    create_mesh(['1', '1'], test_list,
                7,
                1.3, 0, True,
                show_meshing_result_method="gmsh",
                number_of_showed_class=3,
                is_saving_to_file=True,
                export_filename="tmp.txt")


if __name__ == "__main__":
    test_module()
