# Finite element mesh generator for FEMM
import gmsh
import numpy as np
from shapely.geometry import Polygon
import shapely.errors


def divide_triangles_into_groups(contours):
    """
        Divides 2D triangular mesh elements into classes based on intersection with given contours.

        Parameters:
        -----------
        contours : list of lists
            A list of contours. Each contour is a list where:
            - The first element is the class ID (an integer).
            - The remaining elements represent the coordinates of the contour's points
              in the format [x1, y1, x2, y2, ..., xn, yn].

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
    for i in range(len(contours)):
        k += 1
        if k <= len(contours) - 1 and len(contours[k]) < 9:
            contours.pop(k)
            k -= 1
    # Retrieve all 2D elements (triangles)
    elem_types, elem_tags, elem_node_tags = gmsh.model.mesh.getElements(2)
    # Get all mesh nodes and their coordinates
    node_tags, node_coords, _ = gmsh.model.mesh.getNodes()
    node_dict = {tag: (node_coords[i * 3], node_coords[i * 3 + 1]) for i, tag in enumerate(node_tags)}
    class_groups = {}
    # Loop over all elements
    for elem_type, tags, nodes in zip(elem_types, elem_tags, elem_node_tags):
        if elem_type == 2:  # Element type 2 corresponds to triangles
            for i in range(len(tags)):
                # Get triangle vertex coordinates
                n1, n2, n3 = nodes[i * 3: (i + 1) * 3]
                triangle = [node_dict[n1], node_dict[n2], node_dict[n3]]
                max_intersection = 0
                best_class = None
                # Compare triangle with all contours
                for j in range(len(contours)):
                    contour_class = contours[j][0]
                    contour_points = [(contours[j][k], contours[j][k + 1]) for k in range(1, len(contours[j]), 2)]
                    contour_poly = Polygon(contour_points)
                    intersection = Polygon(triangle).intersection(contour_poly).area
                    if intersection > max_intersection:
                        max_intersection = intersection
                        best_class = contour_class
                # Assign triangle to the best matching class
                if best_class is not None:
                    if best_class not in class_groups:
                        class_groups[best_class] = []
                    class_groups[best_class].append(tags[i])
        # Create physical groups in Gmsh for each class
        for class_id, elements in class_groups.items():
            group = gmsh.model.addPhysicalGroup(2, elements)
            gmsh.model.setPhysicalName(2, group, f"Class_{class_id}")
    return class_groups


def export_mesh_for_femm(filename, class_groups):
    """
    Export a 2D triangular mesh from Gmsh to a simple text-based format suitable for use in FEMM or other tools.

    Parameters:
    -----------
    filename : str
        Path to the output file (e.g. "mesh_for_femm.txt")

    class_groups : dict
        Dictionary mapping class IDs to lists of triangle element tags.
        Example: {0: [1, 2, 3], 1: [4, 5, 6]}

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
    # Map: node tag → local index
    tag_to_index = {tag: i + 1 for i, tag in enumerate(sorted(node_tags))}

    # Elements
    elem_types, elem_tags, elem_node_tags = gmsh.model.mesh.getElements(2)

    # Combining triangles list with their classes
    triangle_data = []
    for elem_type, tags, nodes in zip(elem_types, elem_tags, elem_node_tags):
        if elem_type == 2:  # Треугольники
            for i in range(len(tags)):
                elem_tag = tags[i]
                n1, n2, n3 = nodes[i * 3:(i + 1) * 3]
                class_id = None
                for cid, tag_list in class_groups.items():
                    if elem_tag in tag_list:
                        class_id = cid
                        break
                triangle_data.append((tag_to_index[n1], tag_to_index[n2], tag_to_index[n3], class_id))

    # Saving into file
    with open(filename, 'w') as f:
        f.write("# NODES\n")
        for tag in sorted(node_tags):
            x, y = node_dict[tag]
            idx = tag_to_index[tag]
            f.write(f"{idx} {x:.12f} {y:.12f}\n")

        f.write("\n# TRIANGLES\n")
        for n1, n2, n3, class_id in triangle_data:
            f.write(f"{n1} {n2} {n3} {class_id}\n")

    print(f"Mesh exported to: {filename}")


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


def create_mesh(pixel_spacing, polygons, lc=7, distance_threshold=1.3, isShowInnerContours=False, iShowMeshingResult=False,
                number_of_showed_class=-1, isExportingToFemm=False, export_filename=None):
    """
    Creates a 2D triangular mesh from contour data and optionally exports or visualizes it.

    Parameters:
    -----------
    pixel_spacing: [0.682, 0.682]
    polygons: ['3 93 390 93 391 94 392 94 393 95 394 98 394 98 393 97 393 94 390 93 390',
               '3 93 390 93 391 94 392 94 393 95 394 98 394 98 393 97 393 94 390 93 390', ...]

    lc : float, optional (default=7)
        Mesh size parameter passed to Gmsh. Controls the fineness of the mesh.

    distance_threshold : float, optional (default=1.3)
        Distance threshold used when merging collinear segments in the contour.

    isShowInnerContours : bool, optional (default=False)
        If True, all inner contours (not just the outer boundary) will be shown and meshed.
        If False, only the outermost contour is used for geometry creation.

    iShowMeshingResult : bool, optional (default=False)
        If True, launches the Gmsh GUI (`gmsh.fltk.run()`) and displays the mesh
        for the specified class (see `number_of_showed_class`).

    number_of_showed_class : int, optional (default=-1)
        Specifies which class to display in the Gmsh GUI when `iShowMeshingResult` is True.
        If set to -1, no class is explicitly hidden.

    isExportingToFemm : bool, optional (default=False)
        If True, exports the mesh with class information to a format compatible with FEMM.

    export_filename : str or None, optional (default=None)
        Path to save the FEMM-compatible mesh file.
        Required if `isExportingToFemm` is True.

    Returns:
    --------
    None

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
    mesh_image = np.array([512,512,3])
    mesh_data = ''
    gmsh.initialize()
    contours = []
    outer_contour = None
    with open(filepath) as file:
        k = -1
        outer_segment = largest_segment_area_index(file)
        file.seek(0)
        for line in file:
            k += 1
            geometry_points = []
            geometry_lines = []
            area = list(map(float, line.strip().split(' ')))
            if k != outer_segment:
                contours.append(area)
            area = [area[0]] + merge_collinear_segments(area[1:], distance_threshold)
            if k == outer_segment or isShowInnerContours:
                for i in range(1, len(area) - 1):
                    if i % 2:
                        geometry_points.append(gmsh.model.geo.add_point(area[i], area[i + 1], 0, lc))
                for i in range(-1, len(geometry_points) - 1):
                    if geometry_points[i] != geometry_points[i + 1]:
                        geometry_lines.append(gmsh.model.geo.add_line(geometry_points[i], geometry_points[i + 1]))
                if k == outer_segment:
                    outer_contour = gmsh.model.geo.add_curve_loop(geometry_lines)

    gmsh.model.geo.add_plane_surface([outer_contour])
    gmsh.model.geo.removeAllDuplicates()
    gmsh.model.geo.synchronize()

    # Generate mesh:
    gmsh.model.mesh.generate(2)

    class_groups = divide_triangles_into_groups(contours)
    if iShowMeshingResult:
        show_class(class_groups, 0)
    if isExportingToFemm and export_filename is not None:
        export_mesh_for_femm(export_filename, class_groups)

    # It finalize the Gmsh API
    gmsh.finalize()
    return mesh_image, mesh_data


def largest_segment_area_index(file):
    """
        Determines the index (line number) of the contour with the largest area in the input file.

        Parameters:
        -----------
        file : file object
            An open text file containing contour data.
            Each line should represent a contour in the format:
            <class_id> x1 y1 x2 y2 ... xn yn

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
    for idx, line in enumerate(file, start=0):
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


def test_module():
    create_mesh('5.txt', 7, 1.3, True, isExportingToFemm=True, export_filename="tmp.txt")


if __name__ == "__main__":
    test_module()
