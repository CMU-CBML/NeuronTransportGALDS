import pyvista as pv
import numpy as np
from collections import defaultdict
from create_cell import create_cell_p2p, create_cell_p2b, create_cell_b2p
import meshio
from scipy.interpolate import griddata
from coarse_refine import coarse_refine_create_points
import argparse

def load_swc_with_types(filename):
    coord_map = {}
    parent_map = {}
    children_map = defaultdict(list)
    flag = True
    with open(filename, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith('#') or line == '':
                continue
            if flag is True:
                start_coord = line.split()[2:5]
                start_x, start_y, start_z = map(float, start_coord)
                flag = False
            parts = line.split()
            node_id = int(parts[0])
            x, y, z = map(float, parts[2:5])
            x = x - start_x
            y = y - start_y
            z = z - start_z
            parent = int(parts[6])
            coord_map[node_id] = (x, y, z)
            parent_map[node_id] = parent
            children_map[parent].append(node_id)

    # Classify each node
    node_type = {}
    for node_id in coord_map:
        num_children = len(children_map.get(node_id, []))
        if num_children == 2:
            node_type[node_id] = "bifurcation"
        elif num_children == 1 or num_children == 0:
            node_type[node_id] = "pipe"
        else:
            raise ValueError(f"Node {node_id} has an unexpected number of children: {num_children}")

    # Get all elements (child-parent pairs)
    elements = []
    for child_id, parent_id in parent_map.items():
        if parent_id == -1:
            continue  # Skip root
        child_info = (coord_map[child_id], node_type[child_id])
        parent_info = (coord_map[parent_id], node_type[parent_id])
        elements.append((child_info, parent_info))

    return elements

def vtu_to_vtk(vtu_field_file, swc_file, vtk_control_mesh, output_fine_file, output_coarse_file="output_coarse.vtk", template_vtu=None):
    print("Creating coarse mesh...")
    # read vtu file
    if template_vtu is not None:
        mesh = pv.read(template_vtu)
    else:
        mesh = pv.read(vtu_field_file)
    coordinates = mesh.points
    latent_exact = mesh.point_data["latent_exact"].reshape(-1,1)
    indexs = np.arange(coordinates.shape[0]).reshape(-1,1)
    combinded = np.hstack((indexs, coordinates, latent_exact))
    remove_index = []
    for i in range(combinded.shape[0]):
        if i == 0:
            old_latent = combinded[i, -1]
            continue
        if combinded[i, -1] == old_latent:
        # if np.abs(combinded[i, -1] - old_latent) < 1E-10:
            remove_index.append(i)
        else:
            old_latent = combinded[i, -1]
    
    # create point data
    mesh = pv.read(vtu_field_file)
    point_data = mesh.point_data

    combinded = np.delete(combinded, remove_index, axis=0)
    search_index = combinded[:, :-1]

    elements = load_swc_with_types(swc_file)

    cells = np.array([0.0])
    for element in elements:
        child_xyz, child_type = element[0]
        parent_xyz, parent_type = element[1]
        # Find the closest point in the search_index
        distances = np.linalg.norm(search_index[:, 1:] - child_xyz, axis=1)
        closest_index = np.argmin(distances)
        child_idx = search_index[closest_index][0]
        distances = np.linalg.norm(search_index[:, 1:] - parent_xyz, axis=1)
        closest_index = np.argmin(distances)
        parent_idx = search_index[closest_index][0]
        
        # if it is pipe-pipe
        if parent_type == "pipe" and child_type == "pipe":
            new_points, point_data = coarse_refine_create_points(parent_idx, child_idx, coordinates, point_data, refine_num=1)
            for new_layers in range(int(new_points.shape[0] / 17)):
                coordinates = np.vstack((coordinates, new_points[new_layers * 17: (new_layers + 1) * 17]))
                cell_indices = create_cell_p2p(parent_idx, coordinates.shape[0] - 17, coordinates)
                cells = np.hstack((cells, cell_indices))
                parent_idx = coordinates.shape[0] - 17
            cell_indices = create_cell_p2p(parent_idx, child_idx, coordinates)
            cells = np.hstack((cells, cell_indices))
        
        if parent_type == "pipe" and child_type == "bifurcation":
            new_points, point_data = coarse_refine_create_points(parent_idx, child_idx, coordinates, point_data, refine_num=10)
            for new_layers in range(int(new_points.shape[0] / 17)):
                coordinates = np.vstack((coordinates, new_points[new_layers * 17: (new_layers + 1) * 17]))
                # Create cells for the new points
                cell_indices = create_cell_p2p(parent_idx, coordinates.shape[0] - 17, coordinates)
                cells = np.hstack((cells, cell_indices))
                parent_idx = coordinates.shape[0] - 17  # Update parent index to the last point of the new layer
            
            cell_indices = create_cell_p2b(parent_idx, child_idx, coordinates)
            cells = np.hstack((cells, cell_indices))
            
        if parent_type == "bifurcation" and child_type == "pipe":
            new_points, point_data = coarse_refine_create_points(parent_idx, child_idx, coordinates, point_data, refine_num=10, inverse_interpolation=True)
            for new_layers in range(int(new_points.shape[0] / 17)):
                coordinates = np.vstack((coordinates, new_points[new_layers * 17: (new_layers + 1) * 17]))
                # Create cells for the new points
                if new_layers == 0:
                    cell_indices = create_cell_b2p(parent_idx, coordinates.shape[0] - 17, coordinates)
                else:
                    cell_indices = create_cell_p2p(parent_idx, coordinates.shape[0] - 17, coordinates)
                cells = np.hstack((cells, cell_indices))
                parent_idx = coordinates.shape[0] - 17
            cell_indices = create_cell_p2p(parent_idx, child_idx, coordinates)
            cells = np.hstack((cells, cell_indices))
            
        
    cells = np.delete(cells, 0, axis=0)
    # change cell to int
    cells = cells.astype(int)

    celltypes = np.full(int(cells.shape[0] / 9), 12)

    new_grid = pv.UnstructuredGrid(cells, celltypes, coordinates)

    # mesh = pv.read(vtu_field_file)
    for name in point_data:
        new_grid.point_data[name] = point_data[name]

    new_grid.save(output_coarse_file)

    print("Creating fine mesh...")
    coarse_mesh = meshio.read(output_coarse_file)
    fine_mesh = meshio.read(vtk_control_mesh)

    # Extract points (coordinates)
    points_coarse = coarse_mesh.points
    points_fine = fine_mesh.points

    # Extract field data from the coarse mesh
    # --- Step 2: Prepare output field dictionary ---
    interpolated_fields = {}

    print("Fields found in coarse mesh:")
    for key in coarse_mesh.point_data.keys():
        print(f"  - {key}")
        
        values_coarse = coarse_mesh.point_data[key]

        # Interpolate each component separately for vector fields
        if values_coarse.ndim == 2:  # e.g., shape (N, 3)
            values_fine = np.zeros((len(points_fine), values_coarse.shape[1]))
            for i in range(values_coarse.shape[1]):
                val_linear = griddata(points_coarse, values_coarse[:, i], points_fine, method="linear")
                val_nearest = griddata(points_coarse, values_coarse[:, i], points_fine, method="nearest")
                values_fine[:, i] = np.where(np.isnan(val_linear), val_nearest, val_linear)
                values_fine[:, i] = val_nearest
        else:  # Scalar field, shape (N,)
            val_linear = griddata(points_coarse, values_coarse, points_fine, method="linear")
            val_nearest = griddata(points_coarse, values_coarse, points_fine, method="nearest")
            values_fine = np.where(np.isnan(val_linear), val_nearest, val_linear)
            values_fine = val_nearest

        interpolated_fields[key] = values_fine

    # --- Step 3: Save the interpolated fields on the fine mesh ---
    output_mesh = meshio.Mesh(
        points=points_fine,
        cells=fine_mesh.cells,
        point_data=interpolated_fields
    )

    output_mesh.write(output_fine_file)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", type=str, default="NMO_66748")
    args = parser.parse_args()
    case_name = args.case
    point_cloud_file1 = f"{case_name}/{case_name}_merged.vtu"
    swc_file = f"{case_name}/geometry_data/skeleton_smooth.swc"
    vtk_control_mesh = f"{case_name}/geometry_data/controlmesh.vtk"
    output_fine_file = f"{case_name}/{case_name}_fine_result.vtk"
    output_coarse_file = f"{case_name}/{case_name}_coarse_result.vtk"
    vtu_to_vtk(point_cloud_file1, swc_file, vtk_control_mesh, output_fine_file, output_coarse_file)
    point_cloud_file2 = f"{case_name}/{case_name}_ns.vtu"
    output_fine_file = f"{case_name}/{case_name}_fine_ns_result.vtk"
    output_coarse_file = f"{case_name}/{case_name}_coarse_ns_result.vtk"
    vtu_to_vtk(point_cloud_file2, swc_file, vtk_control_mesh, output_fine_file, output_coarse_file, template_vtu=point_cloud_file1)
    print("VTU to VTK conversion completed.")