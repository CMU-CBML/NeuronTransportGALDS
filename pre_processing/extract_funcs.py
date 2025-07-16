import meshio
import pandas as pd
import numpy as np
import os
import h5py

def read_vtk(vtk_fname):
    mesh = meshio.read(vtk_fname)
    # Extract points
    vtk_points = mesh.points
    # Extract point data (scalars/vectors)
    point_values_dict = {}
    for i, (key, value) in enumerate(mesh.point_data.items()):
        point_values_dict[key] = value
    keys = list(point_values_dict.keys())
    return vtk_points, point_values_dict, keys

def read_swc(swc_fname, whole_skeleton_path=None):
    if not os.path.exists(swc_fname):
        raise ValueError("Warning: swc file not found")
    swc_data = pd.read_csv(
        swc_fname,
        sep=r'\s+',
        comment='#',  # Skip comment lines
        header=None,  # No header in the file
        names=['ID', 'Type', 'X', 'Y', 'Z', 'Radius', 'Parent']  # Assign column names
    )
    swc_points = swc_data[['ID', 'X', 'Y', 'Z', 'Parent']].to_numpy()
    swc_points[:, 1:4] = swc_points[:, 1:4] - swc_points[0, 1:4]
    return swc_points

def get_tol(swc_fname, alpha=1E-3):
    swc_data = pd.read_csv(
        swc_fname,
        sep=r'\s+',
        comment='#',  # Skip comment lines
        header=None,  # No header in the file
        names=['ID', 'Type', 'X', 'Y', 'Z', 'Radius', 'Parent']  # Assign column names
    )
    swc_points = swc_data[['X', 'Y', 'Z']].values
    tol = np.max(np.abs(swc_points[1:] - swc_points[:-1])) * alpha
    return tol

point_selection_bias = [0, 1, 2, 3, 7, 13, 20, 27, 35, 62, 96, 108, 112, 119, 128, 155, 189]
point_selection_bifurcation_bias = [0, 1, 2, 3, 7, 13, 20, 27, 35, 62, 96, 108, 112, 119, 128, 155, 189, 201, 205, 212, 221, 248, 282]

def extract_data_vp(vtk_fname, skeleton_fname, save_file_name):
    vtk_points, point_values_dict, keys = read_vtk(vtk_fname)
    swc_points = read_swc(skeleton_fname)
    tol = get_tol(skeleton_fname, alpha=1E-3)
    ls_x, ls_y, ls_z = [], [], []
    ls_vx, ls_vy, ls_vz = [], [], []
    ls_p = []
    ls_center_x, ls_center_y, ls_center_z = [], [], []
    ls_index, ls_parent = [], []
    # for idx, row in enumerate(tqdm(swc_points)):
    for idx, row in enumerate(swc_points):
        index = row[0]
        parent = row[4]
        # Check if the point is a bifurcation point
        cnt = 0
        for points in swc_points:
            if row[0] == points[4]:
                cnt += 1
        if cnt >= 2:
            # Then (row) is a bifurcation point
            match_idxs = np.where((np.abs(vtk_points - row[1:4]) <tol).all(axis=1))[0]
            if len(match_idxs) > 1:
                raise ValueError("Warning: multiple matches found")
            elif len(match_idxs) == 0:
                raise ValueError("Warning: no match found")
            else:
                for bias in point_selection_bifurcation_bias:
                    ls_x.append(vtk_points[match_idxs[0] + bias][0])
                    ls_y.append(vtk_points[match_idxs[0] + bias][1])
                    ls_z.append(vtk_points[match_idxs[0] + bias][2])
                    ls_vx.append(point_values_dict[keys[0]][match_idxs[0] + bias][0])
                    ls_vy.append(point_values_dict[keys[0]][match_idxs[0] + bias][1])
                    ls_vz.append(point_values_dict[keys[0]][match_idxs[0] + bias][2])
                    ls_p.append(point_values_dict[keys[2]][match_idxs[0] + bias])
                    ls_center_x.append(row[1])
                    ls_center_y.append(row[2])
                    ls_center_z.append(row[3])
                    ls_index.append(index)
                    ls_parent.append(parent)
        else:
            # This (row) is parent of a single child
            match_idxs = np.where((np.abs(vtk_points - row[1:4]) <tol).all(axis=1))[0]
            if len(match_idxs) > 1:
                raise ValueError("Warning: multiple matches found")
            elif len(match_idxs) == 0:
                print(row[1:4], flush=True)
                raise ValueError("Warning: no match found")
            else:
                for bias in point_selection_bias:
                    ls_x.append(vtk_points[match_idxs[0] + bias][0])
                    ls_y.append(vtk_points[match_idxs[0] + bias][1])
                    ls_z.append(vtk_points[match_idxs[0] + bias][2])
                    ls_vx.append(point_values_dict[keys[0]][match_idxs[0] + bias][0])
                    ls_vy.append(point_values_dict[keys[0]][match_idxs[0] + bias][1])
                    ls_vz.append(point_values_dict[keys[0]][match_idxs[0] + bias][2])
                    ls_p.append(point_values_dict[keys[2]][match_idxs[0] + bias])
                    ls_center_x.append(row[1])
                    ls_center_y.append(row[2])
                    ls_center_z.append(row[3])
                    ls_index.append(index)
                    ls_parent.append(parent)
    
    data_x = np.array(ls_x)
    data_y = np.array(ls_y)
    data_z = np.array(ls_z)
    vx_value = np.array(ls_vx)
    vy_value = np.array(ls_vy)
    vz_value = np.array(ls_vz)
    center_x = np.array(ls_center_x)
    center_y = np.array(ls_center_y)
    center_z = np.array(ls_center_z)
    data_index = np.array(ls_index)
    data_parent = np.array(ls_parent)

    with h5py.File(f'{save_file_name}.h5', 'w') as f:
        f.create_dataset('data_x', data=data_x)
        f.create_dataset('data_y', data=data_y)
        f.create_dataset('data_z', data=data_z)
        f.create_dataset('vx_value', data=vx_value)
        f.create_dataset('vy_value', data=vy_value)
        f.create_dataset('vz_value', data=vz_value)
        f.create_dataset('center_x', data=center_x)
        f.create_dataset('center_y', data=center_y)
        f.create_dataset('center_z', data=center_z)
        f.create_dataset('data_index', data=data_index)
        f.create_dataset('data_parent', data=data_parent)


def extract_data_n(vtk_fname, skeleton_fname, save_file_name):
    vtk_points, point_values_dict, keys = read_vtk(vtk_fname)
    swc_points = read_swc(skeleton_fname)
    tol = get_tol(skeleton_fname, alpha=1E-3)
    ls_x, ls_y, ls_z = [], [], []
    ls_n0 = []
    ls_nplus = []
    ls_center_x, ls_center_y, ls_center_z = [], [], []
    ls_index, ls_parent = [], []
    # for idx, row in enumerate(tqdm(swc_points)):
    for idx, row in enumerate(swc_points):
        index = row[0]
        parent = row[4]
        # Check if the point is a bifurcation point
        cnt = 0
        for points in swc_points:
            if row[0] == points[4]:
                cnt += 1
        if cnt >= 2:
            # Then (row) is a bifurcation points
            match_idxs = np.where((np.abs(vtk_points - row[1:4]) <tol).all(axis=1))[0]
            if len(match_idxs) > 1:
                raise ValueError("Warning: multiple matches found")
            elif len(match_idxs) == 0:
                raise ValueError("Warning: no match found")
            else:
                for bias in point_selection_bifurcation_bias:
                    ls_x.append(vtk_points[match_idxs[0] + bias][0])
                    ls_y.append(vtk_points[match_idxs[0] + bias][1])
                    ls_z.append(vtk_points[match_idxs[0] + bias][2])
                    ls_n0.append(point_values_dict["N0"][match_idxs[0] + bias])
                    ls_nplus.append(point_values_dict["N_plus"][match_idxs[0] + bias])
                    ls_center_x.append(row[1])
                    ls_center_y.append(row[2])
                    ls_center_z.append(row[3])
                    ls_index.append(index)
                    ls_parent.append(parent)
        else:
            # This (row) is parent of a single child or a end point
            match_idxs = np.where((np.abs(vtk_points - row[1:4]) <tol).all(axis=1))[0]
            if len(match_idxs) > 1:
                raise ValueError("Warning: multiple matches found")
            elif len(match_idxs) == 0:
                raise ValueError("Warning: no match found")
            else:
                for bias in point_selection_bias:
                    ls_x.append(vtk_points[match_idxs[0] + bias][0])
                    ls_y.append(vtk_points[match_idxs[0] + bias][1])
                    ls_z.append(vtk_points[match_idxs[0] + bias][2])
                    ls_n0.append(point_values_dict["N0"][match_idxs[0] + bias])
                    ls_nplus.append(point_values_dict["N_plus"][match_idxs[0] + bias])
                    ls_center_x.append(row[1])
                    ls_center_y.append(row[2])
                    ls_center_z.append(row[3])
                    ls_index.append(index)
                    ls_parent.append(parent)

    data_x = np.array(ls_x)
    data_y = np.array(ls_y)
    data_z = np.array(ls_z)
    n0_value = np.array(ls_n0).squeeze()
    nplus_value = np.array(ls_nplus).squeeze()
    center_x = np.array(ls_center_x)
    center_y = np.array(ls_center_y)
    center_z = np.array(ls_center_z)
    data_index = np.array(ls_index)
    data_parent = np.array(ls_parent)

    with h5py.File(f'{save_file_name}.h5', 'w') as f:
        f.create_dataset('data_x', data=data_x)
        f.create_dataset('data_y', data=data_y)
        f.create_dataset('data_z', data=data_z)
        f.create_dataset('n0_value', data=n0_value)
        f.create_dataset('nplus_value', data=nplus_value)
        f.create_dataset('center_x', data=center_x)
        f.create_dataset('center_y', data=center_y)
        f.create_dataset('center_z', data=center_z)
        f.create_dataset('data_index', data=data_index)
        f.create_dataset('data_parent', data=data_parent)

def extract_data_k(vtk_fname, skeleton_fname, save_file_name):
    vtk_points, point_values_dict, keys = read_vtk(vtk_fname)
    swc_points = read_swc(skeleton_fname)
    tol = get_tol(skeleton_fname, alpha=1E-5)
    ls_x, ls_y, ls_z = [], [], []
    ls_kplus = []
    ls_kprimeplus = []
    ls_center_x, ls_center_y, ls_center_z = [], [], []
    ls_index, ls_parent = [], []
    # for idx, row in enumerate(tqdm(swc_points)):
    for idx, row in enumerate(swc_points):
        index = row[0]
        parent = row[4]
        # Check if the point is a bifurcation point
        cnt = 0
        for points in swc_points:
            if row[0] == points[4]:
                cnt += 1
        if cnt >= 2:
            # Then (row) is a bifurcation points
            match_idxs = np.where((np.abs(vtk_points - row[1:4]) <tol).all(axis=1))[0]
            if len(match_idxs) > 1:
                raise ValueError("Warning: multiple matches found")
            elif len(match_idxs) == 0:
                raise ValueError("Warning: no match found")
            else:
                for bias in point_selection_bifurcation_bias:
                    ls_x.append(vtk_points[match_idxs[0] + bias][0])
                    ls_y.append(vtk_points[match_idxs[0] + bias][1])
                    ls_z.append(vtk_points[match_idxs[0] + bias][2])
                    ls_kplus.append(point_values_dict["K3_field"][match_idxs[0] + bias])
                    ls_kprimeplus.append(point_values_dict["K5_field"][match_idxs[0] + bias])
                    ls_center_x.append(row[1])
                    ls_center_y.append(row[2])
                    ls_center_z.append(row[3])
                    ls_index.append(index)
                    ls_parent.append(parent)
        else:
            match_idxs = np.where((np.abs(vtk_points - row[1:4]) <tol).all(axis=1))[0]
            if len(match_idxs) > 1:
                raise ValueError("Warning: multiple matches found")
            elif len(match_idxs) == 0:
                print(row[1:4], flush=True)
                raise ValueError("Warning: no match found")
            else:
                for bias in point_selection_bias:
                    ls_x.append(vtk_points[match_idxs[0] + bias][0])
                    ls_y.append(vtk_points[match_idxs[0] + bias][1])
                    ls_z.append(vtk_points[match_idxs[0] + bias][2])
                    ls_kplus.append(point_values_dict["K3_field"][match_idxs[0] + bias])
                    ls_kprimeplus.append(point_values_dict["K5_field"][match_idxs[0] + bias])
                    ls_center_x.append(row[1])
                    ls_center_y.append(row[2])
                    ls_center_z.append(row[3])
                    ls_index.append(index)
                    ls_parent.append(parent)

    data_x = np.array(ls_x)
    data_y = np.array(ls_y)
    data_z = np.array(ls_z)
    kplus_value = np.array(ls_kplus).squeeze()
    kprimeplus_value = np.array(ls_kprimeplus).squeeze()
    center_x = np.array(ls_center_x)
    center_y = np.array(ls_center_y)
    center_z = np.array(ls_center_z)
    data_index = np.array(ls_index)
    data_parent = np.array(ls_parent)
    print("save file name:", save_file_name, flush=True)
    with h5py.File(f'{save_file_name}.h5', 'w') as f:
        f.create_dataset('data_x', data=data_x)
        f.create_dataset('data_y', data=data_y)
        f.create_dataset('data_z', data=data_z)
        f.create_dataset('kplus_value', data=kplus_value)
        f.create_dataset('kprimeplus_value', data=kprimeplus_value)
        f.create_dataset('center_x', data=center_x)
        f.create_dataset('center_y', data=center_y)
        f.create_dataset('center_z', data=center_z)
        f.create_dataset('data_index', data=data_index)
        f.create_dataset('data_parent', data=data_parent)

def main():
    extract_data_vp("NMO_66731/sim_result/sim_0/ns_result/controlmesh_VelocityPressure.vtk", "NMO_66731/skeleton_smooth.swc", "NMO_66731/Data/sim_0/wholeTree_ns")
    extract_data_n("NMO_66731/sim_result/sim_0/tr_result/controlmesh_allparticle_3.vtk", "NMO_66731/skeleton_smooth.swc", "NMO_66731/Data/sim_0/wholeTree_tr_step_3")

if __name__ == "__main__":
    main()