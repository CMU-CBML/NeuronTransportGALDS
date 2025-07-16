import h5py
import numpy as np
import torch
from scripts.GNN_models import edge_index_circle, edge_index_bifurcation
from torch_geometric.data import Data
from torch.utils.data import Dataset
from torch_geometric.data import Batch

def readHDF5(file_name, key=None, dtype = np.float32):
    """read the .h5 file and return the data

    Args:
        file_name (string): file path of the .h5 file
        key (string, optional): key name to read column values. Defaults to None.
        dtype (dtype, optional): data type of the return np.array. Defaults to np.float32.

    Returns:
        dict or np.array(): return the dict contain np.array if key is None, otherwise return np.array
    """
    with h5py.File(file_name, 'r') as f:
        # return everything if key is not specified
        if key is None:
            data = {key: np.array(f[key], dtype=dtype) for key in f.keys()}
            return data
        else:
            return np.array(f[key], dtype = dtype).T

def data_sep_circle(data_dict):
    """Return the data separated by surfaces when the data only contain circle faces (pipe geometry)
    It is separated by every 17 data
    Args:
        data_dict (dict of np.array): the data from readHDH5 function

    Returns:
        list of dict: the separated data by surfaces
    """
    keys = list(data_dict.keys())  # Get all keys
    num_rows = len(next(iter(data_dict.values())))  # Get the number of rows (same for all keys)
    separated_data = []
    for i in range(num_rows // 17):
        chunk = {key: data_dict[key][i * 17:(i + 1) * 17] for key in keys}
        separated_data.append(chunk)

    return separated_data

def data_sep_bifurcation(data_dict, with_series=False):
    """Return the data separated by surfaces and bifurcations when the data contain both circle and bifurcation faces
    It is separated by every 17 data for surfaces and 23 data for bifurcations

    Args:
        data_dict (dict of np.array): the data from readHDH5 function

    Returns:
        list: list of dict for surfaces, if the data does not contain surfaces, return empty list
        list: list of dict for bifurcations, if the data does not contain bifurcation, return empty list
        list: list of data, surface data = 0, bifurcation data = 1. This is for visualization purpose
    """
    if "vx_value" in data_dict.keys():
        center_points_x = data_dict["center_x"]
        center_points_y = data_dict["center_y"]
        center_points_z = data_dict["center_z"]
    elif "n0_value" in data_dict.keys():
        center_points_x = data_dict["center_x"]
        center_points_y = data_dict["center_y"]
        center_points_z = data_dict["center_z"]
    elif "kplus_value" in data_dict.keys():
        center_points_x = data_dict["center_x"]
        center_points_y = data_dict["center_y"]
        center_points_z = data_dict["center_z"]
    else:
        raise ValueError("This file is not from Navier-Stokes, Transport or Kplus data extraction")

    data_len = center_points_x.shape[0]
    cnt = 1
    chunk_start_index = [0]
    old_data_x, old_data_y, old_data_z = center_points_x[0], center_points_y[0], center_points_z[0]
    for i in range(1, data_len):
        if center_points_x[i] == old_data_x and center_points_y[i] == old_data_y and center_points_z[i] == old_data_z:
            cnt += 1
        else:
            chunk_start_index.append(i)
            old_data_x, old_data_y, old_data_z = center_points_x[i], center_points_y[i], center_points_z[i]
            cnt += 1
    chunk_start_index.append(data_len)
    # return two list of np arrays
    # Data series 0 as surfaces, 1 as bifurcations
    data_surfaces = []
    data_bifurcations = []
    data_series = []
    for i in range(len(chunk_start_index)-1):
        if chunk_start_index[i+1] - chunk_start_index[i] == 17:
            data_surfaces.append({key: data_dict[key][chunk_start_index[i]:chunk_start_index[i+1]] for key in data_dict.keys()})
            data_series.append(0)
        elif chunk_start_index[i+1] - chunk_start_index[i] == 23:
            data_bifurcations.append({key: data_dict[key][chunk_start_index[i]:chunk_start_index[i+1]] for key in data_dict.keys()})
            data_series.append(1)
        else:
            print(chunk_start_index[i+1])
            print(chunk_start_index[i])
            raise ValueError("Unknown data")
    if with_series:
        return data_surfaces, data_bifurcations, data_series
    else:
        return data_surfaces, data_bifurcations


def file_to_graph_data_ls(fname, with_series=False, data_key="n0_value"):
    """ Create list of Data objects from single .h5 file to be used in DataLoader

    Args:
        fname (string): file name for the .h5 file
        with_series (bool): if True, return the data series as well
        data_key (string): key name to read column values (for transport simulation). Defaults to "n0_value"
    Returns:
        list: list of graph data in Data format, can be used to create data loader (circle)
        list: list of graph data in Data format, can be used to create data loader (bifurcation)
        list: list of data series, 0 as surfaces, 1 as bifurcations. This is for mainly visualization purpose.
    """
    data_dict = readHDF5(fname, None)
    if fname.split("/")[-2] == "wholeTree":
        if with_series:
            data_surfaces, data_bifurcations, data_series = data_sep_bifurcation(data_dict, with_series=True)
        else:
            data_surfaces, data_bifurcations = data_sep_bifurcation(data_dict)
    else:
        if with_series:
            data_surfaces, data_bifurcations, data_series = data_sep_bifurcation(data_dict, with_series=True)
        else:
            data_surfaces, data_bifurcations = data_sep_bifurcation(data_dict)
    indices_to_remove = []
    for i, data_surface_item in enumerate(data_surfaces):  
        for key, value in data_surface_item.items():
            if isinstance(value, np.ndarray):  # Ensure the value is a NumPy array
                if np.any(np.isnan(value)) or np.any(np.isinf(value)):  # Check all NaN or Inf
                    indices_to_remove.append(i)
                    break  # Stop checking other keys once the item is marked for removal
    for i in sorted(indices_to_remove, reverse=True):
        print("Removing surface data")
        del data_surfaces[i]
    
    indices_to_remove = []
    for i, data_bifurcation_item in enumerate(data_bifurcations):
        for key, value in data_bifurcation_item.items():
            if isinstance(value, np.ndarray):
                if np.any(np.isnan(value)) or np.any(np.isinf(value)):
                    indices_to_remove.append(i)
                    break
    for i in sorted(indices_to_remove, reverse=True):
        print("Removing bifurcation data")
        del data_bifurcations[i]
        
    if fname.split("/")[-1] == "wholeTree_ns.h5":
        circle_data_ls = []
        for i in range(len(data_surfaces)):
            data_vx = data_surfaces[i]["vx_value"]
            data_vy = data_surfaces[i]["vy_value"]
            data_vz = data_surfaces[i]["vz_value"]
            node_feature_tmp = np.hstack((data_vx.reshape(-1,1), data_vy.reshape(-1,1), data_vz.reshape(-1,1)))
            node_feature_tmp = torch.tensor(node_feature_tmp, dtype=torch.float32)
            edge_index_tmp = edge_index_circle
            graph_data_tmp = Data(x=node_feature_tmp, edge_index=edge_index_tmp)
            circle_data_ls.append(graph_data_tmp)
        
        bifurcation_data_ls = []
        # Deal with bifurcation data
        for i in range(len(data_bifurcations)):
            data_vx = data_bifurcations[i]["vx_value"]
            data_vy = data_bifurcations[i]["vy_value"]
            data_vz = data_bifurcations[i]["vz_value"]
            node_feature_tmp = np.hstack((data_vx.reshape(-1,1), data_vy.reshape(-1,1), data_vz.reshape(-1,1)))
            node_feature_tmp = torch.tensor(node_feature_tmp, dtype=torch.float32)
            edge_index_tmp = edge_index_bifurcation
            graph_data_tmp = Data(x=node_feature_tmp, edge_index=edge_index_tmp)
            bifurcation_data_ls.append(graph_data_tmp)
    else:
        circle_data_ls = []
        for i in range(len(data_surfaces)):
            # print(data_surfaces[i].keys())
            # data_n = data_surfaces[i]["n_value"]
            if data_key == "mixed":
                data_n = data_surfaces[i]["n0_value"] + data_surfaces[i]["nplus_value"]
            else:
                data_n = data_surfaces[i][data_key]
            node_feature_tmp = data_n.reshape(-1,1)
            node_feature_tmp = torch.tensor(node_feature_tmp, dtype=torch.float32)
            edge_index_tmp = edge_index_circle
            graph_data_tmp = Data(x=node_feature_tmp, edge_index=edge_index_tmp)
            circle_data_ls.append(graph_data_tmp)
        
        bifurcation_data_ls = []
        # Deal with bifurcation data
        for i in range(len(data_bifurcations)):
            # data_n = data_bifurcations[i]["n_value"]
            if data_key == "mixed":
                data_n = data_bifurcations[i]["n0_value"] + data_bifurcations[i]["nplus_value"]
            else:
                data_n = data_bifurcations[i][data_key]
            node_feature_tmp = data_n.reshape(-1,1)
            node_feature_tmp = torch.tensor(node_feature_tmp, dtype=torch.float32)
            edge_index_tmp = edge_index_bifurcation
            graph_data_tmp = Data(x=node_feature_tmp, edge_index=edge_index_tmp)
            bifurcation_data_ls.append(graph_data_tmp)
    
    if with_series:
        return circle_data_ls, bifurcation_data_ls, data_series
    else:
        return circle_data_ls, bifurcation_data_ls


def Two_file_to_graph_data(fname1, fname2, fname_ns, fname_k, data_key="n0_value"):
    """ Create Data object, Data contains
    [u : Data(x, edge_index, batch),
    n_{t=n} : Data(x, edge_index, batch),
    n_{t=n+1} : Data(x, edge_index, batch),
    index : list of index,
    parent : list of parent
    ]
    
    Args:
        fname1 (string): file name for t=n .h5 file
        fname2 (string): file name for t=n+1 .h5 file
        fname_ns (string): file name for navier-stokes .h5 file
        fname_k (string): file name for k .h5 file
    Returns:
        Data (Data): A list of lists
    """
    data_dict1 = readHDF5(fname1, None)
    data_dict2 = readHDF5(fname2, None)
    data_dictns = readHDF5(fname_ns, None)
    data_diectk = readHDF5(fname_k, None)
    # Initialize the returned lists
    u_ls_circle = []
    u_ls_bifurcation = []
    kplus_ls_circle = []
    kplus_ls_bifurcation = []
    kprimeplus_ls_circle = []
    kprimeplus_ls_bifurcation = []
    n_t1_ls_circle = []
    n_t1_ls_bifurcation = []
    n_t2_ls_circle = []
    n_t2_ls_bifurcation = []
    index_ls_circle = []
    index_ls_bifurcation = []
    parent_ls_circle = []
    parent_ls_bifurcation = []
    # Extract Navier-stokes result
    data_surfaces, data_bifurcations = data_sep_bifurcation(data_dictns)
    for data_surface_dict in data_surfaces:
        data_vx = data_surface_dict["vx_value"]
        data_vy = data_surface_dict["vy_value"]
        data_vz = data_surface_dict["vz_value"]
        if np.isinf(data_vx).any() or np.isinf(data_vy).any() or np.isinf(data_vz).any():
            raise ValueError("Inf value found in Navier-Stokes data")
        node_feature_tmp = np.hstack((data_vx.reshape(-1,1), data_vy.reshape(-1,1), data_vz.reshape(-1,1)))
        node_feature_tmp = torch.tensor(node_feature_tmp, dtype=torch.float32)
        edge_index_tmp = edge_index_circle
        graph_data_tmp = Data(x=node_feature_tmp, edge_index=edge_index_tmp)
        u_ls_circle.append(graph_data_tmp)
        index_ls_circle.append(data_surface_dict["data_index"][0])
        parent_ls_circle.append(data_surface_dict["data_parent"][0])
    for data_bifurcation_dict in data_bifurcations:
        data_vx = data_bifurcation_dict["vx_value"]
        data_vy = data_bifurcation_dict["vy_value"]
        data_vz = data_bifurcation_dict["vz_value"]
        if np.isinf(data_vx).any() or np.isinf(data_vy).any() or np.isinf(data_vz).any():
            raise ValueError("Inf value found in Navier-Stokes data")
        node_feature_tmp = np.hstack((data_vx.reshape(-1,1), data_vy.reshape(-1,1), data_vz.reshape(-1,1)))
        node_feature_tmp = torch.tensor(node_feature_tmp, dtype=torch.float32)
        edge_index_tmp = edge_index_bifurcation
        graph_data_tmp = Data(x=node_feature_tmp, edge_index=edge_index_tmp)
        u_ls_bifurcation.append(graph_data_tmp)
        index_ls_bifurcation.append(data_bifurcation_dict["data_index"][0])
        parent_ls_bifurcation.append(data_bifurcation_dict["data_parent"][0])
    # Extract t=n data
    data_surfaces, data_bifurcations = data_sep_bifurcation(data_dict1)
    for data_surface_dict in data_surfaces:
        # data_n = data_surface_dict["n_value"]
        if data_key == "mixed":
            data_n = data_surface_dict["n0_value"] + data_surface_dict["nplus_value"]
        else:
            data_n = data_surface_dict[data_key]
        node_feature_tmp = data_n.reshape(-1,1)
        node_feature_tmp = torch.tensor(node_feature_tmp, dtype=torch.float32)
        edge_index_tmp = edge_index_circle
        graph_data_tmp = Data(x=node_feature_tmp, edge_index=edge_index_tmp)
        n_t1_ls_circle.append(graph_data_tmp)
    for data_bifurcation_dict in data_bifurcations:
        # data_n = data_bifurcation_dict["n_value"]
        if data_key == "mixed":
            data_n = data_bifurcation_dict["n0_value"] + data_bifurcation_dict["nplus_value"]
        else:
            data_n = data_bifurcation_dict[data_key]
        node_feature_tmp = data_n.reshape(-1,1)
        node_feature_tmp = torch.tensor(node_feature_tmp, dtype=torch.float32)
        edge_index_tmp = edge_index_bifurcation
        graph_data_tmp = Data(x=node_feature_tmp, edge_index=edge_index_tmp)
        n_t1_ls_bifurcation.append(graph_data_tmp)
    # Extract t=n+1 data
    data_surfaces, data_bifurcations = data_sep_bifurcation(data_dict2)
    for data_surface_dict in data_surfaces:
        # data_n = data_surface_dict["n_value"]
        if data_key == "mixed":
            data_n = data_surface_dict["n0_value"] + data_surface_dict["nplus_value"]
        else:
            data_n = data_surface_dict[data_key]
        node_feature_tmp = data_n.reshape(-1,1)
        node_feature_tmp = torch.tensor(node_feature_tmp, dtype=torch.float32)
        edge_index_tmp = edge_index_circle
        graph_data_tmp = Data(x=node_feature_tmp, edge_index=edge_index_tmp)
        n_t2_ls_circle.append(graph_data_tmp)
    for data_bifurcation_dict in data_bifurcations:
        # data_n = data_bifurcation_dict["n_value"]
        if data_key == "mixed":
            data_n = data_bifurcation_dict["n0_value"] + data_bifurcation_dict["nplus_value"]
        else:
            data_n = data_bifurcation_dict[data_key]
        node_feature_tmp = data_n.reshape(-1,1)
        node_feature_tmp = torch.tensor(node_feature_tmp, dtype=torch.float32)
        edge_index_tmp = edge_index_bifurcation
        graph_data_tmp = Data(x=node_feature_tmp, edge_index=edge_index_tmp)
        n_t2_ls_bifurcation.append(graph_data_tmp)
    data_surfaces, data_bifurcations = data_sep_bifurcation(data_diectk)
    for data_surface_dict in data_surfaces:
        data_kplus = data_surface_dict["kplus_value"]
        node_feature_tmp = data_kplus.reshape(-1,1)
        node_feature_tmp = torch.tensor(node_feature_tmp, dtype=torch.float32)
        edge_index_tmp = edge_index_circle
        graph_data_tmp = Data(x=node_feature_tmp, edge_index=edge_index_tmp)
        kplus_ls_circle.append(graph_data_tmp)
        data_kprimeplus = data_surface_dict["kprimeplus_value"]
        node_feature_tmp = data_kprimeplus.reshape(-1,1)
        node_feature_tmp = torch.tensor(node_feature_tmp, dtype=torch.float32)
        edge_index_tmp = edge_index_circle
        graph_data_tmp = Data(x=node_feature_tmp, edge_index=edge_index_tmp)
        kprimeplus_ls_circle.append(graph_data_tmp)

    for data_bifurcation_dict in data_bifurcations:
        data_kplus = data_bifurcation_dict["kplus_value"]
        node_feature_tmp = data_kplus.reshape(-1,1)
        node_feature_tmp = torch.tensor(node_feature_tmp, dtype=torch.float32)
        edge_index_tmp = edge_index_circle
        graph_data_tmp = Data(x=node_feature_tmp, edge_index=edge_index_tmp)
        kplus_ls_bifurcation.append(graph_data_tmp)
        data_kprimeplus = data_bifurcation_dict["kprimeplus_value"]
        node_feature_tmp = data_kprimeplus.reshape(-1,1)
        node_feature_tmp = torch.tensor(node_feature_tmp, dtype=torch.float32)
        edge_index_tmp = edge_index_circle
        graph_data_tmp = Data(x=node_feature_tmp, edge_index=edge_index_tmp)
        kprimeplus_ls_bifurcation.append(graph_data_tmp)
    
    full_batch_u_circle = Batch.from_data_list(u_ls_circle) if len(u_ls_circle) > 0 else None
    full_batch_u_bifurcation = Batch.from_data_list(u_ls_bifurcation) if len(u_ls_bifurcation) > 0 else None
    full_batch_n_t1_circle = Batch.from_data_list(n_t1_ls_circle) if len(n_t1_ls_circle) > 0 else None
    full_batch_n_t1_bifurcation = Batch.from_data_list(n_t1_ls_bifurcation) if len(n_t1_ls_bifurcation) > 0 else None
    full_batch_n_t2_circle = Batch.from_data_list(n_t2_ls_circle) if len(n_t2_ls_circle) > 0 else None
    full_batch_n_t2_bifurcation = Batch.from_data_list(n_t2_ls_bifurcation) if len(n_t2_ls_bifurcation) > 0 else None
    full_batch_kplus_circle = Batch.from_data_list(kplus_ls_circle) if len(kplus_ls_circle) > 0 else None
    full_batch_kplus_bifurcation = Batch.from_data_list(kplus_ls_bifurcation) if len(kplus_ls_bifurcation) > 0 else None
    full_batch_kprimeplus_circle = Batch.from_data_list(kprimeplus_ls_circle) if len(kprimeplus_ls_circle) > 0 else None
    full_batch_kprimeplus_bifurcation = Batch.from_data_list(kprimeplus_ls_bifurcation) if len(kprimeplus_ls_bifurcation) > 0 else None

    return [full_batch_u_circle, full_batch_kplus_circle, full_batch_kprimeplus_circle, full_batch_n_t1_circle, full_batch_n_t2_circle, index_ls_circle, parent_ls_circle], [full_batch_u_bifurcation, full_batch_kplus_bifurcation, full_batch_kprimeplus_bifurcation, full_batch_n_t1_bifurcation, full_batch_n_t2_bifurcation, index_ls_bifurcation, parent_ls_bifurcation]

class CustomDataset(Dataset):
    def __init__(self, data_list_1, data_list_2):
        """
        Args:
        - data_list_1: List of lists, each containing `Data()` objects.
        """
        self.data_list_1 = data_list_1
        self.data_list_2 = data_list_2
        self.length = min(len(data_list_1[0]), len(data_list_2[0]))  # Assume equal lengths for all lists

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        # Extract the corresponding Data() objects from each inner list
        data1 = [self.data_list_1[i][idx] for i in range(len(self.data_list_1))]
        data2 = [self.data_list_2[i][idx] for i in range(len(self.data_list_2))]
        return data1, data2

class CustomDatasetWrapper(Dataset):
    def __init__(self, dataset_list):
        """
        Args:
        - dataset_list: List of CustomDataset instances.
        """
        self.dataset_list = dataset_list

    def __len__(self):
        return len(self.dataset_list)

    def __getitem__(self, idx):
        # Simply return the CustomDataset at the given index
        return self.dataset_list[idx]

def get_index_coordinate(fname):
    data_dict = readHDF5(fname, None)
    center_points_x = data_dict["center_x"]
    center_points_y = data_dict["center_y"]
    center_points_z = data_dict["center_z"]
    indexs = data_dict["data_index"]
    all_data = np.vstack((indexs, center_points_x, center_points_y, center_points_z)).T
    all_data = all_data[np.argsort(all_data[:, 0])]
    unique_data = np.unique(all_data, axis=0)
    return unique_data

def get_edge_distance(pos, edge_index, index_circle, index_bifurcation):
    pos = torch.tensor(pos[:, 1:]).to("cuda")
    row, col = edge_index
    new_index = torch.hstack((index_circle, index_bifurcation))
    row_new = new_index[row]
    col_new = new_index[col]
    edge_vec = pos[row_new - 1] - pos[col_new -1]
    edge_distance = torch.norm(edge_vec, dim=1, keepdim=True)
    return edge_distance