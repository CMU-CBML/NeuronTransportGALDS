import torch
import os
from scripts.GNN_models import GNN_Autoencoder_circle, GNN_Autoencoder_bifurcation, SharedEncoder, GNNODEFunc
import tqdm
from parameter import ENABLE_GPU
from scripts.utils import  Two_file_to_graph_data, readHDF5, file_to_graph_data_ls, get_index_coordinate, get_edge_distance, data_sep_bifurcation
from scripts.visualization import Save_vtk_with_dict
import numpy as np
import pandas as pd
from torchdiffeq import odeint
import pyvista as pv
import time

# Change dir to current dir
current_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(current_dir)

def get_dt(path):
    simulation_parameter_file = os.path.join(path, "simulation_parameter.txt")
    simulation_parameter_file = simulation_parameter_file.replace("\\", "/")
    simulation_parameter_file = pd.read_csv(simulation_parameter_file, sep='\s+', header=None, names=['parameter', 'value'])
    simulation_parameter_file = dict(zip(simulation_parameter_file['parameter'], simulation_parameter_file['value']))
    dt = simulation_parameter_file["dt"]
    return dt

def process_filenames(file_list):
    """rearrange the list of file names into a organized list of file names
    The file has a format of "wholeTree_tr_step_N.h5"

    Args:
        file_list (list): list of file names

    Returns:
        list(list): organized list of file names
    """
    # First scan the file_list to get min and max values of N and M
    N_max = -1
    for fname in file_list:
        N = int(fname.split("/")[-1].split("_")[-1].split(".")[0])
        N_max = max(N, N_max)
    # get current dir
    current_dir = file_list[0].split("/")[:-1]
    current_dir = "/".join(current_dir)
    ret_ls = []
    for j in range(0, N_max+1):
        tmp_fname = f"{current_dir}/wholeTree_tr_step_{j}.h5"
        if tmp_fname in file_list:
            ret_ls.append(tmp_fname)
    return ret_ls

def create_data_loader_NN2(path, batch_size=-1, idx_start=0, idx_end=-1):
    """ Create DataLoader object from list of .h5 files

    Args:
        path (string): path of the /wholeTree folder

    Returns:
        DataLoaders: DataLoader object for all data
    """
    # first get all the file names
    data_files = os.listdir(path)
    data_files = [os.path.join(path, f).replace("\\", "/") for f in data_files]
    data_files = [f for f in data_files if f.endswith(".h5")]
    # Separate the data files into ns and tr (wholeTree_ns.h5 and wholeTree_tr_step_N.h5)
    data_files_ns = [f for f in data_files if "wholeTree_ns" in f]
    data_files_tr = [f for f in data_files if "wholeTree_tr" in f]
    # Rearrange the file names
    file_name_ls = process_filenames(data_files_tr)[idx_start:idx_end]
    # Create DataLoader object
    
    index_coordinate = get_index_coordinate(data_files_ns[0])
    
    data_set_list = []
    data_time_list = []
    spacing = 1
    for idx in range(len(file_name_ls)-spacing):
        data_circle, data_bifurcation = Two_file_to_graph_data(file_name_ls[idx], file_name_ls[idx+spacing], data_files_ns[0], data_key="nplus_value")
        # data_circle, data_bifurcation = Two_file_to_graph_data(file_name_ls[idx], file_name_ls[idx+spacing], data_files_ns[0], data_key="mixed")
        data_set_list.append([data_circle, data_bifurcation])
        # get the file name for the data
        data_time_list.append(file_name_ls[idx].split("/")[-1].split("_")[-1].split(".")[0]) 
    data_loader = data_set_list

    return data_loader, index_coordinate

def create_global_graph(index_circle, index_bifurcation, parent_circle, parent_bifurcation):
    print("Creating global graph")
    def find_children(idx, index, parent):
        children = []
        for i in range(len(parent)):
            if parent[i] == index[idx]:
                children.append(i)
        return children
         
    # Cat all the datas
    index_total = torch.hstack((index_circle, index_bifurcation))
    parent_total = torch.hstack((parent_circle, parent_bifurcation))
    print(f"number of pipes: {len(index_circle)}")
    print(f"number of bifurcations: {len(index_bifurcation)}")

    # Find the start of the global graph
    start_idx = -1
    for idx in range(len(parent_total)):
        if parent_total[idx] == -1:
            start_idx = idx
            break
    queue = [start_idx]
    edge_index = torch.zeros((2, 1))
    distance_from_root = torch.full((len(index_total),), -1, dtype=torch.int64)
    distance_from_root[start_idx] = 0
    for idx in queue:
        childrens = find_children(idx, index_total, parent_total)
        queue += childrens
        for child in childrens:
            # Create edge index
            distance_from_root[child] = distance_from_root[idx] + 1
            edge_index_tmp = torch.tensor([[idx, child], [child, idx]])
            edge_index = torch.hstack((edge_index, edge_index_tmp))
    edge_index = edge_index[:, 1:].to(torch.int64)
    # Create the global graph
    print("Finish creating global graph")
    return edge_index, distance_from_root.reshape(-1,1)

def edge_distance_2_node_physical_distance(edge_index, edge_distance):
    # get index of the edge_distance in odd number 
    edge_distance = edge_distance[0::2]
    edge_index = edge_index.T[0::2]
    edge_distance_pair = torch.hstack((edge_index[:,1].reshape(-1,1), edge_distance))
    top_pair = torch.tensor([[edge_index[0,0], 0.0]]).to("cuda")
    edge_distance_pair = torch.vstack((top_pair, edge_distance_pair))
    node_physical_distance = torch.zeros((edge_distance.shape[0] + 1, 1)).to("cuda")
    for edge_nodes in edge_index:
        begin_node, end_node = edge_nodes
        begin_distnace = node_physical_distance[begin_node]
        mask = edge_distance_pair[:,0] == end_node
        add_distance = edge_distance_pair[mask, 1]
        node_physical_distance[end_node,0] = begin_distnace + add_distance

    return node_physical_distance

def calculate_error_NN2_AD2(model_NN2, AD1_models, AD2_models, error_fns, data_loaders, dt):
    if ENABLE_GPU:
        new_data_loaders = []
        for data_loader in data_loaders:
            # if move all data to gpu
            data_loader_new = []
            cnt = 0
            for data_pipe, data_bif in data_loader:
                u, t, t1, index, parent = data_pipe
                u_bif, t_bif, t1_bif, index_bif, parent_bif = data_bif
                # Check if the data has nan or inf for u, t, t1
                if torch.isnan(u.x).any() or torch.isnan(t.x).any() or torch.isnan(t1.x).any():
                    print("Data has nan for pipe")
                if torch.isinf(u.x).any() or torch.isinf(t.x).any() or torch.isinf(t1.x).any():
                    cnt += 1
                    print("Data has inf for pipe")
                if torch.isnan(u_bif.x).any() or torch.isnan(t_bif.x).any() or torch.isnan(t1_bif.x).any():
                    print("Data has nan for bif")
                if torch.isinf(u_bif.x).any() or torch.isinf(t_bif.x).any() or torch.isinf(t1_bif.x).any():
                    print("Data has inf for bif")
                
                u = u.to("cuda")
                t = t.to("cuda")
                t1 = t1.to("cuda")
                index = torch.tensor(index).to(torch.int64).to("cuda")
                parent = torch.tensor(parent).to(torch.int64).to("cuda")
                u_bif = u_bif.to("cuda")
                t_bif = t_bif.to("cuda")
                t1_bif = t1_bif.to("cuda")
                index_bif = torch.tensor(index_bif).to(torch.int64).to("cuda")
                parent_bif = torch.tensor(parent_bif).to(torch.int64).to("cuda")
                data_loader_new.append([[u, t, t1, index, parent], [u_bif, t_bif, t1_bif, index_bif, parent_bif]])
            data_loader = data_loader_new
            new_data_loaders.append(data_loader)
        data_loaders = new_data_loaders
    
    model_NN2.eval()
    AD1_circle, AD1_bifurcation = AD1_models
    AD2_circle, AD2_bifurcation = AD2_models
    AD1_circle.eval()
    AD1_bifurcation.eval()
    AD2_circle.eval()
    AD2_bifurcation.eval()
    edge_index_ls = []
    distance_to_root_ls = []
    for data_loader in data_loaders:
        batch_circle, batch_bifurcation = data_loader[0]
        _,_,_, index_circle, parent_circle = batch_circle
        _,_,_, index_bifurcation, parent_bifurcation = batch_bifurcation
        edge_index, distance_to_root = create_global_graph(index_circle, index_bifurcation, parent_circle, parent_bifurcation)
        edge_index = edge_index.to("cuda")
        distance_to_root = distance_to_root.to("cuda")
        edge_index_ls.append(edge_index)
        distance_to_root_ls.append(distance_to_root)

    total_samples_latent = 0
    total_samples_physical = 0
    for error_fn in error_fns:
        total_loss_NN2_ls = [0] * len(error_fns)
        total_loss_AD2_ls = [0] * len(error_fns)
    for idx, data_loader in enumerate(data_loaders):
        for idxx in range(len(data_loader)):
            _, full_batch_t_circle, full_batch_t1_circle, index_circle, parent_circle = batch_circle
            _, full_batch_t_bifurcation, full_batch_t1_bifurcation, index_bifurcation, parent_bifurcation = batch_bifurcation
            # ! CHange here
            batch_circle_init, batch_bifurcation_init = data_loader[0]
            _, full_batch_t_circle_init, _, _, _ = batch_circle_init
            _, full_batch_t_bifurcation_init, _, _, _ = batch_bifurcation_init
            n_latent_circle_init = AD2_circle.encode(full_batch_t_circle_init.x, full_batch_t_circle_init.edge_index, full_batch_t_circle_init.batch)
            n_latent_bifurcation_init = AD2_bifurcation.encode(full_batch_t_bifurcation_init.x, full_batch_t_bifurcation_init.edge_index, full_batch_t_bifurcation_init.batch)
            data_init = torch.vstack((n_latent_circle_init, n_latent_bifurcation_init))
            pred_graph = odeint(model_NN2, data_init, torch.tensor([0.0, (idxx+1)*dt]).to("cuda"), method="dopri5")[1]

            n1_latent_circle = AD2_circle.encode(full_batch_t1_circle.x, full_batch_t1_circle.edge_index, full_batch_t1_circle.batch)
            n1_latent_bifurcation = AD2_bifurcation.encode(full_batch_t1_bifurcation.x, full_batch_t1_bifurcation.edge_index, full_batch_t1_bifurcation.batch)
            target = torch.vstack((n1_latent_circle, n1_latent_bifurcation))
            for i, error_fn in enumerate(error_fns):
                # Calculate the loss for NN2
                loss_NN2 = error_fn(pred_graph, target)
                total_loss_NN2_ls[i] += loss_NN2.item() * data_init.size(0)
            num_of_surfaces = int(full_batch_t_circle.x.shape[0] / 17)
            indexs = [i for i in range(num_of_surfaces)]
            latent_result_on_cross_section = pred_graph[indexs]
            physical_result_circle = AD2_circle.decode(latent_result_on_cross_section, full_batch_t_circle.edge_index, full_batch_t_circle.batch)
            num_of_bifurcations = int(full_batch_t_bifurcation.x.shape[0] / 23)
            indexs = [i + num_of_surfaces for i in range(num_of_bifurcations)]
            latent_result_on_cross_section = pred_graph[indexs]
            physical_result_bifurcation = AD2_bifurcation.decode(latent_result_on_cross_section, full_batch_t_bifurcation.edge_index, full_batch_t_bifurcation.batch)
            target_physical_circle = full_batch_t1_circle.x.view(-1, 17, 1)
            target_physical_bifurcation = full_batch_t1_bifurcation.x.view(-1, 23, 1)
            for i, error_fn in enumerate(error_fns):
                # Calculate the loss for NN2 + AD2
                loss_AD2 = error_fn(physical_result_circle, target_physical_circle) * full_batch_t1_circle.x.shape[0]
                total_loss_AD2_ls[i] += loss_AD2.item()
                loss_AD2 = error_fn(physical_result_bifurcation, target_physical_bifurcation) * full_batch_t1_bifurcation.x.shape[0]
                total_loss_AD2_ls[i] += loss_AD2.item()
            total_samples_latent += data_init.shape[0]
            total_samples_physical += (full_batch_t_circle.x.shape[0] + full_batch_t_bifurcation.x.shape[0])
    for i, error_fn in enumerate(error_fns):
        total_loss_NN2_ls[i] /= total_samples_latent
        total_loss_AD2_ls[i] /= total_samples_physical
    
    return total_loss_NN2_ls, total_loss_AD2_ls

def vtk_NN2_AD2(model_NN2, AD1_models, AD2_models, data_loader, data_path, fname, dt, index_coordinate, save_only_last=False):
    print("Creating vtk file")
    if ENABLE_GPU:
        # if move all data to gpu
        data_loader_new = []
        for data_pipe, data_bif in data_loader:
            u, t, t1, index, parent = data_pipe
            u_bif, t_bif, t1_bif, index_bif, parent_bif = data_bif
            u = u.to("cuda")
            t = t.to("cuda")
            t1 = t1.to("cuda")
            index = torch.tensor(index).to(torch.int64).to("cuda")
            parent = torch.tensor(parent).to(torch.int64).to("cuda")
            u_bif = u_bif.to("cuda")
            t_bif = t_bif.to("cuda")
            t1_bif = t1_bif.to("cuda")
            index_bif = torch.tensor(index_bif).to(torch.int64).to("cuda")
            parent_bif = torch.tensor(parent_bif).to(torch.int64).to("cuda")
            data_loader_new.append([[u, t, t1, index, parent], [u_bif, t_bif, t1_bif, index_bif, parent_bif]])
        data_loader = data_loader_new
    
    model_NN2.eval()
    AD1_circle, AD1_bifurcation = AD1_models
    AD2_circle, AD2_bifurcation = AD2_models
    AD1_circle.eval()
    AD1_bifurcation.eval()
    AD2_circle.eval()
    AD2_bifurcation.eval()
    # Sincet the index and parent will not change in different batches, we can calculate it once
    batch_circle, batch_bifurcation = data_loader[0]
    _,n_circle,_, index_circle, parent_circle = batch_circle
    _,n_bifurcation,_, index_bifurcation, parent_bifurcation = batch_bifurcation
    edge_index, distance_to_root = create_global_graph(index_circle, index_bifurcation, parent_circle, parent_bifurcation)
    edge_index = edge_index.to("cuda")
    distance_to_root = distance_to_root.to("cuda")
    edge_distance = get_edge_distance(index_coordinate, edge_index, index_circle, index_bifurcation)
    node_physical_distance = edge_distance_2_node_physical_distance(edge_index, edge_distance)
    
    num_of_surfaces = int(n_circle.x.shape[0] / 17)
    indexs = [i for i in range(num_of_surfaces)]
    num_of_bifurcations = int(n_bifurcation.x.shape[0] / 23)
    indexs = [i + num_of_surfaces for i in range(num_of_bifurcations)]

    target_physical_circle = n_circle.x.view(-1, 17, 1)
    target_physical_bifurcation = n_bifurcation.x.view(-1, 23, 1)
    target_latent_circle = AD2_circle.encode(n_circle.x, n_circle.edge_index, n_circle.batch)
    target_latent_bifurcation = AD2_bifurcation.encode(n_bifurcation.x, n_bifurcation.edge_index, n_bifurcation.batch)

    # Save the result to vtk file
    _, _, data_series = file_to_graph_data_ls(data_path, with_series=True)
    ret_data = torch.zeros((1, 1))
    ret_exact_data = torch.zeros((1, 1))
    ret_latent_data = torch.zeros((1, 1))
    ret_latent_exact_data = torch.zeros((1, 1))
    idx_circle = 0
    idx_bifurcation = 0
    for series_index in data_series:
        if series_index == 0:
            # circle data
            physical_result_circle_tmp = target_physical_circle[idx_circle]
            physical_result_circle_exact_tmp = target_physical_circle[idx_circle]
            ret_data = torch.vstack((ret_data, physical_result_circle_tmp.cpu().detach()))
            ret_exact_data = torch.vstack((ret_exact_data, physical_result_circle_exact_tmp.cpu().detach()))
            ret_latent_data = torch.vstack((ret_latent_data, (torch.ones_like(physical_result_circle_exact_tmp)*target_latent_circle[idx_circle]).cpu().detach()))
            ret_latent_exact_data = torch.vstack((ret_latent_exact_data, (torch.ones_like(physical_result_circle_exact_tmp)*target_latent_circle[idx_circle]).cpu().detach()))
            idx_circle += 1
        else:
            # bifurcation data
            physical_result_bifurcation_tmp = target_physical_bifurcation[idx_bifurcation]
            physical_result_bifurcation_exact_tmp = target_physical_bifurcation[idx_bifurcation]
            ret_data = torch.vstack((ret_data, physical_result_bifurcation_tmp.cpu().detach()))
            ret_exact_data = torch.vstack((ret_exact_data, physical_result_bifurcation_exact_tmp.cpu().detach()))
            ret_latent_data = torch.vstack((ret_latent_data, (torch.ones_like(physical_result_bifurcation_exact_tmp)*target_latent_bifurcation[idx_bifurcation]).cpu().detach()))
            ret_latent_exact_data = torch.vstack((ret_latent_exact_data, (torch.ones_like(physical_result_bifurcation_exact_tmp)*target_latent_bifurcation[idx_bifurcation]).cpu().detach()))
            idx_bifurcation += 1
    ret_data = ret_data[1:].numpy()
    ret_exact_data = ret_exact_data[1:].numpy()
    ret_latent_data = ret_latent_data[1:].numpy()
    ret_latent_exact_data = ret_latent_exact_data[1:].numpy()
    ret_error = np.abs(ret_data - ret_exact_data)
    ret_latent_error = np.abs(ret_latent_data - ret_latent_exact_data)
    ret_data = np.ascontiguousarray(ret_data[:,0])
    ret_exact_data = np.ascontiguousarray(ret_exact_data[:,0])
    ret_error = np.ascontiguousarray(ret_error[:,0])
    ret_latent = np.ascontiguousarray(ret_latent_data[:,0])
    ret_latent_exact = np.ascontiguousarray(ret_latent_exact_data[:,0])
    ret_latent_error = np.ascontiguousarray(ret_latent_error[:,0])
    # Save the data to vtk file
    x_coord = readHDF5(data_path, "data_x")
    y_coord = readHDF5(data_path, "data_y")
    z_coord = readHDF5(data_path, "data_z")
    coordinate = np.vstack((x_coord, y_coord, z_coord)).T
    # Save the data to vtk
    field = {"nplus_pred": ret_data, "nplus_exact": ret_exact_data, "error": ret_error, "latent": ret_latent, "latent_exact": ret_latent_exact, "latent_error": ret_latent_error}
    if not os.path.exists(f"vtks/NN2_AD2/{fname}/nplus"):
        os.makedirs(f"vtks/NN2_AD2/{fname}/nplus")
    Save_vtk_with_dict(coordinate, field,   f"vtks/NN2_AD2/{fname}/nplus/NN2_AD2_result_{fname}_0")

    model_NN2.set_edge_index(edge_index)
    model_NN2.set_distance_to_root(distance_to_root)
    model_NN2.set_edge_distance(edge_distance)
    model_NN2.set_node_physical_distance(node_physical_distance)
    data_window = data_loader[0]
    circle_data, bifurcation_data = data_window
    u_circle, _,_,_,_ = circle_data
    u_bifurcation, _,_,_,_ = bifurcation_data
    u_latent_circle = AD1_circle.encode(u_circle.x, u_circle.edge_index, u_circle.batch)
    u_latent_bifurcation = AD1_bifurcation.encode(u_bifurcation.x, u_bifurcation.edge_index, u_bifurcation.batch)
    u_latent = torch.vstack((u_latent_circle, u_latent_bifurcation))
    model_NN2.set_velocity(u_latent)

    batch_circle_init, batch_bifurcation_init = data_loader[0]
    _, full_batch_t_circle_init, _, _, _ = batch_circle_init
    _, full_batch_t_bifurcation_init, _, _, _ = batch_bifurcation_init
    n_latent_circle_init = AD2_circle.encode(full_batch_t_circle_init.x, full_batch_t_circle_init.edge_index, full_batch_t_circle_init.batch)
    n_latent_bifurcation_init = AD2_bifurcation.encode(full_batch_t_bifurcation_init.x, full_batch_t_bifurcation_init.edge_index, full_batch_t_bifurcation_init.batch)
    data_init = torch.vstack((n_latent_circle_init, n_latent_bifurcation_init))
    model_NN2.set_initial_condition(data_init)
    eval_time_step = [t*dt for t in range(0, len(data_loader)+1)]
    start_time = time.time()
    pre_graph_tensor = odeint(model_NN2, data_init, torch.tensor(eval_time_step).to("cuda"), method="dopri5", atol=1e-3, rtol=1e-4)
    
    if save_only_last is True:
        cnt = len(data_loader) - 1
        print(f"cnt: {cnt}")
        data_loader = [data_loader[-1]]
    else:
        cnt = 1
    for batch_circle, batch_bifurcation in data_loader:
        # Read for exact solution
        _, full_batch_t_circle, full_batch_t1_circle, index_circle, parent_circle = batch_circle
        _, full_batch_t_bifurcation, full_batch_t1_bifurcation, index_bifurcation, parent_bifurcation = batch_bifurcation
        pred_graph = pre_graph_tensor[cnt]
        num_of_surfaces = int(full_batch_t_circle.x.shape[0] / 17)
        indexs = [i for i in range(num_of_surfaces)]
        latent_result_on_cross_section_circle = pred_graph[indexs]
        physical_result_circle = AD2_circle.decode(latent_result_on_cross_section_circle, full_batch_t_circle.edge_index, full_batch_t_circle.batch)
        num_of_bifurcations = int(full_batch_t_bifurcation.x.shape[0] / 23)
        indexs = [i + num_of_surfaces for i in range(num_of_bifurcations)]
        latent_result_on_cross_section_bifurcation = pred_graph[indexs]
        physical_result_bifurcation = AD2_bifurcation.decode(latent_result_on_cross_section_bifurcation, full_batch_t_bifurcation.edge_index, full_batch_t_bifurcation.batch)
        target_physical_circle = full_batch_t1_circle.x.view(-1, 17, 1)
        target_physical_bifurcation = full_batch_t1_bifurcation.x.view(-1, 23, 1)
        target_latent_circle = AD2_circle.encode(full_batch_t1_circle.x, full_batch_t1_circle.edge_index, full_batch_t1_circle.batch)
        target_latent_bifurcation = AD2_bifurcation.encode(full_batch_t1_bifurcation.x, full_batch_t1_bifurcation.edge_index, full_batch_t1_bifurcation.batch)
        # Save the result to vtk file
        _, _, data_series = file_to_graph_data_ls(data_path, with_series=True)
        ret_data = torch.zeros((1, 1))
        ret_exact_data = torch.zeros((1, 1))
        ret_latent_data = torch.zeros((1, 1))
        ret_latent_exact_data = torch.zeros((1, 1))
        idx_circle = 0
        idx_bifurcation = 0
        for series_index in data_series:
            if series_index == 0:
                # circle data
                physical_result_circle_tmp = physical_result_circle[idx_circle]
                physical_result_circle_exact_tmp = target_physical_circle[idx_circle]
                ret_data = torch.vstack((ret_data, physical_result_circle_tmp.cpu().detach()))
                ret_exact_data = torch.vstack((ret_exact_data, physical_result_circle_exact_tmp.cpu().detach()))
                ret_latent_data = torch.vstack((ret_latent_data, (torch.ones_like(physical_result_circle_exact_tmp)*latent_result_on_cross_section_circle[idx_circle]).cpu().detach()))
                ret_latent_exact_data = torch.vstack((ret_latent_exact_data, (torch.ones_like(physical_result_circle_exact_tmp)*target_latent_circle[idx_circle]).cpu().detach()))
                idx_circle += 1
            else:
                # bifurcation data
                physical_result_bifurcation_tmp = physical_result_bifurcation[idx_bifurcation]
                physical_result_bifurcation_exact_tmp = target_physical_bifurcation[idx_bifurcation]
                ret_data = torch.vstack((ret_data, physical_result_bifurcation_tmp.cpu().detach()))
                ret_exact_data = torch.vstack((ret_exact_data, physical_result_bifurcation_exact_tmp.cpu().detach()))
                ret_latent_data = torch.vstack((ret_latent_data, (torch.ones_like(physical_result_bifurcation_exact_tmp)*latent_result_on_cross_section_bifurcation[idx_bifurcation]).cpu().detach()))
                ret_latent_exact_data = torch.vstack((ret_latent_exact_data, (torch.ones_like(physical_result_bifurcation_exact_tmp)*target_latent_bifurcation[idx_bifurcation]).cpu().detach()))
                idx_bifurcation += 1
        ret_data = ret_data[1:].numpy()
        ret_exact_data = ret_exact_data[1:].numpy()
        ret_latent_data = ret_latent_data[1:].numpy()
        ret_latent_exact_data = ret_latent_exact_data[1:].numpy()
        ret_error = np.abs(ret_data - ret_exact_data)
        ret_latent_error = np.abs(ret_latent_data - ret_latent_exact_data)
        ret_data = np.ascontiguousarray(ret_data[:,0])
        ret_exact_data = np.ascontiguousarray(ret_exact_data[:,0])
        ret_error = np.ascontiguousarray(ret_error[:,0])
        ret_latent = np.ascontiguousarray(ret_latent_data[:,0])
        ret_latent_exact = np.ascontiguousarray(ret_latent_exact_data[:,0])
        ret_latent_error = np.ascontiguousarray(ret_latent_error[:,0])
        # Save the data to vtk file
        x_coord = readHDF5(data_path, "data_x")
        y_coord = readHDF5(data_path, "data_y")
        z_coord = readHDF5(data_path, "data_z")
        coordinate = np.vstack((x_coord, y_coord, z_coord)).T
        # Save the data to vtk
        field = {"nplus_pred": ret_data, "nplus_exact": ret_exact_data, "error": ret_error, "latent": ret_latent, "latent_exact": ret_latent_exact, "latent_error": ret_latent_error}
        Save_vtk_with_dict(coordinate, field,   f"vtks/NN2_AD2/{fname}/nplus/NN2_AD2_result_{fname}_{cnt}")
        cnt += 1
    end_time = time.time()
    print(f"Time taken for the whole process: {end_time - start_time} seconds")

def Model_training_NN2(odefunc, AD1_models, AD2_models, optimizer, scheduler, loss_fn, data_loaders, dts, index_coordinates, percentage, num_epoch):
    print("Moving all data to GPU")
    if ENABLE_GPU:
        new_data_loaders = []
        for data_loader in data_loaders:
            # if move all data to gpu
            data_loader_new = []
            cnt = 0
            for data_pipe, data_bif in data_loader:
                u, t, t1, index, parent = data_pipe
                u_bif, t_bif, t1_bif, index_bif, parent_bif = data_bif
                # Check if the data has nan or inf for u, t, t1
                if torch.isnan(u.x).any() or torch.isnan(t.x).any() or torch.isnan(t1.x).any():
                    print("Data has nan for pipe")
                if torch.isinf(u.x).any() or torch.isinf(t.x).any() or torch.isinf(t1.x).any():
                    cnt += 1
                    print("Data has inf for pipe")
                if torch.isnan(u_bif.x).any() or torch.isnan(t_bif.x).any() or torch.isnan(t1_bif.x).any():
                    print("Data has nan for bif")
                if torch.isinf(u_bif.x).any() or torch.isinf(t_bif.x).any() or torch.isinf(t1_bif.x).any():
                    print("Data has inf for bif")      
                u = u.to("cuda")
                t = t.to("cuda")
                t1 = t1.to("cuda")
                index = torch.tensor(index).to(torch.int64).to("cuda")
                parent = torch.tensor(parent).to(torch.int64).to("cuda")
                u_bif = u_bif.to("cuda")
                t_bif = t_bif.to("cuda")
                t1_bif = t1_bif.to("cuda")
                index_bif = torch.tensor(index_bif).to(torch.int64).to("cuda")
                parent_bif = torch.tensor(parent_bif).to(torch.int64).to("cuda")
                data_loader_new.append([[u, t, t1, index, parent], [u_bif, t_bif, t1_bif, index_bif, parent_bif]])
            print(f"Data has nan for {cnt} times")
            data_loader = data_loader_new
            new_data_loaders.append(data_loader)
        data_loaders = new_data_loaders
    
    best_model_state = None
    best_loss = float("inf")
    odefunc.train()
    AD1_circle, AD1_bifurcation = AD1_models
    AD2_circle, AD2_bifurcation = AD2_models
    AD1_circle.eval()
    AD1_bifurcation.eval()
    AD2_circle.eval()
    AD2_bifurcation.eval()
    # Sincet the index and parent will not change in different batches, we can calculate it once
    print("Calculating global edge index for all dataloaders")
    edge_index_ls = []
    distance_to_root_ls = []
    mask_ls = []
    edge_distance_ls = []
    node_physical_distance_ls = []
    for idx_tmp, data_loader in enumerate(data_loaders):
        batch_circle, batch_bifurcation = data_loader[0]
        _,_,_, index_circle, parent_circle = batch_circle
        _,_,_, index_bifurcation, parent_bifurcation = batch_bifurcation
        edge_index, distance_to_root = create_global_graph(index_circle, index_bifurcation, parent_circle, parent_bifurcation)
        edge_index = edge_index.to("cuda")
        distance_to_root = distance_to_root.to("cuda")
        edge_index_ls.append(edge_index)
        distance_to_root_ls.append(distance_to_root)
        non_root_mask = (distance_to_root != 0).float().to("cuda")
        mask_ls.append(non_root_mask)
        edge_distance = get_edge_distance(index_coordinates[idx_tmp], edge_index, index_circle, index_bifurcation)
        node_physical_distance = edge_distance_2_node_physical_distance(edge_index, edge_distance).to("cuda")
        node_physical_distance_ls.append(node_physical_distance)
        edge_distance_ls.append(edge_distance)

    print("Creating new data loader for efficiency")
    new_data_loaders = []
    for idx, data_loader in enumerate(data_loaders):
        new_data_loader = []
        for idxx, (batch_circle, batch_bifurcation) in enumerate(data_loader):
            full_batch_u_circle, full_batch_t_circle, full_batch_t1_circle, index_circle, parent_circle = batch_circle
            full_batch_u_bifurcation, full_batch_t_bifurcation, full_batch_t1_bifurcation, index_bifurcation, parent_bifurcation = batch_bifurcation
            u_latent_circle = AD1_circle.encode(full_batch_u_circle.x, full_batch_u_circle.edge_index, full_batch_u_circle.batch).detach()
            u_latent_bifurcation = AD1_bifurcation.encode(full_batch_u_bifurcation.x, full_batch_u_bifurcation.edge_index, full_batch_u_bifurcation.batch).detach()
            n_latent_circle = AD2_circle.encode(full_batch_t_circle.x, full_batch_t_circle.edge_index, full_batch_t_circle.batch).detach()
            n_latent_bifurcation = AD2_bifurcation.encode(full_batch_t_bifurcation.x, full_batch_t_bifurcation.edge_index, full_batch_t_bifurcation.batch).detach()
            n1_latent_circle = AD2_circle.encode(full_batch_t1_circle.x, full_batch_t1_circle.edge_index, full_batch_t1_circle.batch).detach()
            n1_latent_bifurcation = AD2_bifurcation.encode(full_batch_t1_bifurcation.x, full_batch_t1_bifurcation.edge_index, full_batch_t1_bifurcation.batch).detach()
            latent_circle = torch.hstack((n_latent_circle, u_latent_circle))
            latent_bifurcation = torch.hstack((n_latent_bifurcation, u_latent_bifurcation))
            data_x = torch.vstack((latent_circle, latent_bifurcation))
            data = data_x[:,0].reshape(-1,1).detach()
            latent_velocity = data_x[:, 1:].reshape(-1, 3).detach()
            target = torch.vstack((n1_latent_circle, n1_latent_bifurcation)).detach()
            target_time = torch.tensor([(idxx) * dts[idx], (idxx+1) * dts[idx]]).to("cuda")
            new_data_loader_data = [data, target, latent_velocity, target_time]
            new_data_loader.append(new_data_loader_data)
        data_loader = new_data_loader
        new_data_loaders.append(data_loader)
        print("End creating the data loader")
    data_loaders = new_data_loaders
    print("Creating dataset with windows")
    new_data_loaders = []
    for data_loader in data_loaders:
        new_data_loader = []
        for j in range(1):
            random_window_len = len(data_loader)
            random_window_start = 0
            tmp_data = []
            tmp_target = []
            tmp_latent_velocity = []
            tmp_target_time = []
            for i in range(random_window_start, random_window_start + random_window_len):
                data, target, latent_velocity, target_time = data_loader[i]
                tmp_data.append(data)
                tmp_target.append(target)
                tmp_latent_velocity.append(latent_velocity)
                tmp_target_time.append(target_time[0])
            new_data_loader.append([tmp_data, tmp_target, tmp_latent_velocity, tmp_target_time])
        new_data_loaders.append(new_data_loader)
    data_loaders = new_data_loaders
    total_loss_list = []
    print("End creating the dataset with windows")
    for epoch in tqdm.tqdm(range(num_epoch)):
        total_loss = 0
        loss_all_batch = torch.tensor(0, dtype=torch.float).to("cuda")
        for data_loader_idx, data_loader in enumerate(data_loaders):
            odefunc.set_edge_index(edge_index_ls[data_loader_idx])
            data_window = data_loader[0]
            n0_ls, _, latent_velocity, _ = data_window
            latent_velocity = latent_velocity[0]
            odefunc.set_velocity(latent_velocity)
            odefunc.set_distance_to_root(distance_to_root_ls[data_loader_idx])
            odefunc.set_edge_distance(edge_distance_ls[data_loader_idx])
            odefunc.set_initial_condition(n0_ls[0])
            odefunc.set_node_physical_distance(node_physical_distance_ls[data_loader_idx])
            for data_window in data_loader:
                n0_ls, _, _, target_time_ls = data_window
                target_time_tensor = torch.tensor(target_time_ls).to("cuda")
                optimizer.zero_grad()
                n1_tensor = torch.stack(n0_ls[1:])
                pred_n_tensor = odeint(odefunc, n0_ls[0], target_time_tensor, method='dopri5', atol=1e-3, rtol=1e-4)
                pred_n1_tensor = pred_n_tensor[1:]
                loss = torch.mean(torch.abs(pred_n1_tensor - n1_tensor)[: int(pred_n1_tensor.shape[0] * percentage)])
                loss_all_batch += loss
                total_loss += loss.item()
                total_loss_list.append(loss.item())    
        loss_all_batch.backward()
        optimizer.step()
        scheduler.step()
        if epoch % 10 == 0:
            print(f"Epoch: {epoch}, Loss: {total_loss:.5f}")
        if epoch % 100 == 0 and epoch > 0:
            if total_loss < best_loss:
                best_loss = total_loss
                best_model_state = odefunc.state_dict().copy()
            torch.save(best_model_state, "model/odefunc_nplus.pth")
    if best_model_state is not None:
        return best_model_state
    else:
        raise ValueError("No best model state found. Please check the training process.")

def data_preprocessing(path, dt_path, idx_start, idx_end):
    dt = get_dt(dt_path)
    data_loader, index_coordinate = create_data_loader_NN2(path, batch_size=-1, idx_start=idx_start, idx_end=idx_end)
    return data_loader, index_coordinate, dt

def check_embedding(exact_solution_tr_path, exact_solution_ns_path, index_coordinate, case_name):
    data_surfaces, data_bifurcations = data_sep_bifurcation(readHDF5(exact_solution_ns_path))
    coord_total = torch.zeros((1, 3))
    for data_surface_dict in data_surfaces:
        coord_circle_x = data_surface_dict["data_x"].reshape(-1, 1)
        coord_circle_y = data_surface_dict["data_y"].reshape(-1, 1)
        coord_circle_z = data_surface_dict["data_z"].reshape(-1, 1)
        coord_central = torch.tensor(np.hstack((coord_circle_x[0], coord_circle_y[0], coord_circle_z[0])))
        coord_total = torch.vstack((coord_total, coord_central))
    for data_bifurcation_dict in data_bifurcations:
        coord_bifurcation_x = data_bifurcation_dict["data_x"].reshape(-1, 1)
        coord_bifurcation_y = data_bifurcation_dict["data_y"].reshape(-1, 1)
        coord_bifurcation_z = data_bifurcation_dict["data_z"].reshape(-1, 1)
        coord_central = torch.tensor(np.hstack((coord_bifurcation_x[0], coord_bifurcation_y[0], coord_bifurcation_z[0])))
        coord_total = torch.vstack((coord_total, coord_central))
    coord_total = coord_total[1:]
    data_circle, data_bifurcation = Two_file_to_graph_data(exact_solution_tr_path, exact_solution_tr_path, exact_solution_ns_path, data_key="nplus_value")
    
    _, _,_, index_circle, parent_circle = data_circle
    _, _,_, index_bifurcation, parent_bifurcation = data_bifurcation
    index_circle = torch.tensor(index_circle).to(torch.int64).to("cuda")
    parent_circle = torch.tensor(parent_circle).to(torch.int64).to("cuda")
    index_bifurcation = torch.tensor(index_bifurcation).to(torch.int64).to("cuda")
    parent_bifurcation = torch.tensor(parent_bifurcation).to(torch.int64).to("cuda")
    edge_index, distance_to_root = create_global_graph(index_circle, index_bifurcation, parent_circle, parent_bifurcation)
    edge_index = edge_index.to("cuda")
    distance_to_root = distance_to_root.to("cuda")
    node_physical_distance = edge_distance_2_node_physical_distance(edge_index, get_edge_distance(index_coordinate, edge_index, index_circle, index_bifurcation).to("cuda"))
    arrow_pairs = []
    for ii in range(edge_index.shape[1]):
        edge_pair_start = edge_index[0][ii]
        edge_pair_end = edge_index[1][ii]
        coord_start = coord_total[edge_pair_start]
        coord_end = coord_total[edge_pair_end]
        arrow_pairs.append((coord_start.numpy(), coord_end.numpy()))
    # Prepare start points and vector directions
    starts = np.array([start for start, end in arrow_pairs])
    vectors = np.array([end - start for start, end in arrow_pairs])
    # Create point cloud with starting points
    points = pv.PolyData(starts)
    # Add vectors as point data (for glyph orientation)
    points["vectors"] = vectors
    # Create arrow glyphs (automatically aligns and scales)
    arrows = points.glyph(orient="vectors", scale="vectors", factor=1.0)
    # Save to VTU
    arrows.save(f"vtks/embedding/{case_name}_edge.vtk")

    points = pv.PolyData(coord_total.numpy())
    points["distance"] = distance_to_root.squeeze().cpu().numpy()
    points["node_physical_distance"] = node_physical_distance.squeeze().cpu().numpy()
    points.save(f"vtks/embedding/{case_name}_embedding.vtk")

def data_loader_preprocessing(sim_name, sim_num):
    path = f"../pre_processing/{sim_name}/Data/sim_{sim_num}/wholeTree"
    dt_path = f"../pre_processing/{sim_name}/sim_result/sim_{sim_num}"
    data_loader, index_coordinate, dt = data_preprocessing(path, dt_path)
    return data_loader, index_coordinate, dt

def main():
    data_loaders = []
    index_coordinates = []
    dts = []
    for sim_name in ["NMO_66731", "NMO_66748"]:
        for sim_num in range(10):
            data_loader, index_coordinate, dt = data_loader_preprocessing(sim_name, sim_num)
            data_loaders.append(data_loader)
            index_coordinates.append(index_coordinate)
            dts.append(dt)

    # Read AD2 model
    encoder_shared_ns = SharedEncoder(feature_num=3)
    encoder_shared_tr = SharedEncoder(feature_num=1)
    AD2_model_circle = GNN_Autoencoder_circle(encoder_shared_tr)
    AD2_model_bifurcation = GNN_Autoencoder_bifurcation(encoder_shared_tr)
    AD1_model_circle = GNN_Autoencoder_circle(encoder_shared_ns)
    AD1_model_bifurcation = GNN_Autoencoder_bifurcation(encoder_shared_ns)
    if ENABLE_GPU:
        AD2_model_circle.to("cuda")
        AD2_model_bifurcation.to("cuda")
        AD1_model_circle.to("cuda")
        AD1_model_bifurcation.to("cuda")
    AD2_model_circle.load_state_dict(torch.load("model/AD2_model_circle.pth", weights_only=True))
    AD2_model_bifurcation.load_state_dict(torch.load("model/AD2_model_bifurcation.pth", weights_only=True))
    AD1_model_circle.load_state_dict(torch.load("model/AD1_model_circle.pth", weights_only=True))
    AD1_model_bifurcation.load_state_dict(torch.load("model/AD1_model_bifurcation.pth", weights_only=True))
    AD1_model_circle.eval()
    AD1_model_bifurcation.eval()
    AD2_model_circle.eval()
    AD2_model_bifurcation.eval()
    AD1_models = [AD1_model_circle, AD1_model_bifurcation]
    AD2_models = [AD2_model_circle, AD2_model_bifurcation]
    # Create model
    model_latent_dynamic = GNNODEFunc(5,[128]*4, 1)
    if ENABLE_GPU:
        model_latent_dynamic.to("cuda")
    # Create optimizer and loss function
    optimizer = torch.optim.Adam(model_latent_dynamic.parameters(), lr=1E-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1000, gamma=0.7)
    loss_fn = torch.nn.MSELoss()
    model_latent_dynamic.load_state_dict(torch.load("model/odefunc_nplus.pth", map_location="cuda:0"))
    best_state = Model_training_NN2(model_latent_dynamic, AD1_models, AD2_models, optimizer, scheduler, loss_fn, data_loaders, dts, index_coordinates, 1.0, num_epoch=10000)
    # Save model
    torch.save(best_state, "model/odefunc_nplus.pth")


if __name__ == "__main__":
    main()