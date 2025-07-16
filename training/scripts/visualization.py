import torch
from parameter import ENABLE_GPU
from scripts.utils import file_to_graph_data_ls
from scripts.mesh_templates import index_mapping_circle_surface, index_extract_order_circle_surface, index_extract_order_bifurcation_surface, index_mapping_bifurcation_surface
from matplotlib import pyplot as plt
import numpy as np
from pyevtk.hl import pointsToVTK

def Check_result_AD1(model, data_path):
    """Visualization for AD1 result using matplotlib\n
    Can be called by setting the PLOT_RESULT flag to True \n
    OR \n
    running the eval_model_AD1.py script

    Args:
        model (_type_): _description_
        data_path (_type_): _description_
    """
    model.eval()
    # Load data
    circle_data_ls, bifurcation_data_ls = file_to_graph_data_ls(data_path)
    if model.num_nodes == 17:
        data = circle_data_ls
    elif model.num_nodes == 23:
        data = bifurcation_data_ls
    else:
        raise ValueError("Unknown data")
    if ENABLE_GPU:
        data = [x.to("cuda") for x in data]
    # Get output
    output = model(data[0])
    # Calculate loss
    loss = torch.nn.MSELoss()(output, data[0].x.view(-1, model.num_nodes, 3))
    print(f"Loss: {loss.item()}")
    # Plot result
    coordinate_by_extract_order = []
    if model.num_nodes == 17:
        for i in index_extract_order_circle_surface:
            coordinate_by_extract_order.append(index_mapping_circle_surface[i])
    elif model.num_nodes == 23:
        for i in index_extract_order_bifurcation_surface:
            coordinate_by_extract_order.append(index_mapping_bifurcation_surface[i])
    else:
        raise ValueError("Unknown data")
    coordinate_by_extract_order = np.array(coordinate_by_extract_order)
    fig, ax = plt.subplots(4, 3, figsize=(10, 5))
    # store max and min values of true for colorbar
    ux_max, ux_min = data[0].x[:, 0].cpu().detach().numpy().max(), data[0].x[:, 0].cpu().detach().numpy().min()
    uy_max, uy_min = data[0].x[:, 1].cpu().detach().numpy().max(), data[0].x[:, 1].cpu().detach().numpy().min()
    uz_max, uz_min = data[0].x[:, 2].cpu().detach().numpy().max(), data[0].x[:, 2].cpu().detach().numpy().min()
    umag_max, umag_min = np.sqrt(data[0].x[:, 0].cpu().detach().numpy()**2 + data[0].x[:, 1].cpu().detach().numpy()**2 + data[0].x[:, 2].cpu().detach().numpy()**2).max(), np.sqrt(data[0].x[:, 0].cpu().detach().numpy()**2 + data[0].x[:, 1].cpu().detach().numpy()**2 + data[0].x[:, 2].cpu().detach().numpy()**2).min()
    # plot scatter
    ux_true = ax[0,0].scatter(coordinate_by_extract_order[:, 0], coordinate_by_extract_order[:, 1], c=data[0].x[:, 0].cpu().detach().numpy(), cmap="coolwarm", vmin=ux_min, vmax=ux_max)
    ux_pred = ax[0,1].scatter(coordinate_by_extract_order[:, 0], coordinate_by_extract_order[:, 1], c=output[0][:, 0].cpu().detach().numpy(), cmap="coolwarm", vmin=ux_min, vmax=ux_max)
    uy_true = ax[1,0].scatter(coordinate_by_extract_order[:, 0], coordinate_by_extract_order[:, 1], c=data[0].x[:, 1].cpu().detach().numpy(), cmap="coolwarm", vmin=uy_min, vmax=uy_max)
    uy_pred = ax[1,1].scatter(coordinate_by_extract_order[:, 0], coordinate_by_extract_order[:, 1], c=output[0][:, 1].cpu().detach().numpy(), cmap="coolwarm", vmin=uy_min, vmax=uy_max)
    uz_true = ax[2,0].scatter(coordinate_by_extract_order[:, 0], coordinate_by_extract_order[:, 1], c=data[0].x[:, 2].cpu().detach().numpy(), cmap="coolwarm", vmin=uz_min, vmax=uz_max)
    uz_pred = ax[2,1].scatter(coordinate_by_extract_order[:, 0], coordinate_by_extract_order[:, 1], c=output[0][:, 2].cpu().detach().numpy(), cmap="coolwarm", vmin=uz_min, vmax=uz_max)
    umag_true = ax[3,0].scatter(coordinate_by_extract_order[:, 0], coordinate_by_extract_order[:, 1], c=np.sqrt(data[0].x[:, 0].cpu().detach().numpy()**2 + data[0].x[:, 1].cpu().detach().numpy()**2 + data[0].x[:, 2].cpu().detach().numpy()**2), cmap="coolwarm", vmin=umag_min, vmax=umag_max)
    umag_pred = ax[3,1].scatter(coordinate_by_extract_order[:, 0], coordinate_by_extract_order[:, 1], c=np.sqrt(output[0][:, 0].cpu().detach().numpy()**2 + output[0][:, 1].cpu().detach().numpy()**2 + output[0][:, 2].cpu().detach().numpy()**2), cmap="coolwarm", vmin=umag_min, vmax=umag_max)
    ux_error = ax[0,2].scatter(coordinate_by_extract_order[:, 0], coordinate_by_extract_order[:, 1], c=np.abs(data[0].x[:, 0].cpu().detach().numpy()-output[0][:, 0].cpu().detach().numpy()), cmap="coolwarm")
    uy_error = ax[1,2].scatter(coordinate_by_extract_order[:, 0], coordinate_by_extract_order[:, 1], c=np.abs(data[0].x[:, 1].cpu().detach().numpy()-output[0][:, 1].cpu().detach().numpy()), cmap="coolwarm")
    uz_error = ax[2,2].scatter(coordinate_by_extract_order[:, 0], coordinate_by_extract_order[:, 1], c=np.abs(data[0].x[:, 2].cpu().detach().numpy()-output[0][:, 2].cpu().detach().numpy()), cmap="coolwarm")
    umag_error = ax[3,2].scatter(coordinate_by_extract_order[:, 0], coordinate_by_extract_order[:, 1], c=np.abs(np.sqrt(data[0].x[:, 0].cpu().detach().numpy()**2 + data[0].x[:, 1].cpu().detach().numpy()**2 + data[0].x[:, 2].cpu().detach().numpy()**2)-np.sqrt(output[0][:, 0].cpu().detach().numpy()**2 + output[0][:, 1].cpu().detach().numpy()**2 + output[0][:, 2].cpu().detach().numpy()**2)), cmap="coolwarm")
    # Add label at top of columns
    ax[0, 0].set_title("True Velocity")
    ax[0, 1].set_title("Predicted Velocity")
    ax[0, 2].set_title("Error")
    # Add label at left of rows
    ax[0, 0].set_ylabel("Velocity x")
    ax[1, 0].set_ylabel("Velocity y")
    ax[2, 0].set_ylabel("Velocity z")
    ax[3, 0].set_ylabel("Velocity mag")
    # plot colorbar
    cbar = fig.colorbar(ux_true, ax=ax[0, 1])
    cbar.set_label("Velocity x (m/s)")
    cbar = fig.colorbar(uy_true, ax=ax[1, 1])
    cbar.set_label("Velocity y (m/s)")
    cbar = fig.colorbar(uz_true, ax=ax[2, 1])
    cbar.set_label("Velocity z (m/s)")
    cbar = fig.colorbar(umag_true, ax=ax[3, 1])
    cbar.set_label("Velocity mag (m/s)")
    cbar = fig.colorbar(ux_error, ax=ax[0, 2])
    cbar.set_label("Error (m/s)")
    cbar = fig.colorbar(uy_error, ax=ax[1, 2])
    cbar.set_label("Error (m/s)")
    cbar = fig.colorbar(uz_error, ax=ax[2, 2])
    cbar.set_label("Error (m/s)")
    cbar = fig.colorbar(umag_error, ax=ax[3, 2])
    cbar.set_label("Error (m/s)")
    plt.show()

def Check_result_NN1(NN1_model, AD1_model, data_path):
    """Function to render the result using matplotlib

    Args:
        NN1_model (_type_): NN1 model
        AD1_model (_type_): AD1 model
        data_path (_type_): the data path to be visualized
    """
    # TODO: Add the visualization for the bifurcation data
    NN1_model.eval()
    AD1_model.eval()
    # Load data
    circle_data_ls, bifurcation_data_ls = file_to_graph_data_ls(data_path)
    if ENABLE_GPU:
        circle_data_ls = [x.to("cuda") for x in circle_data_ls]
        bifurcation_data_ls = [x.to("cuda") for x in bifurcation_data_ls]
    # Get output
    NN1_result = NN1_model(circle_data_ls[0].x[0].reshape(-1,1))
    output = AD1_model.decode(NN1_result, circle_data_ls[0].edge_index, circle_data_ls[0].batch)
    # Calculate loss
    loss = torch.nn.MSELoss()(output, circle_data_ls[0].x.view(-1, AD1_model.num_nodes, 3))
    print(f"Loss: {loss.item()}")
    # Plot result
    coordinate_by_extract_order = []
    for i in index_extract_order_circle_surface:
        coordinate_by_extract_order.append(index_mapping_circle_surface[i])
    coordinate_by_extract_order = np.array(coordinate_by_extract_order)
    fig, ax = plt.subplots(4, 3, figsize=(10, 5))
    # store max and min values of true for colorbar
    ux_max, ux_min = circle_data_ls[0].x[:, 0].cpu().detach().numpy().max(), circle_data_ls[0].x[:, 0].cpu().detach().numpy().min()
    uy_max, uy_min = circle_data_ls[0].x[:, 1].cpu().detach().numpy().max(), circle_data_ls[0].x[:, 1].cpu().detach().numpy().min()
    uz_max, uz_min = circle_data_ls[0].x[:, 2].cpu().detach().numpy().max(), circle_data_ls[0].x[:, 2].cpu().detach().numpy().min()
    umag_max, umag_min = np.sqrt(circle_data_ls[0].x[:, 0].cpu().detach().numpy()**2 + circle_data_ls[0].x[:, 1].cpu().detach().numpy()**2 + circle_data_ls[0].x[:, 2].cpu().detach().numpy()**2).max(), np.sqrt(circle_data_ls[0].x[:, 0].cpu().detach().numpy()**2 + circle_data_ls[0].x[:, 1].cpu().detach().numpy()**2 + circle_data_ls[0].x[:, 2].cpu().detach().numpy()**2).min()
    # plot scatter
    ux_true = ax[0,0].scatter(coordinate_by_extract_order[:, 0], coordinate_by_extract_order[:, 1], c=circle_data_ls[0].x[:, 0].cpu().detach().numpy(), cmap="coolwarm", vmin=ux_min, vmax=ux_max)
    ux_pred = ax[0,1].scatter(coordinate_by_extract_order[:, 0], coordinate_by_extract_order[:, 1], c=output[0][:, 0].cpu().detach().numpy(), cmap="coolwarm", vmin=ux_min, vmax=ux_max)
    uy_true = ax[1,0].scatter(coordinate_by_extract_order[:, 0], coordinate_by_extract_order[:, 1], c=circle_data_ls[0].x[:, 1].cpu().detach().numpy(), cmap="coolwarm", vmin=uy_min, vmax=uy_max)
    uy_pred = ax[1,1].scatter(coordinate_by_extract_order[:, 0], coordinate_by_extract_order[:, 1], c=output[0][:, 1].cpu().detach().numpy(), cmap="coolwarm", vmin=uy_min, vmax=uy_max)
    uz_true = ax[2,0].scatter(coordinate_by_extract_order[:, 0], coordinate_by_extract_order[:, 1], c=circle_data_ls[0].x[:, 2].cpu().detach().numpy(), cmap="coolwarm", vmin=uz_min, vmax=uz_max)
    uz_pred = ax[2,1].scatter(coordinate_by_extract_order[:, 0], coordinate_by_extract_order[:, 1], c=output[0][:, 2].cpu().detach().numpy(), cmap="coolwarm", vmin=uz_min, vmax=uz_max)
    umag_true = ax[3,0].scatter(coordinate_by_extract_order[:, 0], coordinate_by_extract_order[:, 1], c=np.sqrt(circle_data_ls[0].x[:, 0].cpu().detach().numpy()**2 + circle_data_ls[0].x[:, 1].cpu().detach().numpy()**2 + circle_data_ls[0].x[:, 2].cpu().detach().numpy()**2), cmap="coolwarm", vmin=umag_min, vmax=umag_max)
    umag_pred = ax[3,1].scatter(coordinate_by_extract_order[:, 0], coordinate_by_extract_order[:, 1], c=np.sqrt(output[0][:, 0].cpu().detach().numpy()**2 + output[0][:, 1].cpu().detach().numpy()**2 + output[0][:, 2].cpu().detach().numpy()**2), cmap="coolwarm", vmin=umag_min, vmax=umag_max)
    ux_error = ax[0,2].scatter(coordinate_by_extract_order[:, 0], coordinate_by_extract_order[:, 1], c=np.abs(circle_data_ls[0].x[:, 0].cpu().detach().numpy()-output[0][:, 0].cpu().detach().numpy()), cmap="coolwarm")
    uy_error = ax[1,2].scatter(coordinate_by_extract_order[:, 0], coordinate_by_extract_order[:, 1], c=np.abs(circle_data_ls[0].x[:, 1].cpu().detach().numpy()-output[0][:, 1].cpu().detach().numpy()), cmap="coolwarm")
    uz_error = ax[2,2].scatter(coordinate_by_extract_order[:, 0], coordinate_by_extract_order[:, 1], c=np.abs(circle_data_ls[0].x[:, 2].cpu().detach().numpy()-output[0][:, 2].cpu().detach().numpy()), cmap="coolwarm")
    umag_error = ax[3,2].scatter(coordinate_by_extract_order[:, 0], coordinate_by_extract_order[:, 1], c=np.abs(np.sqrt(circle_data_ls[0].x[:, 0].cpu().detach().numpy()**2 + circle_data_ls[0].x[:, 1].cpu().detach().numpy()**2 + circle_data_ls[0].x[:, 2].cpu().detach().numpy()**2)-np.sqrt(output[0][:, 0].cpu().detach().numpy()**2 + output[0][:, 1].cpu().detach().numpy()**2 + output[0][:, 2].cpu().detach().numpy()**2)), cmap="coolwarm")
    # Add label at top of columns
    ax[0, 0].set_title("True Velocity")
    ax[0, 1].set_title("Predicted Velocity")
    ax[0, 2].set_title("Error")
    # Add label at left of rows
    ax[0, 0].set_ylabel("Velocity x")
    ax[1, 0].set_ylabel("Velocity y")
    ax[2, 0].set_ylabel("Velocity z")
    ax[3, 0].set_ylabel("Velocity mag")
    # plot colorbar
    cbar = fig.colorbar(ux_true, ax=ax[0, 1])
    cbar.set_label("Velocity x (m/s)")
    cbar = fig.colorbar(uy_true, ax=ax[1, 1])
    cbar.set_label("Velocity y (m/s)")
    cbar = fig.colorbar(uz_true, ax=ax[2, 1])
    cbar.set_label("Velocity z (m/s)")
    cbar = fig.colorbar(umag_true, ax=ax[3, 1])
    cbar.set_label("Velocity mag (m/s)")
    cbar = fig.colorbar(ux_error, ax=ax[0, 2])
    cbar.set_label("Error (m/s)")
    cbar = fig.colorbar(uy_error, ax=ax[1, 2])
    cbar.set_label("Error (m/s)")
    cbar = fig.colorbar(uz_error, ax=ax[2, 2])
    cbar.set_label("Error (m/s)")
    cbar = fig.colorbar(umag_error, ax=ax[3, 2])
    cbar.set_label("Error (m/s)")
    plt.show()

def Save_vtk(coordinate, field, data_path):
    """Function to save the output of the model in vtk format

    Args:
        coordinate (nx3 np.array): point coordinates
        field (dict of np.array): field values
        data_path (_type_): the data path to be visualized
    """
    x = coordinate[:, 0]
    y = coordinate[:, 1]
    z = coordinate[:, 2]
    x = np.ascontiguousarray(x)
    y = np.ascontiguousarray(y)
    z = np.ascontiguousarray(z)
    
    if field.shape[1] == 1:
        field = {"n": np.ascontiguousarray(field[:,0])}
    elif field.shape[1] == 3:
        # save for velocity field
        field = {"velocity": (np.ascontiguousarray(field[:,0]), np.ascontiguousarray(field[:,1]), np.ascontiguousarray(field[:,2]))}
    pointsToVTK(data_path, x, y, z, data=field)

def Save_vtk_with_dict(coordinate, field, data_path):
    """Function to save the output of the model in vtk format

    Args:
        coordinate (nx3 np.array): point coordinates
        field (dict of np.array): field values
        data_path (_type_): the data path to be visualized
    """
    x = coordinate[:, 0]
    y = coordinate[:, 1]
    z = coordinate[:, 2]
    x = np.ascontiguousarray(x)
    y = np.ascontiguousarray(y)
    z = np.ascontiguousarray(z)
    
    pointsToVTK(data_path, x, y, z, data=field)