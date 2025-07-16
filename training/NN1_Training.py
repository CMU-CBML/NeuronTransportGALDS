import torch
from scripts.GNN_models import GNN_Autoencoder_circle, GNN_Autoencoder_bifurcation, SharedEncoder
import tqdm
from scripts.error_calculation import calculate_error_NN1, Mean_Relative_Error, Mean_Absolute_Error, Mean_Squared_Error, calculate_error_NN1_AD1
from parameter import ENABLE_GPU
from scripts.NN_Models import MultiLayerNet
from scripts.utils import file_to_graph_data_ls
from torch_geometric.data import DataLoader
from sklearn.model_selection import train_test_split

def create_data_loader_NN1(fname_ls, test_size=0.2, batch_size=32):
    """ Create DataLoader object from list of .h5 files

    Args:
        fname_ls (list): list of file names for the .h5 files

    Returns:
        DataLoader: DataLoader object for all data
    """
    data_ls_circle_total = []
    data_ls_bifurcation_total = []
    print(f"Creating data loader... (Total {len(fname_ls)} files)")
    for fname in tqdm.tqdm(fname_ls):
        data_ls_circle, data_ls_bifurcation = file_to_graph_data_ls(fname)
        if len(data_ls_circle) != 0:
            data_ls_circle_total.extend(data_ls_circle)
        if len(data_ls_bifurcation) != 0:
            data_ls_bifurcation_total.extend(data_ls_bifurcation)
    
    train_data_loader_circle = None
    test_data_loader_circle = None
    train_data_loader_bifurcation = None
    test_data_loader_bifurcation = None
    if test_size != 0:
        if len(data_ls_circle_total) != 0:
            train_data_ls_circle, test_data_ls_circle = train_test_split(data_ls_circle_total, test_size=test_size)
            train_data_loader_circle = DataLoader(train_data_ls_circle, batch_size=(batch_size if batch_size != -1 else len(train_data_ls_circle)), num_workers=0)
            test_data_loader_circle = DataLoader(test_data_ls_circle, batch_size=(batch_size if batch_size != -1 else len(test_data_ls_circle)), num_workers=0)
        if len(data_ls_bifurcation_total) != 0:
            train_data_ls_bifurcation, test_data_ls_bifurcation = train_test_split(data_ls_bifurcation_total, test_size=test_size)
            train_data_loader_bifurcation = DataLoader(train_data_ls_bifurcation, batch_size=(batch_size if batch_size != -1 else len(train_data_ls_bifurcation)), num_workers=0)
            test_data_loader_bifurcation = DataLoader(test_data_ls_bifurcation, batch_size=(batch_size if batch_size != -1 else len(test_data_ls_bifurcation)), num_workers=0)
    else:
        if len(data_ls_circle_total) != 0:
            train_data_loader_circle = DataLoader(data_ls_circle_total, batch_size=(batch_size if batch_size != -1 else len(data_ls_circle)), num_workers=0)
        if len(data_ls_bifurcation_total) != 0:
            train_data_loader_bifurcation = DataLoader(data_ls_bifurcation_total, batch_size=(batch_size if batch_size != -1 else len(data_ls_bifurcation)), num_workers=0)
        
    return train_data_loader_circle, test_data_loader_circle, train_data_loader_bifurcation, test_data_loader_bifurcation

def Model_training_NN1(NN1_model, AD1_model, optimizer, loss_fn, data_loader):
    if len(data_loader) == 1:
        data_loader_new = []
        for data in data_loader:
            data = data.to("cuda")
            data_loader_new.append(data)
        data_loader = data_loader_new
    best_model_state = None
    tmp_model_state = None
    best_loss = float("inf")
    NN1_model.train()
    for epoch in tqdm.tqdm(range(5000)):
        total_loss = 0
        tmp_model_state = NN1_model.state_dict().copy()
        for data in data_loader:
            if AD1_model.num_nodes == 17:
                num_of_surfaces = int(data.x.shape[0] / 17)
                indexs = [i*17 for i in range(num_of_surfaces)]
            elif AD1_model.num_nodes == 23:
                num_of_bifurcations = int(data.x.shape[0] / 23)
                indexs = [i*23 for i in range(num_of_bifurcations)]
            optimizer.zero_grad()
            if ENABLE_GPU:
                data = data.to("cuda")
            output = NN1_model(data.x[indexs])
            target = AD1_model.encode(data.x, data.edge_index, data.batch)
            loss = loss_fn(output, target)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        if total_loss < best_loss:
            best_loss = total_loss
            best_model_state = tmp_model_state
        if epoch % 100 == 0:
            print(f"Epoch: {epoch}, Loss: {total_loss:.5f}")
    if best_model_state is not None:
        return best_model_state
    else:
        raise ValueError("No best model state found. Please check the training process.")

def main():
    # First scan the data folder to get all the data names
    data_files1 = "../pre_processing/NMO_66748/Data/sim_0/wholeTree/wholeTree_ns.h5"
    data_files2 = "../pre_processing/NMO_66731/Data/sim_0/wholeTree/wholeTree_ns.h5"
    data_files = [data_files1, data_files2]
    # Create DataLoader object
    batch_size = -1
    train_data_loader_circle, test_data_loader_circle, train_data_loader_bifurcation, test_data_loader_bifurcation  = create_data_loader_NN1(data_files, batch_size=batch_size)
    print(f"Number of batches in training (circle): {len(train_data_loader_circle) if train_data_loader_circle is not None else 0}")
    print(f"Number of batches in testing (circle): {len(test_data_loader_circle) if test_data_loader_circle is not None else 0}")
    print(f"Number of batches in training (bifurcation): {len(train_data_loader_bifurcation) if train_data_loader_bifurcation is not None else 0}")
    print(f"Number of batches in testing (bifurcation): {len(test_data_loader_bifurcation) if test_data_loader_bifurcation is not None else 0}")
    # Read AD1 model
    encoder_shared = SharedEncoder(feature_num=3)
    AD1_model_circle = GNN_Autoencoder_circle(encoder_shared)
    AD1_model_bifurcation = GNN_Autoencoder_bifurcation(encoder_shared)
    if ENABLE_GPU:
        AD1_model_circle.to("cuda")
        AD1_model_bifurcation.to("cuda")
    AD1_model_circle.load_state_dict(torch.load("model/AD1_model_circle.pth", weights_only=True))
    AD1_model_bifurcation.load_state_dict(torch.load("model/AD1_model_bifurcation.pth", weights_only=True))
    AD1_model_circle.eval()
    AD1_model_bifurcation.eval()
    # Create model
    NN1_model_circle = MultiLayerNet(3, [128,128,128,128], 3)
    NN1_model_bifurcation = MultiLayerNet(3, [128,128,128,128], 3)
    if ENABLE_GPU:
        NN1_model_circle.to("cuda")
        NN1_model_bifurcation.to("cuda")
    # Create optimizer and loss function
    optimizer_circle = torch.optim.Adam(NN1_model_circle.parameters(), lr=1E-4)
    optimizer_bifurcation = torch.optim.Adam(NN1_model_bifurcation.parameters(), lr=1E-4)
    loss_fn = torch.nn.MSELoss()
    # Train model
    best_circle_state = Model_training_NN1(NN1_model_circle, AD1_model_circle, optimizer_circle, loss_fn, train_data_loader_circle)
    best_bifurcation_state = Model_training_NN1(NN1_model_bifurcation, AD1_model_bifurcation, optimizer_bifurcation, loss_fn, train_data_loader_bifurcation)
    # Save model
    torch.save(best_circle_state, "model/NN1_model_circle.pth")
    torch.save(best_bifurcation_state, "model/NN1_model_bifurcation.pth")
    # Calculate loss for testing
    NN1_model_circle.load_state_dict(best_circle_state)
    NN1_model_bifurcation.load_state_dict(best_bifurcation_state)
    NN1_model_circle.eval()
    NN1_model_bifurcation.eval()
    test_error_circle_mae = calculate_error_NN1(NN1_model_circle, AD1_model_circle, test_data_loader_circle, Mean_Absolute_Error)
    test_error_circle_mse = calculate_error_NN1(NN1_model_circle, AD1_model_circle, test_data_loader_circle, Mean_Squared_Error)
    test_error_circle_mre = calculate_error_NN1(NN1_model_circle, AD1_model_circle, test_data_loader_circle, Mean_Relative_Error)
    test_error_bifurcation_mae = calculate_error_NN1(NN1_model_bifurcation, AD1_model_bifurcation, test_data_loader_bifurcation, Mean_Absolute_Error)
    test_error_bifurcation_mse = calculate_error_NN1(NN1_model_bifurcation, AD1_model_bifurcation, test_data_loader_bifurcation, Mean_Squared_Error)
    test_error_bifurcation_mre = calculate_error_NN1(NN1_model_bifurcation, AD1_model_bifurcation, test_data_loader_bifurcation, Mean_Relative_Error)
    print("Circle NN1 Results:")
    print(f"Mean Relative Error: {test_error_circle_mre*100:.2f}%")
    print(f"Mean Absolute Error: {test_error_circle_mae}")
    print(f"Mean Squared Error: {test_error_circle_mse}")
    print("Bifurcation NN1 Results:")
    print(f"Mean Relative Error: {test_error_bifurcation_mre:.2f}%")
    print(f"Mean Absolute Error: {test_error_bifurcation_mae}")
    print(f"Mean Squared Error: {test_error_bifurcation_mse}")
    # Calculate error for AD1 + NN1 in testing set
    test_error_circle_mae = calculate_error_NN1_AD1(NN1_model_circle, AD1_model_circle, test_data_loader_circle, Mean_Absolute_Error)
    test_error_circle_mse = calculate_error_NN1_AD1(NN1_model_circle, AD1_model_circle, test_data_loader_circle, Mean_Squared_Error)
    test_error_circle_mre = calculate_error_NN1_AD1(NN1_model_circle, AD1_model_circle, test_data_loader_circle, Mean_Relative_Error)
    test_error_bifurcation_mae = calculate_error_NN1_AD1(NN1_model_bifurcation, AD1_model_bifurcation, test_data_loader_bifurcation, Mean_Absolute_Error)
    test_error_bifurcation_mse = calculate_error_NN1_AD1(NN1_model_bifurcation, AD1_model_bifurcation, test_data_loader_bifurcation, Mean_Squared_Error)
    test_error_bifurcation_mre = calculate_error_NN1_AD1(NN1_model_bifurcation, AD1_model_bifurcation, test_data_loader_bifurcation, Mean_Relative_Error)
    print("Circle AD1 + NN1 Results:")
    print(f"Mean Relative Error: {test_error_circle_mre*100:.2f}%")
    print(f"Mean Absolute Error: {test_error_circle_mae}")
    print(f"Mean Squared Error: {test_error_circle_mse}")
    print("Bifurcation AD1 + NN1 Results:")
    print(f"Mean Relative Error: {test_error_bifurcation_mre:.2f}%")
    print(f"Mean Absolute Error: {test_error_bifurcation_mae}")
    print(f"Mean Squared Error: {test_error_bifurcation_mse}")
    

if __name__ == "__main__":
    main()