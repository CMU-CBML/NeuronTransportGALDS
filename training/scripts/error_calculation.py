import torch
from parameter import ENABLE_GPU

# TODO: Move the functions back to AD1 training script
def calculate_error_AD1(model, data_loader, error_fn):
    total_loss = 0
    total_samples = 0
    for data in data_loader:
        if ENABLE_GPU:
            data = data.to("cuda")
        with torch.no_grad():
            output = model(data)
            loss = error_fn(output, data.x.view(-1, model.num_nodes, 3))
            total_loss += loss.item() * data.size(0)
            total_samples += data.size(0)
    return total_loss / total_samples

def Mean_Absolute_Error(output, target):
    return torch.mean(torch.abs(output - target))

def Mean_Squared_Error(output, target):
    return torch.mean((output - target) ** 2)

def Mean_Relative_Error(output, target):
    bot = torch.max(target) - torch.min(target)
    top = (torch.mean((output - target)**2))**0.5
    return top / bot

def calculate_error_NN1(NN1_model, AD1_model, data_loader, error_fn):
    NN1_model.eval()
    AD1_model.eval()
    total_loss = 0
    total_samples = 0
    for data in data_loader:
        if AD1_model.num_nodes == 17:
            num_of_surfaces = int(data.x.shape[0] / 17)
            indexs = [i*17 for i in range(num_of_surfaces)]
        elif AD1_model.num_nodes == 23:
            num_of_bifurcations = int(data.x.shape[0] / 23)
            indexs = [i*23 for i in range(num_of_bifurcations)]
        if ENABLE_GPU:
            data = data.to("cuda")
        with torch.no_grad():
            output = NN1_model(data.x[indexs])
            target = AD1_model.encode(data.x, data.edge_index, data.batch)
            loss = error_fn(output, target)
            total_loss += loss.item() * data.size(0)
            total_samples += data.size(0)
    return total_loss / total_samples

# TODO: Move the functions back to NN1 training script
def calculate_error_NN1_AD1(NN1_model, AD1_model, data_loader, error_fn):
    NN1_model.eval()
    AD1_model.eval()
    total_loss = 0
    total_samples = 0
    for data in data_loader:
        if AD1_model.num_nodes == 17:
            num_of_surfaces = int(data.x.shape[0] / 17)
            indexs = [i*17 for i in range(num_of_surfaces)]
        elif AD1_model.num_nodes == 23:
            num_of_bifurcations = int(data.x.shape[0] / 23)
            indexs = [i*23 for i in range(num_of_bifurcations)]
        if ENABLE_GPU:
            data = data.to("cuda")
        with torch.no_grad():
            NN1_result = NN1_model(data.x[indexs])
            output = AD1_model.decode(NN1_result, data.edge_index, data.batch)
            target = data.x.view(-1, AD1_model.num_nodes, 3)
            loss = error_fn(output, target)
            total_loss += loss.item() * data.size(0)
            total_samples += data.size(0)
    return total_loss / total_samples


# TODO: Move this function back to AD2 training script
def calculate_error_AD2(model, data_loader, error_fn):
    total_loss = 0
    total_samples = 0
    for data in data_loader:
        if ENABLE_GPU:
            data = data.to("cuda")
        with torch.no_grad():
            output = model(data)
            loss = error_fn(output, data.x.view(-1, model.num_nodes, model.feature_num))
            total_loss += loss.item() * data.size(0)
            total_samples += data.size(0)
    return total_loss / total_samples

# TODO: Remove this if not used
def calculate_error_NN2(model_NN2, model_AD1, model_AD2, data_loader_t, data_loader_t1, data_loader_u, error_fn):
    if len(data_loader_t) == 1:
        data_loader_t_new = []
        for data in data_loader_t:
            data = data.to("cuda")
            data_loader_t_new.append(data)
        data_loader_t = data_loader_t_new
    if len(data_loader_t1) == 1:
        data_loader_t1_new = []
        for data in data_loader_t1:
            data = data.to("cuda")
            data_loader_t1_new.append(data)
        data_loader_t1 = data_loader_t1_new
    if len(data_loader_u) == 1:
        data_loader_u_new = []
        for data in data_loader_u:
            data = data.to("cuda")
            data_loader_u_new.append(data)
        data_loader_u = data_loader_u_new
    model_NN2.eval()
    model_AD1.eval()
    model_AD2.eval()
    total_loss = 0
    total_samples = 0
    for data_t, data_t1, data_u in zip(data_loader_t, data_loader_t1, data_loader_u):
        if ENABLE_GPU:
            data_t = data_t.to("cuda")
            data_t1 = data_t1.to("cuda")
        with torch.no_grad():
            n_latent = model_AD2.encode(data_t.x, data_t.edge_index, data_t.batch)
            u_latent = model_AD1.encode(data_u.x, data_u.edge_index, data_u.batch)
            latent = torch.hstack((n_latent, u_latent))
            output_t1 = model_NN2(latent)
            target = model_AD2.encode(data_t1.x, data_t1.edge_index, data_t1.batch)
            loss = error_fn(output_t1, target)
            total_loss += loss.item() * data_t.size(0)
            total_samples += data_t.size(0)
    return total_loss / total_samples

# TODO: Remove this if not used
def calculate_error_NN2_AD2(model_NN2, model_AD1, model_AD2, data_loader_t, data_loader_t1, data_loader_u, error_fn):
    if len(data_loader_t) == 1:
        data_loader_t_new = []
        for data in data_loader_t:
            data = data.to("cuda")
            data_loader_t_new.append(data)
        data_loader_t = data_loader_t_new
    if len(data_loader_t1) == 1:
        data_loader_t1_new = []
        for data in data_loader_t1:
            data = data.to("cuda")
            data_loader_t1_new.append(data)
        data_loader_t1 = data_loader_t1_new
    if len(data_loader_u) == 1:
        data_loader_u_new = []
        for data in data_loader_u:
            data = data.to("cuda")
            data_loader_u_new.append(data)
        data_loader_u = data_loader_u_new
    model_NN2.eval()
    model_AD1.eval()
    model_AD2.eval()
    total_loss = 0
    total_samples = 0
    for data_t, data_t1, data_u in zip(data_loader_t, data_loader_t1, data_loader_u):
        if ENABLE_GPU:
            data_t = data_t.to("cuda")
            data_t1 = data_t1.to("cuda")
        with torch.no_grad():
            n_latent = model_AD2.encode(data_t.x, data_t.edge_index, data_t.batch)
            u_latent = model_AD1.encode(data_u.x, data_u.edge_index, data_u.batch)
            latent = torch.hstack((n_latent, u_latent))
            output_latent = model_NN2(latent)
            physical_prediction = model_AD2.decode(output_latent, data_t.edge_index, data_t.batch)
            target = data_t1.x.view(-1, model_AD2.num_nodes, model_AD2.feature_num)
            loss = error_fn(physical_prediction, target)
            total_loss += loss.item() * data_t.size(0)
            total_samples += data_t.size(0)
    return total_loss / total_samples