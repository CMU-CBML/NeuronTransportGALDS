import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool

edge_index_circle = torch.tensor([
    [0, 0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5, 6, 6, 6, 7, 7, 7, 7, 8, 8, 8, 9, 9, 9, 10, 10, 10, 11, 11, 11, 12, 12, 12, 12, 13, 13, 13, 14, 14, 14, 15, 15, 15, 16, 16, 16],
    [4, 5, 7, 12, 4, 6, 13, 5, 6, 8, 7, 8, 14, 0, 1, 9, 15, 0, 2, 9, 10, 1, 2, 9, 0, 3, 10, 16, 2, 3, 10, 4, 5, 6, 5, 7, 8, 12, 13, 14, 0, 11, 15, 16, 1, 11, 15, 3, 11, 16, 4, 12, 13, 12, 14, 7]], 
    dtype=torch.long)

edge_index_bifurcation = torch.tensor([
    [0, 0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5, 6, 6, 6, 7, 7, 7, 7, 8, 8, 8, 9, 9, 9, 10, 10, 10, 11, 11, 11, 12, 12, 12, 12, 13, 13, 13, 14, 14, 14, 15, 15, 15, 16, 16, 16, 0, 4, 7, 1, 3, 17, 17, 17, 18, 18, 18, 18, 19, 19, 19, 20, 20, 20, 21, 21, 21, 22, 22, 22],
    [4, 5, 7, 12, 4, 6, 13, 5, 6, 8, 7, 8, 14, 0, 1, 9, 15, 0, 2, 9, 10, 1, 2, 9, 0, 3, 10, 16, 2, 3, 10, 4, 5, 6, 5, 7, 8, 12, 13, 14, 0, 11, 15, 16, 1, 11, 15, 3, 11, 16, 4, 12, 13, 12, 14, 7, 18, 21, 22, 19, 20, 18, 19, 20, 0, 17, 21, 22, 1, 17, 21, 3, 17, 22, 4, 18, 19, 7, 18, 20]], 
    dtype=torch.long)
class SharedEncoder(torch.nn.Module):
    def __init__(self, feature_num=3):
        super(SharedEncoder, self).__init__()
        self.feature_num = feature_num
        
        # Encoder: GNN layers
        self.conve1 = GCNConv(self.feature_num, 32)
        self.conve2 = GCNConv(32, 16)
        self.conve3 = GCNConv(16, 8)
        
        # Fully Connected to Latent Space (single 3D vector)
        self.fc_latent = torch.nn.Linear(8, self.feature_num)

    def forward(self, x, edge_index, batch):
        x = F.relu(self.conve1(x, edge_index))
        x = F.relu(self.conve2(x, edge_index))
        x = F.relu(self.conve3(x, edge_index))

        # Global pooling to get a single vector
        x = global_mean_pool(x, batch)  # Shape: [batch_size, 8]
        x = self.fc_latent(x)  # Shape: [batch_size, feature_num]
        # x = F.sigmoid(x)  # Shape: [batch_size, feature_num]
        return x

class GNN_Autoencoder_circle(torch.nn.Module):
    def __init__(self, encoder, feature_num=None):
        super(GNN_Autoencoder_circle, self).__init__()
        self.num_nodes = 17
        if feature_num is None:
            self.feature_num = encoder.feature_num
        else:
            self.feature_num = feature_num
        self.encoder = encoder
        
        # Fully Connected to Expand Latent Vector back to Nodes
        self.fc_expand = torch.nn.Linear(self.feature_num, self.num_nodes * 8)

        # Decoder: GNN layers (Using Graph Structure)
        self.convd1 = GCNConv(8, 16)
        self.convd2 = GCNConv(16, 32)
        self.convd3 = GCNConv(32, self.feature_num)
    
    def encode(self, x, edge_index, batch):
        return self.encoder(x, edge_index, batch)  # Use shared encoder

    def decode(self, z, edge_index, batch):
        # Expand single latent vector into node-wise features
        z = self.fc_expand(z)  # Shape: [batch_size, 17 * 8]
        z = z.view(-1, self.num_nodes, 8)  # Reshape to [batch_size, 17, 8]

        # Convert into graph data structure
        z = z.view(-1, 8)  # Flatten for GNN processing

        # Apply GNN layers (Using Graph Structure)
        z = F.relu(self.convd1(z, edge_index))
        z = F.relu(self.convd2(z, edge_index))
        z = self.convd3(z, edge_index)  # Output shape: [num_nodes, 3]
        return z.view(-1, self.num_nodes, self.feature_num)

    def forward(self, data):
        # z = self.encoder(data.x, data.edge_index, data.batch)  # Use shared encoder
        z = self.encode(data.x, data.edge_index, data.batch)  # Use shared encoder
        x_reconstructed = self.decode(z, data.edge_index, data.batch)
        return x_reconstructed

class GNN_Autoencoder_bifurcation(torch.nn.Module):
    def __init__(self, encoder, feature_num=None):
        super(GNN_Autoencoder_bifurcation, self).__init__()
        self.num_nodes = 23
        if feature_num is None:
            self.feature_num = encoder.feature_num
        else:
            self.feature_num = feature_num
        self.encoder = encoder
        
        # Fully Connected to Expand Latent Vector back to Nodes
        self.fc_expand = torch.nn.Linear(self.feature_num, self.num_nodes * 8)

        # Decoder: GNN layers (Using Graph Structure)
        self.convd1 = GCNConv(8, 16)
        self.convd2 = GCNConv(16, 32)
        self.convd3 = GCNConv(32, self.feature_num)
    
    def encode(self, x, edge_index, batch):
        return self.encoder(x, edge_index, batch)

    def decode(self, z, edge_index, batch):
        # Expand single latent vector into node-wise features
        z = self.fc_expand(z)  # Shape: [batch_size, 23 * 8]
        z = z.view(-1, self.num_nodes, 8)  # Reshape to [batch_size, 23, 8]

        # Convert into graph data structure
        z = z.view(-1, 8)  # Flatten for GNN processing

        # Apply GNN layers (Using Graph Structure)
        z = F.relu(self.convd1(z, edge_index))
        z = F.relu(self.convd2(z, edge_index))
        z = self.convd3(z, edge_index)  # Output shape: [num_nodes, 3]
        return z.view(-1, self.num_nodes, self.feature_num)

    def forward(self, data):
        # z = self.encoder(data.x, data.edge_index, data.batch)  # Use shared encoder
        z = self.encode(data.x, data.edge_index, data.batch)  # Use shared encoder
        x_reconstructed = self.decode(z, data.edge_index, data.batch)
        return x_reconstructed

class GNNODEFunc(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels_list, out_channels):
        super().__init__()
        self.convs = torch.nn.ModuleList()
        input_dim = in_channels

        for hidden_dim in hidden_channels_list:
            self.convs.append(GCNConv(input_dim, hidden_dim))
            input_dim = hidden_dim
        
        self.out_conv = GCNConv(input_dim, out_channels)

    def forward(self, t, x):
        x = torch.hstack((x, self.distance_to_root, self.node_physical_distance, self.kplus, self.kprimeplus, self.initial_min, self.initial_max))
        edge_weight = self.edge_weight
        for conv in self.convs:
            x = F.softplus(conv(x, self.edge_index, edge_weight=edge_weight))
        x = self.out_conv(x, self.edge_index)
        return x

    def set_edge_index(self, edge_index):
        self.edge_index = edge_index

    def set_velocity(self, velocity):
        self.velocity = velocity
        self.velocity_magnitude = torch.norm(velocity, dim=1, keepdim=True)
    
    def set_k(self, kplus, kprimeplus):
        self.kplus = kplus
        self.kprimeplus = kprimeplus
    
    def set_distance_to_root(self, distance_to_root):
        self.distance_to_root = distance_to_root
    
    def set_edge_distance(self, edge_distance):
        self.edge_distance = edge_distance
        self.edge_weight = 1 / (edge_distance + 1e-6)
    
    def set_initial_condition(self, initial_condition):
        self.initial_condition = initial_condition
        self.initial_max = torch.ones_like(initial_condition) * torch.max(initial_condition)
        self.initial_min = torch.ones_like(initial_condition) * torch.min(initial_condition)
    
    def set_node_physical_distance(self, node_physical_distance):
        self.node_physical_distance = node_physical_distance