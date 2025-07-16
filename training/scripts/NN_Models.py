import torch

class MultiLayerNet(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim_ls, output_dim):
        super(MultiLayerNet, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim_ls = hidden_dim_ls
        self.output_dim = output_dim
        self.layers = []
        for i in range(len(hidden_dim_ls)):
            if i == 0:
                self.layers.append(torch.nn.Linear(self.input_dim, self.hidden_dim_ls[i]))
            else:
                self.layers.append(torch.nn.Linear(self.hidden_dim_ls[i-1], self.hidden_dim_ls[i]))
            self.layers.append(torch.nn.ReLU())
        self.layers.append(torch.nn.Linear(self.hidden_dim_ls[-1], self.output_dim))
        self.net = torch.nn.Sequential(*self.layers)
    def forward(self, x):
        return self.net(x)