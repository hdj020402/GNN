import torch
import torch.nn.functional as F
from torch.nn import GRU, Linear, ReLU, Sequential

from torch_geometric.nn import NNConv
from torch_geometric.nn import Set2Set
from torch_geometric.utils import remove_self_loops, to_networkx


class mpnn_basic(torch.nn.Module):
    def __init__(self, dataset, dim):
        super().__init__()
        # linear layer transform node feature
        self.lin0 = torch.nn.Linear(dataset.num_features, dim)

        # linear multi layer transform edge feature
        nn = Sequential(Linear(dataset.num_edge_features, dim),
                               ReLU(),
                               Linear(dim, dim * dim))
        # convolutional layer (Message passing)
        self.conv = NNConv(dim, dim, nn, aggr='mean')
        
        self.gru = GRU(dim, dim)

        # Set2Set是一种将集合到集合用LSTM时序模型进行映射的聚合方式，注意out_channels = 2 * in_channels
        self.set2set = Set2Set(dim, processing_steps=3)
        self.lin1 = torch.nn.Linear(2 * dim, dim)
        self.lin2 = torch.nn.Linear(dim, 1)

    def forward(self, data):
        out = F.relu(self.lin0(data.x))
        h = out.unsqueeze(0)

        for i in range(3):
            m = F.relu(self.conv(out, data.edge_index, data.edge_attr))
            out, h = self.gru(m.unsqueeze(0), h)
            out = out.squeeze(0)

        out = self.set2set(out, data.batch)
        out = F.relu(self.lin1(out))
        out = self.lin2(out)
        return out.view(-1)