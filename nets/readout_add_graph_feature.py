import torch
import torch.nn.functional as F
from torch.nn import GRU, Linear, ReLU, Sequential, Dropout

from torch_geometric.nn import NNConv
from torch_geometric.nn import Set2Set
from torch_geometric.utils import remove_self_loops, to_networkx


class GraphPredictionModel(torch.nn.Module):
    def __init__(self, dataset, dim_linear, dim_conv, dim_output, processing_steps, mp_times):
        super().__init__()
        self.lin0 = torch.nn.Linear(dataset.num_features, dim_conv)

        nn = Sequential(Linear(dataset.num_edge_features, dim_linear),
                               ReLU(),
                               #Dropout(p=0.5),
                               Linear(dim_linear, dim_conv * dim_conv))
        self.conv = NNConv(dim_conv, dim_conv, nn, aggr='mean')
        self.gru = GRU(dim_conv, dim_conv)

        # Set2Set是一种将集合到集合用LSTM时序模型进行映射的聚合方式，注意out_channels = 2 * in_channels
        self.set2set = Set2Set(dim_conv, processing_steps = processing_steps)
        self.lin1 = torch.nn.Linear(2 * dim_conv+dataset.num_graph_features, dim_conv)
        self.lin2 = torch.nn.Linear(dim_conv, dim_output)
        self.mp_times = mp_times

    def forward(self, data):
        out = F.relu(self.lin0(data.x))
        h = out.unsqueeze(0)

        for i in range(self.mp_times):
            m = F.relu(self.conv(out, data.edge_index, data.edge_attr))
            out, h = self.gru(m.unsqueeze(0), h)
            out = out.squeeze(0)

        out = self.set2set(out, data.batch)
        out = torch.cat((out, data.graph_attr), dim=1)
        out = F.relu(self.lin1(out))
        out = self.lin2(out)
        return out

class NodePredictionModel(torch.nn.Module):
    def __init__(self, dataset, dim_linear, dim_conv, dim_output, mp_times):
        super(NodePredictionModel, self).__init__()
        self.lin0 = torch.nn.Linear(dataset.num_features, dim_conv)

        nn = Sequential(Linear(dataset.num_edge_features, dim_linear),
                        ReLU(),
                        # Dropout(p=0.5),
                        Linear(dim_linear, dim_conv * dim_conv))
        self.conv = NNConv(dim_conv, dim_conv, nn, aggr='mean')
        self.gru = GRU(dim_conv, dim_conv)

        # 修改了线性层以适应节点级别的预测任务
        self.lin1 = torch.nn.Linear(dim_conv, dim_conv)
        self.lin2 = torch.nn.Linear(dim_conv, dim_output)
        self.mp_times = mp_times

    def forward(self, data):
        out = F.relu(self.lin0(data.x))
        h = out.unsqueeze(0)

        for i in range(self.mp_times):
            m = F.relu(self.conv(out, data.edge_index, data.edge_attr))
            out, h = self.gru(m.unsqueeze(0), h)
            out = out.squeeze(0)

        # 移除 set2set 聚合操作，直接使用节点特征进行预测
        out = torch.cat((out, data.graph_attr[data.batch]), dim=-1)  # 连接每个节点和图属性
        out = F.relu(self.lin1(out))
        out = self.lin2(out)
        return out
