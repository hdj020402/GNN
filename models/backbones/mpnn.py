"""Message Passing Neural Network (MPNN) backbone."""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import GRU, Linear, ReLU, Sequential, Dropout, LayerNorm

from torch_geometric.nn import NNConv

from models.backbones.base import BackboneBase


class MPNNBackbone(BackboneBase):
    """MPNN backbone using NNConv with edge features and GRU for temporal modeling.

    This backbone uses message passing with edge features and GRU cells for temporal
    modeling. Each message passing iteration is followed by layer normalization.

    Args:
        dataset: PyG dataset (used to get feature dimensions)
        dim_linear: Hidden dimension for the MLP in NNConv
        dim_conv: Convolution dimension (feature dimension after first layer)
        mp_times: Number of message passing iterations
        processing_steps: Number of steps for Set2Set aggregation
    """

    def __init__(self, dataset, dim_linear: int, dim_conv: int, mp_times: int,
                 processing_steps: int = 1, use_dropout: bool = False):
        super().__init__()
        self.dim_conv = dim_conv
        self.mp_times = mp_times
        self.use_dropout = use_dropout

        # Initial linear layer to project node features to dim_conv
        self.lin0 = Linear(dataset.num_features, dim_conv)

        # NNConv with MLP for edge-dependent weight generation
        nn = Sequential(
            Linear(dataset.num_edge_features, dim_linear),
            ReLU(),
            Linear(dim_linear, dim_conv * dim_conv)
        )
        self.conv = NNConv(dim_conv, dim_conv, nn, aggr='mean')
        self.gru = GRU(dim_conv, dim_conv)

        # Layer normalization for each message passing step
        self.layer_norms = nn.ModuleList([LayerNorm(dim_conv) for _ in range(mp_times)])

        if use_dropout:
            self.dropout = Dropout(0.5)
        else:
            self.dropout = None

    def forward(self, data) -> torch.Tensor:
        """Forward pass for MPNN backbone.

        Args:
            data: PyG Data object

        Returns:
            torch.Tensor: Node features with shape [num_nodes, dim_conv]
        """
        out = F.relu(self.lin0(data.x))
        h = out.unsqueeze(0)

        for i in range(self.mp_times):
            m = F.relu(self.conv(out, data.edge_index, data.edge_attr))
            out, h = self.gru(m.unsqueeze(0), h)
            out = out.squeeze(0)
            out = self.layer_norms[i](out)
            if self.dropout is not None:
                out = self.dropout(out)

        return out

    @property
    def output_dim(self) -> int:
        return self.dim_conv
