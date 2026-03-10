"""Message Passing Neural Network (MPNN) backbone."""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import GRU, Linear, ReLU, Sequential, Dropout, LayerNorm

from torch_geometric.nn import NNConv

from models.backbones.base import BackboneBase


class MPNNBackbone(BackboneBase):
    """MPNN backbone using NNConv with edge features and GRU for temporal modeling.

    The architecture follows: linear projection → message passing × mp_times,
    where each step applies NNConv (edge-conditioned convolution), GRU, and
    optionally LayerNorm.

    Args:
        dataset: PyG dataset, used to infer num_features and num_edge_features.
        node_dim: Node feature dimension used throughout the network.
            The initial node features are projected to this dimension, and all
            subsequent layers maintain this dimension.
        edge_nn_dim: Hidden dimension of the MLP inside NNConv that maps edge
            features to the per-node convolution weights.
        mp_times: Number of message passing iterations.
        use_layer_norm: Whether to apply LayerNorm after each GRU step.
        use_dropout: Whether to apply Dropout after each GRU step.
        dropout_p: Dropout probability (only used when use_dropout=True).
    """

    def __init__(self, dataset, node_dim: int = 64, edge_nn_dim: int = 128,
                 mp_times: int = 3, use_layer_norm: bool = True,
                 use_dropout: bool = False, dropout_p: float = 0.5):
        super().__init__()
        self._node_dim = node_dim
        self.mp_times = mp_times

        # Project node features to node_dim
        self.lin0 = Linear(dataset.num_features, node_dim)

        # NNConv: the edge MLP maps edge features to a [node_dim, node_dim] weight matrix
        edge_nn = Sequential(
            Linear(dataset.num_edge_features, edge_nn_dim),
            ReLU(),
            Linear(edge_nn_dim, node_dim * node_dim)
        )
        self.conv = NNConv(node_dim, node_dim, edge_nn, aggr='mean')
        self.gru = GRU(node_dim, node_dim)

        # Single shared LayerNorm — valid because node_dim is the same at every step.
        # Using one shared instance is simpler and reduces parameters; each step
        # re-uses the same learned scale/bias.
        self.layer_norm = LayerNorm(node_dim) if use_layer_norm else None
        self.dropout = Dropout(dropout_p) if use_dropout else None

    def forward(self, data) -> torch.Tensor:
        """Forward pass.

        Args:
            data: PyG Data object with x, edge_index, edge_attr.

        Returns:
            torch.Tensor: Node features [num_nodes, node_dim].
        """
        out = F.relu(self.lin0(data.x))
        h = out.unsqueeze(0)

        for _ in range(self.mp_times):
            m = F.relu(self.conv(out, data.edge_index, data.edge_attr))
            out, h = self.gru(m.unsqueeze(0), h)
            out = out.squeeze(0)
            if self.layer_norm is not None:
                out = self.layer_norm(out)
            if self.dropout is not None:
                out = self.dropout(out)

        return out

    @property
    def output_dim(self) -> int:
        return self._node_dim
