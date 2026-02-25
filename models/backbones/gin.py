"""Graph Isomorphism Network (GIN) backbone."""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Linear, ReLU, LayerNorm, Dropout

from torch_geometric.nn import GINConv

from models.backbones.base import BackboneBase


class GINBackbone(BackboneBase):
    """GIN backbone (Xu et al. 2019 "How Powerful are Graph Neural Networks?").

    GIN is the most expressive message-passing GNN in the WL-test sense among
    sum-aggregation models. Each layer uses a 2-layer MLP as the aggregation
    function. Note: GIN does not use edge features.

    Args:
        dataset: PyG dataset, used to infer num_features.
        node_dim: Dimension of all hidden and output node features.
        num_layers: Number of GINConv layers.
        use_layer_norm: Whether to apply shared LayerNorm after each layer.
        use_dropout: Whether to apply Dropout after each non-final layer.
        dropout_p: Dropout probability.
    """

    def __init__(self, dataset, node_dim: int = 64, num_layers: int = 3,
                 use_layer_norm: bool = True, use_dropout: bool = False,
                 dropout_p: float = 0.5):
        super().__init__()
        self._node_dim = node_dim

        self.convs = nn.ModuleList()
        # Single shared LayerNorm — node_dim is constant at every layer.
        self.layer_norm = LayerNorm(node_dim) if use_layer_norm else None
        self.dropout = Dropout(dropout_p) if use_dropout else None

        in_dim = dataset.num_features
        for _ in range(num_layers):
            mlp = nn.Sequential(
                Linear(in_dim, node_dim),
                ReLU(),
                Linear(node_dim, node_dim),
            )
            self.convs.append(GINConv(mlp))
            in_dim = node_dim

    def forward(self, data) -> torch.Tensor:
        x = data.x
        for i, conv in enumerate(self.convs):
            x = conv(x, data.edge_index)
            if self.layer_norm is not None:
                x = self.layer_norm(x)
            if i < len(self.convs) - 1:
                x = F.relu(x)
                if self.dropout is not None:
                    x = self.dropout(x)
        return x

    @property
    def output_dim(self) -> int:
        return self._node_dim
