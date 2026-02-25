"""Graph Convolutional Network (GCN) backbone."""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Linear, LayerNorm, Dropout

from torch_geometric.nn import GCNConv

from models.backbones.base import BackboneBase


class GCNBackbone(BackboneBase):
    """GCN backbone (Kipf & Welling 2017).

    Stacks GCNConv layers with optional independent LayerNorm per layer and Dropout.
    Each layer has its own LayerNorm instance with separate learned scale/bias,
    which is correct for stacked (non-recurrent) architectures where each layer
    processes differently distributed features.

    Note: GCN does not use edge features.

    Args:
        dataset: PyG dataset, used to infer num_features.
        node_dim: Dimension of all hidden and output node features.
        num_layers: Number of GCNConv layers.
        use_layer_norm: Whether to apply independent LayerNorm after each layer.
        use_dropout: Whether to apply Dropout after each non-final layer.
        dropout_p: Dropout probability.
    """

    def __init__(self, dataset, node_dim: int = 64, num_layers: int = 3,
                 use_layer_norm: bool = True, use_dropout: bool = False,
                 dropout_p: float = 0.5):
        super().__init__()
        self._node_dim = node_dim

        self.convs = nn.ModuleList()
        # Independent LayerNorm per layer — each layer learns its own scale/shift.
        self.norms = nn.ModuleList(
            [LayerNorm(node_dim) for _ in range(num_layers)]
        ) if use_layer_norm else None
        self.dropout = Dropout(dropout_p) if use_dropout else None

        in_dim = dataset.num_features
        for _ in range(num_layers):
            self.convs.append(GCNConv(in_dim, node_dim))
            in_dim = node_dim

    def forward(self, data) -> torch.Tensor:
        x = data.x
        for i, conv in enumerate(self.convs):
            x = conv(x, data.edge_index)
            if self.norms is not None:
                x = self.norms[i](x)
            if i < len(self.convs) - 1:
                x = F.relu(x)
                if self.dropout is not None:
                    x = self.dropout(x)
        return x

    @property
    def output_dim(self) -> int:
        return self._node_dim
