"""Graph Convolutional Network (GCN), GAT, and GIN backbones."""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Linear, ReLU, LayerNorm, Dropout

from torch_geometric.nn import GCNConv, GATConv, GINConv

from models.backbones.base import BackboneBase


class GCNBackbone(BackboneBase):
    """Graph Convolutional Network backbone.

    Args:
        dataset: PyG dataset
        hidden_dim: Hidden dimension for all layers
        num_layers: Number of GCN layers
        use_dropout: Whether to use dropout
        dropout_p: Dropout probability
    """

    def __init__(self, dataset, hidden_dim: int = 64, num_layers: int = 3,
                 use_dropout: bool = False, dropout_p: float = 0.5):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        layers = []
        # First layer: num_features -> hidden_dim
        layers.append(GCNConv(dataset.num_features, hidden_dim))
        layers.append(LayerNorm(hidden_dim))
        layers.append(ReLU())
        if use_dropout:
            layers.append(Dropout(dropout_p))

        # Hidden layers
        for _ in range(num_layers - 2):
            layers.append(GCNConv(hidden_dim, hidden_dim))
            layers.append(LayerNorm(hidden_dim))
            layers.append(ReLU())
            if use_dropout:
                layers.append(Dropout(dropout_p))

        # Last layer
        layers.append(GCNConv(hidden_dim, hidden_dim))
        layers.append(LayerNorm(hidden_dim))

        self.layers = nn.ModuleList(layers)

    def forward(self, data) -> torch.Tensor:
        x = data.x
        for layer in self.layers:
            if isinstance(layer, GCNConv):
                x = layer(x, data.edge_index)
            else:
                x = layer(x)
        return x

    @property
    def output_dim(self) -> int:
        return self.hidden_dim


class GATBackbone(BackboneBase):
    """Graph Attention Network backbone.

    Args:
        dataset: PyG dataset
        hidden_dim: Hidden dimension for all layers
        num_layers: Number of GAT layers
        heads: Number of attention heads
        use_dropout: Whether to use dropout
        dropout_p: Dropout probability
    """

    def __init__(self, dataset, hidden_dim: int = 64, num_layers: int = 3,
                 heads: int = 4, use_dropout: bool = False, dropout_p: float = 0.5):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.heads = heads

        layers = []
        # First layer: num_features -> hidden_dim
        layers.append(GATConv(dataset.num_features, hidden_dim, heads=heads, dropout=dropout_p))
        layers.append(LayerNorm(hidden_dim * heads))
        layers.append(ReLU())

        # Hidden layers
        for _ in range(num_layers - 2):
            layers.append(GATConv(hidden_dim * heads, hidden_dim, heads=heads, dropout=dropout_p))
            layers.append(LayerNorm(hidden_dim * heads))
            layers.append(ReLU())

        # Last layer - return to hidden_dim dimension
        layers.append(GATConv(hidden_dim * heads, hidden_dim, heads=1, dropout=dropout_p))
        layers.append(LayerNorm(hidden_dim))

        self.layers = nn.ModuleList(layers)

    def forward(self, data) -> torch.Tensor:
        x = data.x
        for layer in self.layers:
            if isinstance(layer, GATConv):
                x = layer(x, data.edge_index)
            else:
                x = layer(x)
        return x

    @property
    def output_dim(self) -> int:
        return self.hidden_dim


class GINBackbone(BackboneBase):
    """Graph Isomorphism Network backbone.

    Args:
        dataset: PyG dataset
        hidden_dim: Hidden dimension for all layers
        num_layers: Number of GIN layers
        use_dropout: Whether to use dropout
        dropout_p: Dropout probability
    """

    def __init__(self, dataset, hidden_dim: int = 64, num_layers: int = 3,
                 use_dropout: bool = False, dropout_p: float = 0.5):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        layers = []
        # First layer: num_features -> hidden_dim
        mlp1 = nn.Sequential(
            Linear(dataset.num_features, hidden_dim),
            ReLU(),
            Linear(hidden_dim, hidden_dim)
        )
        layers.append(GINConv(mlp1))
        layers.append(LayerNorm(hidden_dim))
        if use_dropout:
            layers.append(Dropout(dropout_p))

        # Hidden layers
        for _ in range(num_layers - 1):
            mlp = nn.Sequential(
                Linear(hidden_dim, hidden_dim),
                ReLU(),
                Linear(hidden_dim, hidden_dim)
            )
            layers.append(GINConv(mlp))
            layers.append(LayerNorm(hidden_dim))
            if use_dropout:
                layers.append(Dropout(dropout_p))

        self.layers = nn.ModuleList(layers)

    def forward(self, data) -> torch.Tensor:
        x = data.x
        for layer in self.layers:
            if isinstance(layer, GINConv):
                x = layer(x, data.edge_index)
            else:
                x = layer(x)
        return x

    @property
    def output_dim(self) -> int:
        return self.hidden_dim
