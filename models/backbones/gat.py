"""Graph Attention Network backbone (GATv2)."""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import LayerNorm, Dropout

from torch_geometric.nn import GATv2Conv

from models.backbones.base import BackboneBase


class GATBackbone(BackboneBase):
    """GAT backbone using GATv2Conv (Brody et al. 2021).

    Uses GATv2Conv rather than the original GATConv for improved expressiveness.
    All layers use concat=True with per_head_dim = node_dim // heads, so the
    output dimension is consistently node_dim throughout.

    Edge features are incorporated when the dataset has edge attributes
    (use_edge_features=True), which is important for molecular graphs.

    Args:
        dataset: PyG dataset, used to infer num_features and num_edge_features.
        node_dim: Total output node feature dimension. Must be divisible by heads.
        num_layers: Number of GATv2Conv layers.
        heads: Number of attention heads. node_dim must be divisible by heads.
        use_edge_features: Whether to pass edge_attr into GATv2Conv.
        use_layer_norm: Whether to apply LayerNorm after each layer.
        use_dropout: Whether to apply Dropout on node features between layers.
        dropout_p: Dropout probability for node features.
        attn_dropout: Dropout probability on attention coefficients (inside GATv2Conv).
    """

    def __init__(self, dataset, node_dim: int = 64, num_layers: int = 3,
                 heads: int = 4, use_edge_features: bool = True,
                 use_layer_norm: bool = True, use_dropout: bool = False,
                 dropout_p: float = 0.5, attn_dropout: float = 0.0):
        super().__init__()
        assert node_dim % heads == 0, f"node_dim ({node_dim}) must be divisible by heads ({heads})"
        self._node_dim = node_dim
        per_head_dim = node_dim // heads

        edge_dim = dataset.num_edge_features if use_edge_features else None

        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList() if use_layer_norm else None
        self.dropout = Dropout(dropout_p) if use_dropout else None

        in_dim = dataset.num_features
        for _ in range(num_layers):
            self.convs.append(GATv2Conv(
                in_dim, per_head_dim,
                heads=heads, concat=True,
                dropout=attn_dropout,
                edge_dim=edge_dim,
            ))
            if use_layer_norm:
                self.norms.append(LayerNorm(node_dim))
            in_dim = node_dim  # concat=True: output is per_head_dim * heads = node_dim

    def forward(self, data) -> torch.Tensor:
        x, edge_index = data.x, data.edge_index
        edge_attr = data.edge_attr if hasattr(data, 'edge_attr') and data.edge_attr is not None else None

        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index, edge_attr=edge_attr)
            if self.norms is not None:
                x = self.norms[i](x)
            if i < len(self.convs) - 1:
                x = F.elu(x)
                if self.dropout is not None:
                    x = self.dropout(x)
        return x

    @property
    def output_dim(self) -> int:
        return self._node_dim
