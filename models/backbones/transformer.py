"""Graph Transformer backbone using TransformerConv."""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Linear, LayerNorm, Dropout

from torch_geometric.nn import TransformerConv

from models.backbones.base import BackboneBase


class GraphTransformerBackbone(BackboneBase):
    """Graph Transformer backbone using TransformerConv (Shi et al. 2021).

    TransformerConv is a graph attention layer with transformer-style multi-head
    attention (key/query/value projections), and natively supports edge features
    in the attention computation — important for molecular graphs where bond types
    carry information.

    Note: GPS (Rampasek et al. 2022) is the more powerful variant that adds
    global/local attention with positional encodings, but it requires PyG ≥ 2.7.
    This implementation uses TransformerConv as a practical alternative.

    All layers use concat=True with per_head_dim = node_dim // heads, so the
    output dimension is consistently node_dim throughout.

    Args:
        dataset: PyG dataset, used to infer num_features and num_edge_features.
        node_dim: Total output node feature dimension. Must be divisible by heads.
        num_layers: Number of TransformerConv layers.
        heads: Number of attention heads. node_dim must be divisible by heads.
        use_edge_features: Whether to pass edge features into attention.
        use_layer_norm: Whether to apply LayerNorm after each layer.
        beta: Whether to use beta for skip connections (recommended True).
        use_dropout: Whether to apply Dropout on node features between layers.
        dropout_p: Dropout probability on node features.
        attn_dropout: Dropout probability on attention coefficients.
    """

    def __init__(self, dataset, node_dim: int = 64, num_layers: int = 3,
                 heads: int = 4, use_edge_features: bool = True,
                 use_layer_norm: bool = True, beta: bool = True,
                 use_dropout: bool = False, dropout_p: float = 0.5,
                 attn_dropout: float = 0.0):
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
            self.convs.append(TransformerConv(
                in_dim, per_head_dim,
                heads=heads, concat=True,
                beta=beta,
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
                x = F.relu(x)
                if self.dropout is not None:
                    x = self.dropout(x)
        return x

    @property
    def output_dim(self) -> int:
        return self._node_dim
