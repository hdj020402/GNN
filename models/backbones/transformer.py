"""Graph Transformer backbone using TransformerConv."""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Linear, LayerNorm, Dropout

from torch_geometric.nn import TransformerConv

from models.backbones.base import BackboneBase


class GraphTransformerBackbone(BackboneBase):
    """Graph Transformer backbone using TransformerConv (Shi et al. 2021).

    TransformerConv uses transformer-style multi-head attention with explicit
    Key/Query/Value projections:
        Q = Wq · hi,  K = Wk · hj + We · e_ij,  V = Wv · hj
    Edge features are incorporated into the key computation, making it well
    suited to molecular graphs where bond type matters.

    This is a purely **local** model — attention is restricted to direct graph
    neighbors. Compare with GPSBackbone which adds a global attention component.

    Args:
        dataset: PyG dataset.
        node_dim: Total node feature dimension. Must be divisible by heads.
        num_layers: Number of TransformerConv layers.
        heads: Number of attention heads.
        use_edge_features: Whether to include edge_attr in attention.
        beta: Whether to learn a gating parameter for the skip connection.
        use_layer_norm: Whether to apply shared LayerNorm after each layer.
        use_dropout: Whether to apply Dropout on node features between layers.
        dropout_p: Dropout probability on node features.
        attn_dropout: Dropout probability on attention coefficients.
    """

    def __init__(self, dataset, node_dim: int = 64, num_layers: int = 3,
                 heads: int = 4, use_edge_features: bool = True,
                 beta: bool = True, use_layer_norm: bool = True,
                 use_dropout: bool = False, dropout_p: float = 0.5,
                 attn_dropout: float = 0.0):
        super().__init__()
        assert node_dim % heads == 0, \
            f"node_dim ({node_dim}) must be divisible by heads ({heads})"
        self._node_dim = node_dim
        per_head_dim = node_dim // heads
        edge_dim = dataset.num_edge_features if use_edge_features else None

        self.convs = nn.ModuleList()
        # Independent LayerNorm per layer — each layer learns its own scale/shift.
        self.norms = nn.ModuleList(
            [LayerNorm(node_dim) for _ in range(num_layers)]
        ) if use_layer_norm else None
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
            in_dim = node_dim

    def forward(self, data) -> torch.Tensor:
        x, edge_index = data.x, data.edge_index
        edge_attr = getattr(data, 'edge_attr', None)

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
