"""Graph Transformer backbones: TransformerConv and GPSConv."""
from typing import Literal
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Linear, LayerNorm, Dropout

from torch_geometric.nn import TransformerConv, GPSConv, GINConv

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
        self.layer_norm = LayerNorm(node_dim) if use_layer_norm else None
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


class GPSBackbone(BackboneBase):
    """GPS backbone (Rampasek et al. 2022, "Recipe for a General, Powerful,
    Scalable Graph Transformer").

    GPS combines a **local** message-passing GNN with a **global** self-attention
    layer (all nodes attend to all others). This allows it to capture both local
    structural information and long-range interactions.

    Differences vs. TransformerConv / GAT:
      - GAT/TransformerConv: local attention (neighbors only).
      - GPS: local GNN (e.g. GIN) + global attention (all pairs).

    Positional encodings (PE) are strongly recommended for GPS to give each node
    a unique identity, because global attention alone cannot distinguish nodes
    with identical features. If `data.x` already contains PE (e.g. RWSE or
    Laplacian PE concatenated as extra columns), GPS will use them automatically.
    Without PE, GPS degenerates to a less discriminative global-attention model.

    Implementation note: GPSConv wraps any local MessagePassing layer (here
    GINConv) and adds global multihead self-attention. The local and global
    branches are combined via a residual connection.

    Args:
        dataset: PyG dataset.
        node_dim: Node feature dimension throughout the network.
        num_layers: Number of GPSConv layers.
        heads: Number of attention heads for the global attention component.
        local_conv: Which local GNN to use inside GPSConv ('gin' or 'gcn').
            'gin' is the recommended choice for expressiveness.
        use_layer_norm: Whether GPSConv uses LayerNorm internally.
            (GPSConv has its own norm parameter; this controls it.)
        dropout: Dropout rate inside GPSConv.
    """

    def __init__(self, dataset, node_dim: int = 64, num_layers: int = 3,
                 heads: int = 4,
                 local_conv: Literal['gin', 'gcn'] = 'gin',
                 use_layer_norm: bool = True, dropout: float = 0.0):
        super().__init__()
        self._node_dim = node_dim

        # Project input to node_dim before GPS layers
        self.input_proj = Linear(dataset.num_features, node_dim)
        norm = 'layer_norm' if use_layer_norm else None

        self.convs = nn.ModuleList()
        for _ in range(num_layers):
            if local_conv == 'gin':
                mlp = nn.Sequential(
                    Linear(node_dim, node_dim), nn.ReLU(),
                    Linear(node_dim, node_dim)
                )
                local = GINConv(mlp)
            else:
                from torch_geometric.nn import GCNConv
                local = GCNConv(node_dim, node_dim)

            self.convs.append(GPSConv(
                node_dim, local,
                heads=heads,
                dropout=dropout,
                norm=norm,
            ))

    def forward(self, data) -> torch.Tensor:
        x = F.relu(self.input_proj(data.x))

        for conv in self.convs:
            x = conv(x, data.edge_index, data.batch)

        return x

    @property
    def output_dim(self) -> int:
        return self._node_dim
