"""Graph Positional/Structural Encoding Transformer (GPS) backbone.

GPS = Local message passing + Global self-attention.
Reference: "Recipe for a General, Powerful, Scalable Graph Transformer"
           Rampasek et al. 2022.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Linear, LayerNorm

from torch_geometric.nn import GPSConv, GINConv

from models.backbones.base import BackboneBase


class GPSBackbone(BackboneBase):
    """GPS backbone combining local GNN layers with global self-attention.

    Each GPS layer contains:
      1. Local message passing (GINConv by default, same as original GPS paper).
      2. Global multi-head self-attention over all nodes in the batch.
    This allows the model to capture both local structural information and
    long-range interactions.

    **On attention mechanism differences vs. GAT / TransformerConv:**

    - **GATv2**: local-only attention. Attention weight between (i,j) is
      computed from ``h_i`` and ``h_j`` (neighbors only). One shared attention
      vector.
    - **TransformerConv**: local-only attention. Uses separate K/Q/V projections
      (transformer-style dot-product attention) but only over direct neighbors.
      Supports edge features in the key computation.
    - **GPSConv**: combines a local GNN (arbitrary message passing) with
      *full global* self-attention over all nodes in the graph. Global attention
      captures non-local dependencies that neither GAT nor TransformerConv can
      express in a single layer.

    **On positional encodings (PE):**
    Without PE, node identities are ambiguous in symmetric graphs (e.g., all
    nodes look identical to the global attention). For full GPS expressiveness,
    pre-compute RWSE or LapPE and concatenate them to ``data.x`` before
    calling this backbone. With our molecular SDF input the richness of atom
    features (element type, connectivity) provides implicit positional
    discrimination, so PE is optional in practice.

    Args:
        dataset: PyG dataset, used to infer ``num_features``.
        node_dim: Feature dimension throughout all GPS layers.
        num_layers: Number of GPSConv layers.
        heads: Number of attention heads for the global attention block.
        attn_type: ``'multihead'`` (exact, O(n²)) or ``'performer'``
            (approximate, O(n)) for the global attention.
        dropout: Dropout rate applied inside GPSConv.
    """

    def __init__(self, dataset, node_dim: int = 64, num_layers: int = 3,
                 heads: int = 4, attn_type: str = 'multihead',
                 dropout: float = 0.0):
        super().__init__()
        self._node_dim = node_dim

        self.lin_in = Linear(dataset.num_features, node_dim)
        self.norms = nn.ModuleList()  # independent LN per GPS layer
        self.layers = nn.ModuleList()

        for _ in range(num_layers):
            local_conv = GINConv(nn.Sequential(
                Linear(node_dim, node_dim),
                nn.ReLU(),
                Linear(node_dim, node_dim),
            ))
            self.layers.append(GPSConv(
                channels=node_dim,
                conv=local_conv,
                heads=heads,
                dropout=dropout,
                act='relu',
                norm='layer_norm',   # LayerNorm is better than BatchNorm for GNNs
                attn_type=attn_type,
            ))
            self.norms.append(LayerNorm(node_dim))  # additional norm after each GPS layer

    def forward(self, data) -> torch.Tensor:
        x = F.relu(self.lin_in(data.x))
        for layer, norm in zip(self.layers, self.norms):
            x = norm(layer(x, data.edge_index, data.batch))
        return x

    @property
    def output_dim(self) -> int:
        return self._node_dim
