"""Node-level prediction head."""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Linear, LayerNorm


class NodePredictionHead(nn.Module):
    """Head for node-level prediction.

    Concatenates node features with their corresponding graph-level attributes
    (broadcast to each node via data.batch), then passes through an MLP.

    Args:
        backbone_dim: Output dimension from the backbone.
        dataset: PyG dataset, used to infer num_graph_features.
        num_targets: Number of prediction targets.
        node_dim: Hidden dimension for the MLP.
        use_layer_norm: Whether to apply LayerNorm inside the MLP.
        use_dropout: Whether to apply Dropout inside the MLP.
        dropout_p: Dropout probability (default 0.5).
    """

    def __init__(self, backbone_dim: int, dataset, num_targets: int,
                 node_dim: int = 64, use_layer_norm: bool = False,
                 use_dropout: bool = False, dropout_p: float = 0.5):
        super().__init__()
        lin1_in = backbone_dim + dataset.num_graph_features
        self.lin1 = Linear(lin1_in, node_dim)
        self.norm1 = LayerNorm(node_dim) if use_layer_norm else None
        self.drop1 = nn.Dropout(dropout_p) if use_dropout else None
        self.lin2 = Linear(node_dim, num_targets)

    def forward(self, node_features: torch.Tensor, data) -> torch.Tensor:
        """Forward pass.

        Args:
            node_features: Node features from backbone [num_nodes, backbone_dim].
            data: PyG Data with batch and graph_attr.

        Returns:
            torch.Tensor: Per-node predictions [num_nodes, num_targets].
        """
        if data.graph_attr.size(-1) > 0:
            graph_attr_per_node = data.graph_attr[data.batch]
            combined = torch.cat([node_features, graph_attr_per_node], dim=1)
        else:
            combined = node_features

        out = self.lin1(combined)
        if self.norm1 is not None:
            out = self.norm1(out)
        out = F.relu(out)
        if self.drop1 is not None:
            out = self.drop1(out)
        return self.lin2(out)
