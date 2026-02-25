"""Node-level prediction head."""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Linear, ReLU, LayerNorm


class NodePredictionHead(nn.Module):
    """Head for node-level prediction.

    Concatenates node features with graph attributes (broadcasted to node level),
    and applies MLPs to produce per-node predictions.

    Args:
        backbone_dim: Input dimension from backbone (output_dim)
        dataset: PyG dataset (used to get num_graph_features and num_targets)
        num_targets: Number of prediction targets
        dim_linear: Hidden dimension for MLPs
        use_dropout: Whether to use dropout
    """

    def __init__(self, backbone_dim: int, dataset, num_targets: int,
                 dim_linear: int = 64, use_dropout: bool = False):
        super().__init__()
        self.backbone_dim = backbone_dim
        self.num_targets = num_targets

        # Calculate input dimension for first linear layer
        # Node features + graph attributes (broadcasted)
        lin1_in_dim = backbone_dim + dataset.num_graph_features

        # MLP layers
        self.lin1 = Linear(lin1_in_dim, dim_linear)
        self.norm1 = LayerNorm(dim_linear)
        self.dropout1 = nn.Dropout(0.5) if use_dropout else None

        self.lin2 = Linear(dim_linear, num_targets)

    def forward(self, node_features: torch.Tensor, data) -> torch.Tensor:
        """Forward pass for node prediction head.

        Args:
            node_features: Node-level features from backbone [num_nodes, backbone_dim]
            data: PyG Data object with batch and graph_attr

        Returns:
            torch.Tensor: Node-level predictions [num_nodes, num_targets]
        """
        # Broadcast graph attributes to node level using batch indices
        if data.graph_attr.size(0) > 0:
            graph_attr_per_node = data.graph_attr[data.batch]
            combined = torch.cat([node_features, graph_attr_per_node], dim=1)
        else:
            combined = node_features

        # MLP
        out = F.relu(self.lin1(combined))
        out = self.norm1(out)
        if self.dropout1 is not None:
            out = self.dropout1(out)

        # Output layer
        out = self.lin2(out)

        return out
