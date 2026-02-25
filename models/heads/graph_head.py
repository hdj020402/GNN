"""Graph-level prediction head."""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Linear, ReLU, LayerNorm

from torch_geometric.nn import Set2Set


class GraphPredictionHead(nn.Module):
    """Head for graph-level prediction.

    Uses Set2Set for graph-level aggregation, concatenates with graph attributes,
    and applies MLPs to produce predictions.

    Args:
        backbone_dim: Input dimension from backbone (output_dim)
        dataset: PyG dataset (used to get num_graph_features and num_targets)
        num_targets: Number of prediction targets
        processing_steps: Number of steps for Set2Set
        dim_linear: Hidden dimension for MLPs
        use_dropout: Whether to use dropout
    """

    def __init__(self, backbone_dim: int, dataset, num_targets: int,
                 processing_steps: int = 1, dim_linear: int = 64,
                 use_dropout: bool = False):
        super().__init__()
        self.backbone_dim = backbone_dim
        self.num_targets = num_targets
        self.processing_steps = processing_steps

        # Set2Set for graph-level aggregation
        # Set2Set output is 2 * input_dim
        self.set2set = Set2Set(backbone_dim, processing_steps=processing_steps)
        set2set_dim = 2 * backbone_dim

        # Calculate input dimension for first linear layer
        # Set2Set output + graph attributes
        lin1_in_dim = set2set_dim + dataset.num_graph_features

        # MLP layers
        self.lin1 = Linear(lin1_in_dim, dim_linear)
        self.norm1 = LayerNorm(dim_linear)
        self.dropout1 = nn.Dropout(0.5) if use_dropout else None

        self.lin2 = Linear(dim_linear, num_targets)

    def forward(self, node_features: torch.Tensor, data) -> torch.Tensor:
        """Forward pass for graph prediction head.

        Args:
            node_features: Node-level features from backbone [num_nodes, backbone_dim]
            data: PyG Data object with batch and graph_attr

        Returns:
            torch.Tensor: Graph-level predictions [batch_size, num_targets]
        """
        # Graph-level aggregation
        graph_feat = self.set2set(node_features, data.batch)

        # Concatenate with graph attributes
        if data.graph_attr.size(0) > 0:
            combined = torch.cat([graph_feat, data.graph_attr], dim=1)
        else:
            combined = graph_feat

        # MLP
        out = F.relu(self.lin1(combined))
        out = self.norm1(out)
        if self.dropout1 is not None:
            out = self.dropout1(out)

        # Output layer
        out = self.lin2(out)

        return out
