"""Graph-level prediction head."""
from typing import Literal
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Linear, LayerNorm

from torch_geometric.nn import Set2Set, global_mean_pool, global_max_pool, global_add_pool


class GraphPredictionHead(nn.Module):
    """Head for graph-level prediction.

    Aggregates node features to a graph-level representation using a chosen
    readout strategy, optionally concatenates graph-level attributes, then
    passes through an MLP.

    Args:
        backbone_dim: Output dimension from the backbone.
        dataset: PyG dataset, used to infer num_graph_features.
        num_targets: Number of prediction targets.
        readout: Aggregation strategy. Options:
            - 'set2set': Learnable recurrent aggregation (expressive but slow).
            - 'mean': Global mean pooling (simple and often competitive).
            - 'max': Global max pooling.
            - 'sum': Global sum pooling.
        processing_steps: Number of LSTM steps for Set2Set (ignored for other readouts).
        node_dim: Hidden dimension for the MLP.
        use_dropout: Whether to apply Dropout inside the MLP.
        dropout_p: Dropout probability (default 0.5).
    """

    def __init__(self, backbone_dim: int, dataset, num_targets: int,
                 readout: Literal['set2set', 'mean', 'max', 'sum'] = 'set2set',
                 processing_steps: int = 3, node_dim: int = 64,
                 use_dropout: bool = False, dropout_p: float = 0.5):
        super().__init__()
        self.readout = readout

        if readout == 'set2set':
            self.aggregator = Set2Set(backbone_dim, processing_steps=processing_steps)
            agg_out_dim = 2 * backbone_dim  # Set2Set doubles the dimension
        else:
            self.aggregator = None
            agg_out_dim = backbone_dim

        # MLP: (aggregated node features + graph attributes) → predictions
        lin1_in = agg_out_dim + dataset.num_graph_features
        self.lin1 = Linear(lin1_in, node_dim)
        self.norm1 = LayerNorm(node_dim)
        self.drop1 = nn.Dropout(dropout_p) if use_dropout else None
        self.lin2 = Linear(node_dim, num_targets)

    def forward(self, node_features: torch.Tensor, data) -> torch.Tensor:
        """Forward pass.

        Args:
            node_features: Node features from backbone [num_nodes, backbone_dim].
            data: PyG Data with batch and graph_attr.

        Returns:
            torch.Tensor: Predictions [batch_size, num_targets].
        """
        # Aggregate node → graph
        if self.readout == 'set2set':
            graph_feat = self.aggregator(node_features, data.batch)
        elif self.readout == 'mean':
            graph_feat = global_mean_pool(node_features, data.batch)
        elif self.readout == 'max':
            graph_feat = global_max_pool(node_features, data.batch)
        elif self.readout == 'sum':
            graph_feat = global_add_pool(node_features, data.batch)
        else:
            raise ValueError(f"Unknown readout: {self.readout}")

        # Concatenate graph-level attributes if present
        if data.graph_attr.size(-1) > 0:
            combined = torch.cat([graph_feat, data.graph_attr], dim=1)
        else:
            combined = graph_feat

        out = F.relu(self.norm1(self.lin1(combined)))
        if self.drop1 is not None:
            out = self.drop1(out)
        return self.lin2(out)
