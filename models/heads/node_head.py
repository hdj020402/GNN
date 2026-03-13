"""Node-level prediction head."""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Linear, LayerNorm
from torch_geometric.nn import global_add_pool


class SumConservationLayer(nn.Module):
    """Differentiable sum conservation constraint for node-level predictions.

    Applies a uniform correction to per-node predictions so that the sum
    per graph equals a known target value. The constraint is exact and
    fully differentiable for end-to-end training.

    The target sum is obtained from one of two sources (checked in order):
      1. ``data.sum_target`` — a graph-level attribute read from
         ``graph_attr_file`` via ``sum_target_list``.  Values are in the
         **original** (pre-normalization) scale; the layer converts them
         internally using stored mean / std.
      2. ``data.y`` — per-node targets (fallback).  The target sum is
         ``sum(y)`` per graph, already in normalized space.

    If neither source is available, a ``ValueError`` is raised.

    Args:
        num_targets: Number of prediction targets per node.
    """

    def __init__(self, num_targets: int):
        super().__init__()
        # Normalization parameters, used only when converting
        # data.sum_target from original scale to normalized scale.
        # Registered as buffers so they are saved / loaded with the checkpoint.
        self.register_buffer('_mean', torch.zeros(num_targets))
        self.register_buffer('_std', torch.ones(num_targets))

    def set_norm_params(self, mean: torch.Tensor, std: torch.Tensor):
        """Store z-score normalization parameters.

        These are used to convert ``data.sum_target`` (original scale) to
        the normalized scale expected by the model.

        Args:
            mean: Per-target mean.
            std: Per-target standard deviation.
        """
        self._mean.copy_(mean.view(-1))
        self._std.copy_(std.view(-1))

    def forward(
        self,
        pred: torch.Tensor,
        batch: torch.Tensor,
        sum_target: torch.Tensor | None = None,
        y: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Apply sum conservation constraint.

        Args:
            pred: Per-node predictions ``[num_nodes, num_targets]``
                (normalized).
            batch: Graph assignment indices ``[num_nodes]``.
            sum_target: Pre-computed target sum per graph
                ``[num_graphs, num_targets]`` in the **original**
                (un-normalized) scale.  Preferred source.
            y: Normalized per-node targets ``[num_nodes, num_targets]``.
                Fallback: target sum is computed as ``sum(y)`` per graph.

        Returns:
            Corrected predictions ``[num_nodes, num_targets]``.

        Raises:
            ValueError: If neither ``sum_target`` nor ``y`` is provided.
        """
        # Number of nodes per graph
        ones = torch.ones(batch.size(0), 1, device=pred.device)
        num_nodes = global_add_pool(ones, batch)  # [num_graphs, 1]

        # Compute target sum in normalized space
        if sum_target is not None:
            # Primary: convert from original scale to normalized scale
            # target_sum_norm = (Q - N * mean) / std
            target_sum = (sum_target - num_nodes * self._mean) / self._std
        elif y is not None:
            # Fallback: sum of normalized targets
            target_sum = global_add_pool(y, batch)  # [num_graphs, num_targets]
        else:
            raise ValueError(
                "sum_conservation is enabled but neither sum_target "
                "(via data.sum_target_list) nor per-node targets (data.y) "
                "are available. Provide at least one."
            )

        # Current predicted sum
        pred_sum = global_add_pool(pred, batch)  # [num_graphs, num_targets]

        # Uniform correction per node
        correction = (target_sum - pred_sum) / num_nodes  # [num_graphs, num_targets]

        return pred + correction[batch]


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
        sum_conservation: Whether to enforce a sum conservation constraint
            on the per-node predictions.
    """

    def __init__(self, backbone_dim: int, dataset, num_targets: int,
                 node_dim: int = 64, use_layer_norm: bool = False,
                 use_dropout: bool = False, dropout_p: float = 0.5,
                 sum_conservation: bool = False):
        super().__init__()
        lin1_in = backbone_dim + dataset.num_graph_features
        self.lin1 = Linear(lin1_in, node_dim)
        self.norm1 = LayerNorm(node_dim) if use_layer_norm else None
        self.drop1 = nn.Dropout(dropout_p) if use_dropout else None
        self.lin2 = Linear(node_dim, num_targets)
        self.sum_conservation = (
            SumConservationLayer(num_targets) if sum_conservation else None
        )

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
        out = self.lin2(out)

        if self.sum_conservation is not None:
            sum_target = getattr(data, 'sum_target', None)
            y = data.y if hasattr(data, 'y') and data.y is not None else None
            out = self.sum_conservation(
                out, data.batch, sum_target=sum_target, y=y
            )

        return out
