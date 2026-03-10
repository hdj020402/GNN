"""Unified model combining backbone and head."""
import torch
import torch.nn as nn

from models.backbones.base import BackboneBase


class UnifiedModel(nn.Module):
    """Unified model combining a backbone and head.

    The backbone processes the graph and produces node-level features.
    The head takes these features and produces task-specific predictions
    (either graph-level or node-level depending on the head).

    Args:
        backbone: BackboneBase instance
        head: Head module (GraphPredictionHead or NodePredictionHead)
    """

    def __init__(self, backbone: BackboneBase, head: nn.Module):
        super().__init__()
        self.backbone = backbone
        self.head = head

    def forward(self, data) -> torch.Tensor:
        """Forward pass for the unified model.

        Args:
            data: PyG Data object

        Returns:
            torch.Tensor: Predictions from the head
        """
        # Backbone produces node-level features
        node_features = self.backbone(data)

        # Head produces task-specific predictions
        output = self.head(node_features, data)

        return output
