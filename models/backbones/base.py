"""Abstract base class for graph neural network backbones."""
from abc import ABC, abstractmethod
import torch
import torch.nn as nn


class BackboneBase(nn.Module, ABC):
    """Abstract base class for GNN backbone architectures.

    All backbones must implement:
    - forward(): Process graph data and return node-level features
    - output_dim property: The dimension of output features
    """

    def __init__(self):
        super().__init__()

    @abstractmethod
    def forward(self, data) -> torch.Tensor:
        """Forward pass for the backbone.

        Args:
            data: PyG Data object with at least x, edge_index, and batch attributes

        Returns:
            torch.Tensor: Node-level features with shape [num_nodes, output_dim]
        """
        pass

    @property
    @abstractmethod
    def output_dim(self) -> int:
        """Output dimension of the backbone.

        Returns:
            int: The dimension of node-level features
        """
        pass
