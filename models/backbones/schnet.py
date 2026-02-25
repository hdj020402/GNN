"""SchNet backbone — E(3)-invariant continuous-filter CNN."""
import torch
import torch.nn as nn

from torch_geometric.nn.models.schnet import (
    GaussianSmearing, InteractionBlock, RadiusInteractionGraph
)

from models.backbones.base import BackboneBase


class SchNetBackbone(BackboneBase):
    """SchNet backbone (Schütt et al. 2017, "SchNet: A Continuous-filter
    Convolutional Neural Network for Modeling Quantum Interactions").

    SchNet is E(3)-invariant — the output scalar energies (or node embeddings)
    are unchanged under rotation, translation, and reflection. It achieves this
    by using interatomic *distances* (scalars) as inputs, discarding directional
    information entirely.

    Key differences from MPNN/GCN/GIN:
      - Input: data.z (atomic numbers, dtype=long) + data.pos (3D coordinates).
        Does NOT use data.x or data.edge_attr.
      - Edges are built internally from a radial cutoff (not from the SDF bonds).
      - Edge "features" are radial basis functions of interatomic distances.

    This backbone borrows SchNet's internal components (embedding, interaction
    blocks) from PyG and runs them, returning node-level embeddings before the
    readout layer. The graph-level prediction is delegated to the Head.

    Args:
        dataset: PyG dataset (unused for dim inference; SchNet uses z/pos).
        hidden_channels: Node embedding dimension throughout the network.
        num_filters: Width of the continuous filter networks.
        num_interactions: Number of interaction (message-passing) blocks.
        num_gaussians: Number of Gaussian basis functions for distance encoding.
        cutoff: Radial cutoff in Angstroms for building the interaction graph.
    """

    def __init__(self, dataset=None, hidden_channels: int = 128,
                 num_filters: int = 128, num_interactions: int = 6,
                 num_gaussians: int = 50, cutoff: float = 10.0):
        super().__init__()
        self._hidden_channels = hidden_channels

        self.embedding = nn.Embedding(100, hidden_channels)
        self.interaction_graph = RadiusInteractionGraph(cutoff, max_num_neighbors=32)
        self.distance_expansion = GaussianSmearing(0.0, cutoff, num_gaussians)

        self.interactions = nn.ModuleList([
            InteractionBlock(hidden_channels, num_gaussians, num_filters, cutoff)
            for _ in range(num_interactions)
        ])

    def forward(self, data) -> torch.Tensor:
        """Forward pass.

        Args:
            data: PyG Data with:
                - data.z: atomic numbers [num_atoms], dtype=long
                - data.pos: 3D positions [num_atoms, 3], dtype=float
                - data.batch: batch index [num_atoms]

        Returns:
            torch.Tensor: Node embeddings [num_atoms, hidden_channels].
        """
        z, pos, batch = data.z, data.pos, data.batch

        h = self.embedding(z)
        edge_index, edge_weight = self.interaction_graph(pos, batch)
        edge_attr = self.distance_expansion(edge_weight)

        for interaction in self.interactions:
            h = h + interaction(h, edge_index, edge_weight, edge_attr)

        return h  # [num_atoms, hidden_channels] — before SchNet's lin1/lin2 readout

    @property
    def output_dim(self) -> int:
        return self._hidden_channels
