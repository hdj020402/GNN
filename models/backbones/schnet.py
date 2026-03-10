"""SchNet backbone for molecular property prediction.

Reference: "SchNet: A continuous-filter convolutional neural network for
           modeling quantum interactions", Schütt et al. 2017.
           https://arxiv.org/abs/1706.08566

Wraps ``torch_geometric.nn.models.SchNet`` and overrides the forward pass
to return per-node embeddings instead of the graph-level energy scalar.
This mirrors the DimeNet++ backbone approach.

**Interface differences from standard backbones:**
- Uses ``data.z`` (atomic numbers as ``torch.long``) for initial embedding.
- Uses ``data.pos`` (3-D Cartesian coordinates in Ångstrom) for distances.
- Builds its own radius graph internally; ``data.edge_index`` / ``data.edge_attr``
  from the data pipeline are **ignored**.
- ``data.x`` is NOT used.
"""
import torch
from torch import Tensor

from torch_geometric.nn import radius_graph
from torch_geometric.nn.models import SchNet as _SchNet

from models.backbones.base import BackboneBase


class _SchNetNodeEncoder(_SchNet):
    """Subclass of PyG's SchNet that returns per-node embeddings.

    PyG's SchNet.forward() scatters node features to graph-level energy.
    We override to return the node embeddings after all interaction blocks,
    before the final linear layers and aggregation.
    """

    def forward(self, z: Tensor, pos: Tensor, batch=None) -> Tensor:
        batch = torch.zeros_like(z) if batch is None else batch

        h = self.embedding(z)

        edge_index = radius_graph(
            pos, r=self.cutoff, batch=batch,
            max_num_neighbors=self.max_num_neighbors,
            flow='source_to_target',
        )
        row, col = edge_index
        edge_weight = (pos[row] - pos[col]).norm(dim=-1)
        edge_attr = self.distance_expansion(edge_weight)

        for interaction in self.interactions:
            h = h + interaction(h, edge_index, edge_weight, edge_attr)

        return h  # [N, hidden_channels]


class SchNetBackbone(BackboneBase):
    """SchNet backbone.

    E(3)-invariant network using continuous-filter convolutions over interatomic
    distances. Does not use directional (vector) information, so it is invariant
    under rotations and reflections but cannot distinguish chiral structures.

    Args:
        dataset: PyG dataset (unused; present for API consistency).
        hidden_channels: Node feature dimension.
        num_filters: Filter width inside CFConv.
        num_interactions: Number of interaction blocks (depth).
        num_gaussians: Number of Gaussian basis functions.
        cutoff: Interaction cutoff radius in Ångstrom.
        max_num_neighbors: Maximum neighbours per atom (caps memory use).
    """

    def __init__(self, dataset=None, hidden_channels: int = 128,
                 num_filters: int = 128, num_interactions: int = 6,
                 num_gaussians: int = 50, cutoff: float = 10.0,
                 max_num_neighbors: int = 32):
        super().__init__()
        self._hidden_channels = hidden_channels

        self.model = _SchNetNodeEncoder(
            hidden_channels=hidden_channels,
            num_filters=num_filters,
            num_interactions=num_interactions,
            num_gaussians=num_gaussians,
            cutoff=cutoff,
            max_num_neighbors=max_num_neighbors,
        )

    def forward(self, data) -> torch.Tensor:
        """Forward pass.

        Args:
            data: PyG Data with ``z`` (long [N]) and ``pos`` (float [N, 3]).

        Returns:
            torch.Tensor: Node embeddings ``[N, hidden_channels]``.
        """
        return self.model(data.z, data.pos, data.batch)

    @property
    def output_dim(self) -> int:
        return self._hidden_channels
