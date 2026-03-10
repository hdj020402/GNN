"""DimeNet++ backbone — E(3)-invariant with angular information."""
import torch
import torch.nn as nn
from torch import Tensor

from torch_geometric.nn import radius_graph
from torch_geometric.nn.models import DimeNetPlusPlus
from torch_geometric.nn.models.dimenet import triplets

from models.backbones.base import BackboneBase


class _DimeNetPlusPlusNodeEncoder(DimeNetPlusPlus):
    """DimeNet++ subclass that returns per-node accumulated features instead
    of the graph-level aggregated output."""

    def forward(self, z: Tensor, pos: Tensor, batch=None) -> Tensor:
        """Run DimeNet++ but return accumulated per-node embeddings P
        instead of scatter(P, batch)."""
        edge_index = radius_graph(
            pos, r=self.cutoff, batch=batch,
            max_num_neighbors=self.max_num_neighbors
        )
        i, j, idx_i, idx_j, idx_k, idx_kj, idx_ji = triplets(
            edge_index, num_nodes=z.size(0)
        )

        dist = (pos[i] - pos[j]).pow(2).sum(dim=-1).sqrt()

        pos_jk = pos[idx_j] - pos[idx_k]
        pos_ij = pos[idx_i] - pos[idx_j]
        a = (pos_ij * pos_jk).sum(dim=-1)
        b = torch.cross(pos_ij, pos_jk, dim=1).norm(dim=-1)
        angle = torch.atan2(b, a)

        rbf = self.rbf(dist)
        sbf = self.sbf(dist, angle, idx_kj)

        x = self.emb(z, rbf, i, j)
        P = self.output_blocks[0](x, rbf, i, num_nodes=pos.size(0))

        for interaction_block, output_block in zip(
                self.interaction_blocks, self.output_blocks[1:]):
            x = interaction_block(x, rbf, sbf, idx_kj, idx_ji)
            P = P + output_block(x, rbf, i, num_nodes=pos.size(0))

        # Return per-node embeddings instead of scatter(P, batch)
        return P  # [num_atoms, out_channels]


class DimeNetPlusPlusBackbone(BackboneBase):
    """DimeNet++ backbone (Klicpera et al. 2020, "Fast and Uncertainty-Aware
    Directional Message Passing for Non-Equilibrium Molecules").

    DimeNet++ extends SchNet by incorporating *angular* information: each
    message depends on the pair of bonds (i–j–k), giving directional awareness
    while remaining E(3)-invariant (angles are rotation-invariant).

    Like SchNet, it uses atomic numbers + 3D positions and does NOT use
    data.x or data.edge_attr. Edges are built from a radial cutoff.

    We set DimeNet++'s `out_channels` to `hidden_channels` so that the
    accumulated per-node output P has shape [num_atoms, hidden_channels],
    which can then be fed directly to any Head.

    Args:
        dataset: Unused (equivariant models do not use data.x).
        hidden_channels: Node and output embedding dimension.
        num_blocks: Number of interaction + output block pairs.
        int_emb_size: Embedding dimension in the interaction blocks.
        basis_emb_size: Embedding dimension for basis representations.
        out_emb_channels: Channels in the output embedding blocks.
        num_spherical: Number of spherical harmonics.
        num_radial: Number of radial basis functions.
        cutoff: Radial cutoff in Angstroms.
    """

    def __init__(self, dataset=None, hidden_channels: int = 128,
                 num_blocks: int = 4, int_emb_size: int = 64,
                 basis_emb_size: int = 8, out_emb_channels: int = 256,
                 num_spherical: int = 7, num_radial: int = 6,
                 cutoff: float = 5.0):
        super().__init__()
        self._hidden_channels = hidden_channels

        self.model = _DimeNetPlusPlusNodeEncoder(
            hidden_channels=hidden_channels,
            out_channels=hidden_channels,     # P has shape [N, hidden_channels]
            num_blocks=num_blocks,
            int_emb_size=int_emb_size,
            basis_emb_size=basis_emb_size,
            out_emb_channels=out_emb_channels,
            num_spherical=num_spherical,
            num_radial=num_radial,
            cutoff=cutoff,
        )

    def forward(self, data) -> torch.Tensor:
        """Forward pass.

        Args:
            data: PyG Data with:
                - data.z: atomic numbers [num_atoms], dtype=long
                - data.pos: 3D positions [num_atoms, 3], dtype=float
                - data.batch: batch index [num_atoms]

        Returns:
            torch.Tensor: Per-node embeddings [num_atoms, hidden_channels].
        """
        return self.model(data.z, data.pos, data.batch)

    @property
    def output_dim(self) -> int:
        return self._hidden_channels
