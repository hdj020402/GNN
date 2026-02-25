"""MACE (Multi-Atomic Cluster Expansion) backbone.

Reference: "MACE: Higher Order Equivariant Message Passing Neural Networks
           for Fast and Accurate Force Fields", Batatia et al. 2022.
           https://arxiv.org/abs/2206.07697

Requires e3nn and mace-torch:
    pip install e3nn mace-torch

Architecture overview
---------------------
MACE extends standard message passing by combining:
  1. O(3)-equivariant features using irreducible representations (irreps) via e3nn.
  2. Many-body interactions via the ACE product basis (EquivariantProductBasisBlock),
     which raises neighbour-aggregated features to body-order `correlation`.
     - correlation=2 → 2-body (standard MP / ACE)
     - correlation=3 → 3-body (MACE)

This makes MACE strictly more expressive than PaiNN (2-body, l=1 only)
and SchNet (2-body, invariant).

Node features
-------------
MACE distinguishes:
  - ``node_attrs``  (fixed, one-hot element embedding) — NOT updated during MP
  - ``node_feats``  (equivariant irreps, updated each layer)

The final ``node_feats`` is a mixed irreps tensor of shape [N, irreps_dim].
Only the *scalar* (l=0) component can be fed directly into standard linear
heads. For full equivariant readout, a separate equivariant head is needed.
"""
import torch
import torch.nn as nn
from torch_geometric.nn import radius_graph

from e3nn import o3
from mace.modules import (
    RealAgnosticInteractionBlock,
    RealAgnosticResidualInteractionBlock,
    EquivariantProductBasisBlock,
)
from mace.modules.blocks import LinearNodeEmbeddingBlock, RadialEmbeddingBlock
from mace.modules.utils import get_edge_vectors_and_lengths

from models.backbones.base import BackboneBase


class MACEBackbone(BackboneBase):
    """MACE backbone exposing node-level scalar embeddings.

    After all MACE interaction + product layers, only the l=0 (scalar)
    component of the equivariant node features is returned. This gives a
    rotation-invariant vector that is compatible with the existing heads.

    **Interface differences from standard backbones:**
    - Uses ``data.z``  (atomic numbers, ``torch.long``).
    - Uses ``data.pos`` (3-D coordinates in Ångstrom).
    - ``data.edge_index`` / ``data.x`` / ``data.edge_attr`` are **ignored**.
    - ``data.shifts`` (periodic boundary shifts, ``[E, 3]``) may be provided
      for periodic systems; set to ``torch.zeros(E, 3)`` for molecules.

    Args:
        dataset: PyG dataset, used only for ``atomic_numbers`` list.
        r_max: Cutoff radius in Ångstrom.
        num_bessel: Number of Bessel radial basis functions.
        num_polynomial_cutoff: Polynomial cutoff order.
        max_ell: Maximum spherical harmonic degree L (e.g. 2 or 3).
        hidden_irreps_str: Equivariant hidden feature spec, e.g.
            ``"128x0e + 128x1o"`` or ``"64x0e + 64x1o + 64x2e"``.
        num_interactions: Number of MACE layers.
        correlation: Body order (2 = ACE, 3 = MACE).
        avg_num_neighbors: Normalization factor; set to the average number
            of atoms within ``r_max`` in your dataset.
        max_num_neighbors: Maximum neighbours per atom.
    """

    def __init__(
        self,
        dataset=None,
        r_max: float = 5.0,
        num_bessel: int = 8,
        num_polynomial_cutoff: int = 5,
        max_ell: int = 2,
        hidden_irreps_str: str = "128x0e + 128x1o",
        num_interactions: int = 2,
        correlation: int = 3,
        avg_num_neighbors: float = 10.0,
        max_num_neighbors: int = 32,
    ):
        super().__init__()

        self.r_max = r_max
        self.max_num_neighbors = max_num_neighbors

        hidden_irreps = o3.Irreps(hidden_irreps_str)
        # Scalar (l=0) output dimension: sum of multiplicities of l=0 terms
        self._scalar_dim = sum(
            mul for mul, ir in hidden_irreps if ir.l == 0
        )

        # --- Adapted from mace.modules.models.MACE ---
        # Node attribute irreps: one-hot over element types
        # We use a fixed set of 100 possible atomic numbers
        num_elements = 100
        node_attr_irreps = o3.Irreps(f"{num_elements}x0e")
        node_feats_irreps = o3.Irreps(
            [(mul, ir) for mul, ir in hidden_irreps if ir.l == 0]
        )

        self.node_embedding = LinearNodeEmbeddingBlock(
            irreps_in=node_attr_irreps,
            irreps_out=node_feats_irreps,
        )

        self.radial_embedding = RadialEmbeddingBlock(
            r_max=r_max,
            num_bessel=num_bessel,
            num_polynomial_cutoff=num_polynomial_cutoff,
        )
        sh_irreps = o3.Irreps.spherical_harmonics(max_ell)
        self.spherical_harmonics = o3.SphericalHarmonics(
            sh_irreps, normalize=True, normalization="component"
        )

        self.interactions = nn.ModuleList()
        self.products = nn.ModuleList()

        for i in range(num_interactions):
            inter_cls = (RealAgnosticInteractionBlock
                         if i == 0
                         else RealAgnosticResidualInteractionBlock)
            interaction = inter_cls(
                node_attrs_irreps=node_attr_irreps,
                node_feats_irreps=node_feats_irreps if i == 0 else hidden_irreps,
                edge_attrs_irreps=sh_irreps,
                edge_feats_irreps=o3.Irreps(f"{self.radial_embedding.out_dim}x0e"),
                target_irreps=hidden_irreps,
                hidden_irreps=hidden_irreps,
                avg_num_neighbors=avg_num_neighbors,
                radial_MLP=[64, 64, 64],
            )
            product = EquivariantProductBasisBlock(
                node_feats_irreps=interaction.target_irreps,
                target_irreps=hidden_irreps,
                correlation=correlation,
                num_elements=num_elements,
                use_sc=True,
            )
            self.interactions.append(interaction)
            self.products.append(product)

        # Track irreps for index extraction in forward
        self._hidden_irreps = hidden_irreps

    # ------------------------------------------------------------------
    # Helper: one-hot encode atomic numbers as node_attrs
    # ------------------------------------------------------------------
    def _one_hot_z(self, z: torch.Tensor, num_classes: int = 100) -> torch.Tensor:
        return torch.zeros(z.shape[0], num_classes,
                           device=z.device, dtype=torch.float).scatter_(
            1, z.clamp(0, num_classes - 1).unsqueeze(1), 1.0
        )

    def forward(self, data) -> torch.Tensor:
        """Forward pass.

        Args:
            data: PyG Data with ``z`` (long [N]), ``pos`` (float [N, 3]),
                  ``batch`` (long [N]).
                  Optionally ``shifts`` (float [E, 3]) for periodic systems.

        Returns:
            torch.Tensor: Scalar node embeddings ``[N, scalar_dim]``.
        """
        z, pos, batch = data.z, data.pos, data.batch

        # Build radius graph
        edge_index = radius_graph(pos, self.r_max, batch=batch,
                                  max_num_neighbors=self.max_num_neighbors)

        # Displacement vectors (accounting for PBC shifts if present)
        shifts = getattr(data, 'shifts', None)
        if shifts is None:
            shifts = torch.zeros(edge_index.shape[1], 3,
                                 device=pos.device, dtype=pos.dtype)
        vectors, lengths = get_edge_vectors_and_lengths(
            positions=pos, edge_index=edge_index, shifts=shifts,
            normalize=False,
        )

        # Edge features
        edge_attrs = self.spherical_harmonics(vectors)  # [E, (L+1)^2]
        edge_feats = self.radial_embedding(lengths)     # [E, num_bessel]

        # Node attributes (fixed one-hot) and initial features
        node_attrs = self._one_hot_z(z)                 # [N, 100]
        node_feats = self.node_embedding(node_attrs)    # [N, scalar_feats_irreps]

        # MACE interaction + product layers
        for interaction, product in zip(self.interactions, self.products):
            node_feats, sc = interaction(
                node_attrs=node_attrs,
                node_feats=node_feats,
                edge_attrs=edge_attrs,
                edge_feats=edge_feats,
                edge_index=edge_index,
            )
            node_feats = product(
                node_feats=node_feats,
                sc=sc,
                node_attrs=node_attrs,
            )

        # Extract only the scalar (l=0) components as invariant embedding
        scalars = []
        idx = 0
        for mul, ir in self._hidden_irreps:
            dim = mul * ir.dim
            if ir.l == 0:
                scalars.append(node_feats[:, idx:idx + dim])
            idx += dim
        return torch.cat(scalars, dim=-1)  # [N, scalar_dim]

    @property
    def output_dim(self) -> int:
        return self._scalar_dim
