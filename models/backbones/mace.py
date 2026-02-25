"""MACE backbone — E(3)-equivariant with higher-order message passing.

MACE (Batatia et al. 2022, "MACE: Higher Order Equivariant Message Passing
Neural Networks for Fast and Accurate Force Fields") is an equivariant GNN
that uses higher-body-order interactions through equivariant tensor products.

Unlike SchNet/DimeNet++ (which are invariant), MACE is **E(3)-equivariant**:
vector/tensor outputs transform properly under rotation. This makes it
suitable for predicting vectorial quantities (forces, dipole moments, etc.).

Requirements:
    pip install mace-torch

MACE is NOT included in standard PyTorch Geometric. This wrapper is provided
for users who have installed `mace-torch` separately.
"""

try:
    from mace.modules import MACE
    _MACE_AVAILABLE = True
except ImportError:
    _MACE_AVAILABLE = False

import torch
import torch.nn as nn

from models.backbones.base import BackboneBase


class MACEBackbone(BackboneBase):
    """MACE backbone for equivariant molecular property prediction.

    Wraps the MACE model from `mace-torch` to expose node-level features
    before the final readout.

    Requires:  pip install mace-torch

    Args:
        dataset: Unused (uses data.z and data.pos).
        r_max: Interaction cutoff radius in Angstroms.
        num_bessel: Number of radial Bessel basis functions.
        num_polynomial_cutoff: Polynomial cutoff envelope degree.
        max_ell: Maximum angular momentum ℓ for spherical harmonics.
        interaction_cls: MACE interaction class name.
        num_interactions: Number of interaction layers.
        hidden_irreps: Hidden irreducible representations string
            (e.g. '128x0e+128x1o' for scalars + vectors).
        MLP_irreps: MLP output irreps.
        atomic_energies: Optional per-atom reference energies.
        avg_num_neighbors: Average number of neighbors (used for normalization).
        atomic_numbers: List of atomic numbers present in the dataset.
        correlation: Many-body correlation order.
    """

    def __init__(self, dataset=None, r_max: float = 5.0, num_bessel: int = 8,
                 num_polynomial_cutoff: int = 5, max_ell: int = 3,
                 interaction_cls_first: str = 'RealAgnosticInterResidualBlock',
                 interaction_cls: str = 'RealAgnosticResidualInteractionBlock',
                 num_interactions: int = 2,
                 hidden_irreps: str = '128x0e',
                 MLP_irreps: str = '16x0e',
                 atomic_energies=None,
                 avg_num_neighbors: float = 8.0,
                 atomic_numbers: list | None = None,
                 correlation: int = 3):
        super().__init__()

        if not _MACE_AVAILABLE:
            raise ImportError(
                "MACE is not installed. Install it with: pip install mace-torch\n"
                "See https://github.com/ACEsuit/mace for details."
            )

        import e3nn.o3 as o3
        import numpy as np

        if atomic_numbers is None:
            # Default: elements H through Xe (Z=1..54)
            atomic_numbers = list(range(1, 55))

        if atomic_energies is None:
            atomic_energies = torch.zeros(len(atomic_numbers))

        self.model = MACE(
            r_max=r_max,
            num_bessel=num_bessel,
            num_polynomial_cutoff=num_polynomial_cutoff,
            max_ell=max_ell,
            interaction_cls=interaction_cls,
            interaction_cls_first=interaction_cls_first,
            num_interactions=num_interactions,
            num_elements=len(atomic_numbers),
            hidden_irreps=o3.Irreps(hidden_irreps),
            MLP_irreps=o3.Irreps(MLP_irreps),
            atomic_energies=atomic_energies,
            avg_num_neighbors=avg_num_neighbors,
            atomic_numbers=atomic_numbers,
            correlation=correlation,
        )

        # Infer output dim from hidden_irreps (count scalars only, i.e. 0e/0o terms)
        irreps = o3.Irreps(hidden_irreps)
        self._output_dim = sum(mul * (2 * ir.l + 1) for mul, ir in irreps)

    def forward(self, data) -> torch.Tensor:
        """Forward pass.

        Args:
            data: PyG Data with:
                - data.z: atomic numbers [num_atoms], dtype=long
                - data.pos: 3D positions [num_atoms, 3], dtype=float
                - data.batch: batch index [num_atoms]

        Returns:
            torch.Tensor: Per-node equivariant features.
        """
        # MACE expects a specific input dict
        mace_input = {
            'positions': data.pos,
            'node_attrs': nn.functional.one_hot(
                data.z, num_classes=self.model.num_elements
            ).float(),
            'batch': data.batch,
            'edge_index': data.edge_index,
            'shifts': torch.zeros(data.edge_index.size(1), 3,
                                  device=data.pos.device),
            'unit_shifts': torch.zeros_like(
                torch.zeros(data.edge_index.size(1), 3, device=data.pos.device)
            ),
        }
        out = self.model(mace_input, compute_force=False)
        return out['node_feats']

    @property
    def output_dim(self) -> int:
        return self._output_dim
