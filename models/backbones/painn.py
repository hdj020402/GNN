"""PaiNN (Polarizable Atom Interaction Neural Network) backbone.

Reference: "Equivariant message passing for the modeling of scalar and
           tensorial properties in atomistic systems", Schütt et al. 2021.
           https://arxiv.org/abs/2102.03150

Adapted from the SchNetPack implementation to use PyG's Data format
(data.edge_index, data.pos, data.z) and our BackboneBase interface.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Linear

from torch_geometric.nn import radius_graph

from models.backbones.base import BackboneBase


# ---------------------------------------------------------------------------
# Utility: scatter_add without extra dependencies
# ---------------------------------------------------------------------------

def _scatter_add(src: torch.Tensor, idx: torch.Tensor,
                 dim_size: int) -> torch.Tensor:
    """Scatter-add src[i] onto out[idx[i]] for i in range(len(idx))."""
    shape = [dim_size] + list(src.shape[1:])
    out = torch.zeros(shape, dtype=src.dtype, device=src.device)
    idx_exp = idx.view(-1, *([1] * (src.dim() - 1))).expand_as(src)
    return out.scatter_add_(0, idx_exp, src)


# ---------------------------------------------------------------------------
# Radial basis and cutoff
# ---------------------------------------------------------------------------

class GaussianRBF(nn.Module):
    """Gaussian radial basis functions spaced linearly from start to cutoff."""
    def __init__(self, n_rbf: int, cutoff: float, start: float = 0.0,
                 trainable: bool = False):
        super().__init__()
        self.n_rbf = n_rbf
        offset = torch.linspace(start, cutoff, n_rbf)
        width = (offset[1] - offset[0]).abs() * torch.ones_like(offset)
        if trainable:
            self.offset = nn.Parameter(offset)
            self.width = nn.Parameter(width)
        else:
            self.register_buffer('offset', offset)
            self.register_buffer('width', width)

    def forward(self, dist: torch.Tensor) -> torch.Tensor:
        # dist: [E]  →  [E, n_rbf]
        return torch.exp(-0.5 * ((dist.unsqueeze(-1) - self.offset) / self.width) ** 2)


class CosineCutoff(nn.Module):
    """Cosine cutoff envelope: goes smoothly from 1 at r=0 to 0 at r=cutoff."""
    def __init__(self, cutoff: float):
        super().__init__()
        self.register_buffer('cutoff', torch.tensor(cutoff))

    def forward(self, dist: torch.Tensor) -> torch.Tensor:
        envelope = 0.5 * (torch.cos(dist * torch.pi / self.cutoff) + 1.0)
        return envelope * (dist < self.cutoff).float()


# ---------------------------------------------------------------------------
# PaiNN building blocks
# ---------------------------------------------------------------------------

class PaiNNInteraction(nn.Module):
    """Inter-atomic message passing block of PaiNN.

    Passes both scalar (q) and vector (mu) messages from neighbors.
    The filter weights W_ij are computed externally from the radial basis.

    Shapes:
        q:      [N, F]     scalar node features
        mu:     [N, 3, F]  vector node features (equivariant under rotations)
        W_ij:   [E, 3F]    pre-computed filter weights per edge
        dir_ij: [E, 3]     unit displacement vectors r_ij / |r_ij|
    """

    def __init__(self, n_atom_basis: int):
        super().__init__()
        # Context network: maps q → 3F weights used to mix neighbour info
        self.context_net = nn.Sequential(
            Linear(n_atom_basis, n_atom_basis),
            nn.SiLU(),
            Linear(n_atom_basis, 3 * n_atom_basis),
        )

    def forward(self, q: torch.Tensor, mu: torch.Tensor,
                W_ij: torch.Tensor, dir_ij: torch.Tensor,
                idx_i: torch.Tensor, idx_j: torch.Tensor,
                n_atoms: int):
        F = q.shape[-1]

        # Gather context from sending atoms (j) and apply filter
        ctx_j = self.context_net(q)[idx_j]      # [E, 3F]
        x = W_ij * ctx_j                         # [E, 3F]  – Hadamard filter

        dq, dmu_r, dmu_mu = torch.split(x, F, dim=-1)  # each [E, F]

        # Scalar aggregation
        dq = _scatter_add(dq, idx_i, n_atoms)           # [N, F]

        # Vector aggregation: direction-scaled + neighbour-vector-scaled
        dmu = (dmu_r.unsqueeze(1) * dir_ij.unsqueeze(-1)    # [E,1,F]*[E,3,1] → [E,3,F]
               + dmu_mu.unsqueeze(1) * mu[idx_j])            # [E,1,F]*[E,3,F] → [E,3,F]
        dmu = _scatter_add(dmu, idx_i, n_atoms)         # [N, 3, F]

        return q + dq, mu + dmu


class PaiNNMixing(nn.Module):
    """Intra-atomic mixing block: couples scalar and vector channels.

    Allows scalar features to influence vectors and vice versa through the
    vector norm (which is a rotation-invariant scalar).
    """

    def __init__(self, n_atom_basis: int, epsilon: float = 1e-8):
        super().__init__()
        self.intra_net = nn.Sequential(
            Linear(2 * n_atom_basis, n_atom_basis),
            nn.SiLU(),
            Linear(n_atom_basis, 3 * n_atom_basis),
        )
        # Mix vector channels: F → 2F (no bias to preserve equivariance)
        self.mu_channel_mix = Linear(n_atom_basis, 2 * n_atom_basis, bias=False)
        self.epsilon = epsilon

    def forward(self, q: torch.Tensor, mu: torch.Tensor):
        F = q.shape[-1]

        mu_mix = self.mu_channel_mix(mu)                   # [N, 3, 2F]
        mu_V, mu_W = torch.split(mu_mix, F, dim=-1)        # [N, 3, F] each

        # Rotation-invariant norm of mu_V: sum over spatial (dim=1)
        mu_Vn = torch.sqrt((mu_V ** 2).sum(dim=1, keepdim=True) + self.epsilon)
        # [N, 1, F]

        # Context: concatenate scalar q and vector norm
        ctx = torch.cat([q, mu_Vn.squeeze(1)], dim=-1)     # [N, 2F]
        x = self.intra_net(ctx)                             # [N, 3F]

        dq, dmu_intra, dqmu_intra = torch.split(x, F, dim=-1)

        # Scalar update from vector inner product (invariant)
        q = q + dq + dqmu_intra * (mu_V * mu_W).sum(dim=1)  # [N, F]
        # Vector update (equivariant)
        mu = mu + dmu_intra.unsqueeze(1) * mu_W              # [N, 3, F]

        return q, mu


# ---------------------------------------------------------------------------
# PaiNN backbone
# ---------------------------------------------------------------------------

class PaiNNBackbone(BackboneBase):
    """PaiNN backbone.

    O(3)-equivariant message passing that maintains both scalar (q) and
    vector (mu) features per atom. Only the final scalar features are
    returned (they are already rotation-invariant and suitable for
    graph/node-level prediction).

    **Interface differences from standard backbones:**
    - Uses ``data.z`` (atomic numbers as ``torch.long``) for initial embedding.
    - Uses ``data.pos`` (3-D Cartesian coordinates in Ångstrom).
    - Builds its own radius graph internally; ``data.edge_index`` / ``data.edge_attr``
      are **ignored**.
    - ``data.x`` is NOT used.

    Args:
        dataset: PyG dataset (unused; present for API consistency).
        n_atom_basis: Scalar (and vector) feature dimension F.
        n_interactions: Number of PaiNNInteraction + PaiNNMixing pairs.
        n_rbf: Number of Gaussian radial basis functions.
        cutoff: Cutoff radius in Ångstrom.
        max_num_neighbors: Maximum neighbours per atom.
        trainable_rbf: Whether Gaussian centres/widths are trainable.
    """

    def __init__(self, dataset, n_atom_basis: int = 128,
                 n_interactions: int = 3, n_rbf: int = 20,
                 cutoff: float = 5.0, max_num_neighbors: int = 32,
                 trainable_rbf: bool = False):
        super().__init__()
        self._n_atom_basis = n_atom_basis
        self.cutoff = cutoff
        self.max_num_neighbors = max_num_neighbors

        self.embedding = nn.Embedding(100, n_atom_basis, padding_idx=0)
        self.radial_basis = GaussianRBF(n_rbf, cutoff, trainable=trainable_rbf)
        self.cutoff_fn = CosineCutoff(cutoff)

        # Maps RBF → 3F per interaction (shared filter bank)
        self.filter_net = Linear(n_rbf,
                                 3 * n_atom_basis * n_interactions,
                                 bias=False)

        self.interactions = nn.ModuleList(
            [PaiNNInteraction(n_atom_basis) for _ in range(n_interactions)])
        self.mixings = nn.ModuleList(
            [PaiNNMixing(n_atom_basis) for _ in range(n_interactions)])

    def forward(self, data) -> torch.Tensor:
        """Forward pass.

        Args:
            data: PyG Data with ``z`` (long [N]), ``pos`` (float [N,3]),
                  and ``batch`` (long [N]).

        Returns:
            torch.Tensor: Scalar node embeddings ``[N, n_atom_basis]``.
        """
        z, pos, batch = data.z, data.pos, data.batch
        n_atoms = z.shape[0]

        # Build radius graph
        edge_index = radius_graph(pos, self.cutoff, batch=batch,
                                  max_num_neighbors=self.max_num_neighbors)
        idx_i, idx_j = edge_index[0], edge_index[1]

        # Displacement vectors and distances
        r_ij = pos[idx_j] - pos[idx_i]                   # [E, 3]
        d_ij = r_ij.norm(dim=1)                           # [E]
        dir_ij = r_ij / (d_ij.unsqueeze(-1) + 1e-10)     # [E, 3]

        # Radial basis + cutoff → filter weights for all interactions
        phi = self.radial_basis(d_ij)                     # [E, n_rbf]
        fcut = self.cutoff_fn(d_ij).unsqueeze(-1)         # [E, 1]
        all_filters = self.filter_net(phi) * fcut         # [E, 3F*n_interactions]

        # Initial embeddings
        q = self.embedding(z)                             # [N, F]
        mu = torch.zeros(n_atoms, 3, self._n_atom_basis,
                         device=q.device, dtype=q.dtype)  # [N, 3, F]

        F = self._n_atom_basis
        for i, (interaction, mixing) in enumerate(
                zip(self.interactions, self.mixings)):
            W_ij = all_filters[:, i * 3 * F:(i + 1) * 3 * F]  # [E, 3F]
            q, mu = interaction(q, mu, W_ij, dir_ij, idx_i, idx_j, n_atoms)
            q, mu = mixing(q, mu)

        return q  # [N, F]  — scalar features (rotation-invariant)

    @property
    def output_dim(self) -> int:
        return self._n_atom_basis
