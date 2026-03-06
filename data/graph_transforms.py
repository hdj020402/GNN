'''PyG Data transforms for graph construction and edge featurization.'''
import torch
from torch_geometric.data import Data


def complete_with_dist_filter(data: Data, distance_threshold: float | None = None) -> Data:
    '''Replace the bond-based edge_index with a radius / complete graph.

    For each pair of atoms (i, j) with i ≠ j, an edge is kept when
    distance(i, j) ≤ distance_threshold (or always, if threshold is None).
    Original edge_attr values are preserved for bonds; non-bond pairs get
    zero-padded edge_attr.

    Args:
        data: PyG Data with pos and edge_index populated.
        distance_threshold: maximum inter-atomic distance to include.
            None → complete graph (all pairs).

    Returns:
        The same Data object with edge_index and edge_attr updated in-place.
    '''
    if data.pos is None:
        raise ValueError("Position must be provided.")

    device = data.edge_index.device
    n = data.num_nodes

    row = torch.arange(n, dtype=torch.long, device=device).view(-1, 1).repeat(1, n).view(-1)
    col = torch.arange(n, dtype=torch.long, device=device).repeat(n)

    dist = torch.norm(data.pos[col] - data.pos[row], p=2, dim=-1)

    if distance_threshold is not None:
        mask = (dist <= distance_threshold) & (row != col)
    else:
        mask = row != col
    row, col = row[mask], col[mask]

    edge_index = torch.stack([row, col], dim=0)

    edge_attr = None
    if data.edge_attr is not None:
        idx = data.edge_index[0] * n + data.edge_index[1]
        size = list(data.edge_attr.size())
        size[0] = n * n
        edge_attr = data.edge_attr.new_zeros(size)
        edge_attr[idx] = data.edge_attr
        edge_attr = edge_attr[mask]

    data.edge_index = edge_index
    data.edge_attr = edge_attr
    return data


def power_dist(data: Data, cat: bool = True, power: float = 1.0) -> Data:
    '''Append r^power distance features to edge_attr.

    Computes the Euclidean distance between endpoint atoms for every edge,
    raises it to `power`, and concatenates the result to the existing
    edge_attr (when cat=True) or replaces it entirely.

    Args:
        data: PyG Data with pos and edge_index populated.
        cat: if True, concatenate to existing edge_attr; otherwise replace.
        power: exponent applied to the distance (e.g. -1 for 1/r).

    Returns:
        The same Data object with edge_attr updated in-place.
    '''
    if data.pos is None or data.edge_index is None:
        raise ValueError("pos and edge_index must both be present.")

    row, col = data.edge_index
    dist = torch.norm(data.pos[col] - data.pos[row], p=2, dim=-1).view(-1, 1)
    powered = dist ** power

    if data.edge_attr is not None and cat:
        pseudo = data.edge_attr
        if pseudo.dim() == 1:
            pseudo = pseudo.view(-1, 1)
        data.edge_attr = torch.cat([pseudo, powered.type_as(pseudo)], dim=-1)
    else:
        data.edge_attr = powered

    return data
