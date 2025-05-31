import os, json, pickle, torch
from typing import Dict, Optional
from torch_geometric.data import Data

'''utils for gnn attribute.
'''

def one_hot(hot_idx: int, total_len: int):
    '''generate one hot repr according to selected index and
    total length.
    
    Args:
        hot_idx: the index chosen to be 1.
        total_len: how long should the repr be.
        
    Return:
        one_hot_list: [0, 0, 1, 0] for hot_idx=2, total_len=4.
    '''
    one_hot_list = []
    for i in range(total_len):
        if i == hot_idx:
            one_hot_list.append(1)
        else:
            one_hot_list.append(0)
    
    return one_hot_list

def read_attr(file: Optional[str], attr_type: str, length: int) -> Optional[Dict]:
    if file is None:
        attr_dict = None
    else:
        _, ext = os.path.splitext(file)
        if ext == '.json':
            with open(file, 'r') as f:
                attr_dict: Dict = json.load(f)
        elif ext == '.pkl':
            with open(file, 'rb') as f:
                attr_dict: Dict = pickle.load(f)
        else:
            raise ValueError(f"Unsupported file format: {ext}")
        length_list = [len(value) for value in attr_dict.values()]
        assert min(length_list) == max(length_list), f'{attr_type} attributes are not of the same length.'
        assert min(length_list) == length, f'{attr_type} attributes and SDF file are not of the same length.'
    return attr_dict

def complete_with_dist_filter(data: Data, distance_threshold: float|None=None):
    if data.pos is None:
        raise ValueError("Position must be provided.")

    device = data.edge_index.device

    # Generate all possible edges between nodes
    row = torch.arange(data.num_nodes, dtype=torch.long, device=device).view(-1, 1).repeat(1, data.num_nodes).view(-1)
    col = torch.arange(data.num_nodes, dtype=torch.long, device=device).repeat(data.num_nodes)

    # Calculate distances for all edges
    dist = torch.norm(data.pos[col] - data.pos[row], p=2, dim=-1)

    # Apply distance threshold to filter out long-range connections and remove self-loops
    mask = (dist <= distance_threshold) & (row != col) if distance_threshold is not None else (row != col)
    row, col = row[mask], col[mask]

    edge_index = torch.stack([row, col], dim=0)

    # Handle original edge attributes if they exist
    edge_attr = None
    if data.edge_attr is not None:
        idx = data.edge_index[0] * data.num_nodes + data.edge_index[1]
        size = list(data.edge_attr.size())
        size[0] = data.num_nodes * data.num_nodes
        edge_attr = data.edge_attr.new_zeros(size)
        edge_attr[idx] = data.edge_attr
        edge_attr = edge_attr[mask]

    data.edge_attr = edge_attr
    data.edge_index = edge_index

    return data

def power_dist(
    data: Data,
    cat: bool=True,
    power: float=1.0,
    distance_threshold: float | None=None
    ):
    if data.pos is None or data.edge_index is None:
        raise ValueError("Position and edge index must be provided.")
    (row, col), pos, pseudo = data.edge_index, data.pos, data.edge_attr

    # Calculate Euclidean distance and then take the inverse
    dist = torch.norm(pos[col] - pos[row], p=2, dim=-1).view(-1, 1)
    dist = apply_distance_threshold(dist, distance_threshold)
    powered_dist = dist ** power

    if pseudo is not None and cat:
        pseudo = pseudo.view(-1, 1) if pseudo.dim() == 1 else pseudo
        data.edge_attr = torch.cat([pseudo, powered_dist.type_as(pseudo)], dim=-1)
    else:
        data.edge_attr = powered_dist

    return data


def apply_distance_threshold(dist: torch.Tensor, threshold: float | None=None) -> torch.Tensor:
    """
    Applies a threshold to the calculated distances. Distances greater than
    the threshold will be set to infinity.

    Args:
        dist (torch.Tensor): The calculated distances.
        threshold (float): The threshold value. If None, no changes are made.
    Returns:
        torch.Tensor: The modified distances with values exceeding the threshold set to infinity.
    """
    if threshold is not None:
        mask = dist > threshold
        dist[mask] = float('inf')
    return dist

