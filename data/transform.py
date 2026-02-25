import torch
from torch_geometric.data import Data
from torch_geometric.utils import remove_self_loops
from torch_geometric.transforms import BaseTransform
from torch_geometric.data.datapipes import functional_transform
from typing import Optional, Tuple

class MyTransform(object):
    def __call__(self, data):
        # Specify target.
        data.y = data.y[:, 0]  # only one target
        return data

class Complete(object):
    def __call__(self, data):
        device = data.edge_index.device

        row = torch.arange(data.num_nodes, dtype=torch.long, device=device)
        col = torch.arange(data.num_nodes, dtype=torch.long, device=device)

        row = row.view(-1, 1).repeat(1, data.num_nodes).view(-1)
        col = col.repeat(data.num_nodes)
        edge_index = torch.stack([row, col], dim=0)

        edge_attr = None
        if data.edge_attr is not None:
            idx = data.edge_index[0] * data.num_nodes + data.edge_index[1]
            size = list(data.edge_attr.size())
            size[0] = data.num_nodes * data.num_nodes
            edge_attr = data.edge_attr.new_zeros(size)
            edge_attr[idx] = data.edge_attr

        edge_index, edge_attr = remove_self_loops(edge_index, edge_attr)
        data.edge_attr = edge_attr
        data.edge_index = edge_index

        return data

class CompleteWithDistanceFilter(object):
    def __init__(self, distance_threshold: float|None=None):
        self.distance_threshold = distance_threshold
        
    def __call__(self, data):
        if data.pos is None:
            raise ValueError("Position must be provided.")
        
        device = data.edge_index.device
        
        # Generate all possible edges between nodes
        row = torch.arange(data.num_nodes, dtype=torch.long, device=device).view(-1, 1).repeat(1, data.num_nodes).view(-1)
        col = torch.arange(data.num_nodes, dtype=torch.long, device=device).repeat(data.num_nodes)

        # Calculate distances for all edges
        dist = torch.norm(data.pos[col] - data.pos[row], p=2, dim=-1)
        
        # Apply distance threshold to filter out long-range connections and remove self-loops
        mask = (dist <= self.distance_threshold) & (row != col) if self.distance_threshold is not None else (row != col)
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

@functional_transform('powered_distance')
class PowerDistance(BaseTransform):
    r"""Saves the powered Euclidean distance of linked nodes in its edge attributes 
    (functional name: :obj:`powered_distance`). Each distance is raised to a specified power and 
    then globally normalized to a specified interval (:math:`[0, 1]` by default).

    Args:
        norm (bool, optional): If set to :obj:`False`, the output will not be
            normalized. (default: :obj:`True`)
        max_value (float, optional): If set and :obj:`norm=True`, normalization
            will be performed based on this value instead of the maximum value
            found in the data. (default: :obj:`None`)
        cat (bool, optional): If set to :obj:`False`, all existing edge
            attributes will be replaced. (default: :obj:`True`)
        interval ((float, float), optional): A tuple specifying the lower and
            upper bound for normalization. (default: :obj:`(0.0, 1.0)`)
        power (float): Raise the distance to a specific power. (default: :obj:`1.0`)
    """
    def __init__(
            self,
            norm: bool = True,
            max_value: Optional[float] = None,
            cat: bool = True,
            interval: Tuple[float, float] = (0.0, 1.0),
            power: float = 1.0,
            distance_threshold: float | None = None
            ) -> None:
        self.norm = norm
        self.max_value = max_value
        self.cat = cat
        self.interval = interval
        self.power = power
        self.distance_threshold = distance_threshold

    def forward(self, data: Data) -> Data:
        if data.pos is None or data.edge_index is None:
            raise ValueError("Position and edge index must be provided.")
        (row, col), pos, pseudo = data.edge_index, data.pos, data.edge_attr

        # Calculate Euclidean distance and then take the inverse
        dist = torch.norm(pos[col] - pos[row], p=2, dim=-1).view(-1, 1)
        dist = self.apply_distance_threshold(dist, self.distance_threshold)
        powered_dist = dist ** self.power

        if self.norm and powered_dist.numel() > 0:
            max_val = float(powered_dist.max()) if self.max_value is None else self.max_value

            length = self.interval[1] - self.interval[0]
            powered_dist = length * (powered_dist / max_val) + self.interval[0]

        if pseudo is not None and self.cat:
            pseudo = pseudo.view(-1, 1) if pseudo.dim() == 1 else pseudo
            data.edge_attr = torch.cat([pseudo, powered_dist.type_as(pseudo)], dim=-1)
        else:
            data.edge_attr = powered_dist

        return data

    def apply_distance_threshold(self, dist: torch.Tensor, threshold: float | None=None) -> torch.Tensor:
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

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}(norm={self.norm}, '
                f'max_value={self.max_value})')