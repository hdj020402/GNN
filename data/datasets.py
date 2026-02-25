import torch
import pandas as pd
import numpy as np
from torch.utils.data import Subset
from torch_geometric.data import Data, InMemoryDataset
from typing import List, Optional, Callable

class CustomSubset(Subset):
    """A custom subset class that retains the 'graph_attr' and 'y' attributes."""
    def __init__(self, dataset, indices) -> None:
        super().__init__(dataset, indices)

        self.graph_attr = dataset.data.graph_attr[indices]
        self.y = dataset.data.y[indices]

        if not isinstance(self.graph_attr, torch.Tensor):
            self.graph_attr = torch.as_tensor(self.graph_attr)
        if not isinstance(self.y, torch.Tensor):
            self.y = torch.as_tensor(self.y)

    def __getitem__(self, idx):
        return self.graph_attr[idx], self.y[idx]

    def __len__(self) -> int:
        return len(self.indices)