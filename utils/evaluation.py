import torch
import torch.nn as nn
from torch_geometric.loader import DataLoader
from typing import Tuple

class Evaluation():
    def __init__(
        self,
        DataLoader: DataLoader,
        model: nn.Module,
        device: torch.device,
        mean: torch.Tensor,
        std: torch.Tensor,
        transform: str | None = None
        ) -> None:
        self.DataLoader = DataLoader
        self.model = model
        self.device = device
        self.mean = mean.to(self.device)
        self.std = std.to(self.device)
        self.transform = transform
        self.pred, self.target = self._get_pred()

    def _get_pred(self) -> Tuple[torch.Tensor, torch.Tensor]:
        self.model.eval()
        sum_pred = []
        sum_target = []
        with torch.no_grad():
            for data in self.DataLoader:
                data = data.to(self.device)
                output = self.model(data)
                y = data.y

                output = (output * self.std + self.mean).cpu()
                y = (y * self.std + self.mean).cpu()

                output = self.target_inverse_transform(output)
                y = self.target_inverse_transform(y)

                sum_pred.append(output)
                sum_target.append(y)

                del output, y

        sum_pred = torch.cat(sum_pred)
        sum_target = torch.cat(sum_target)

        return sum_pred, sum_target

    def target_inverse_transform(self, data: torch.tensor):
        if self.transform == 'LN':
            return torch.e ** data
        elif self.transform == 'LG':
            return 10 ** data
        elif self.transform == 'E^-x':
            return -torch.log(data)
        elif not self.transform:
            return data
