import torch
import torch.nn as nn
from torch_geometric.loader import DataLoader

class Evaluation():
    def __init__(
        self,
        DataLoader: DataLoader,
        model: nn.Module,
        param: dict,
        device: torch.device,
        norm_dict: dict[str, tuple[torch.Tensor, torch.Tensor]],
        transform: str | None = None,
        ) -> None:
        self.param = param
        self.DataLoader = DataLoader
        self.model = model
        self.device = device
        mean, std = norm_dict['y']
        self.mean = mean.to('cpu')
        self.std = std.to('cpu')
        self.transform = transform
        self.pred, self.target = self._get_pred()

    def _get_pred(self) -> tuple[torch.Tensor, torch.Tensor]:
        self.model.eval()
        sum_pred = []
        sum_target = []
        with torch.no_grad():
            for data in self.DataLoader:
                data = data.to(self.device)
                output = self.model(data)
                y = data.y

                output = output.cpu()
                y = y.cpu()

                sum_pred.append(output)
                sum_target.append(y)

                del output, y

        sum_pred = torch.cat(sum_pred)
        sum_target = torch.cat(sum_target)

        if not self.param['target_type'] == 'vector':
            sum_pred = sum_pred * self.std + self.mean
            sum_pred = self.target_inverse_transform(sum_pred)
            sum_target = sum_target * self.std + self.mean
            sum_target = self.target_inverse_transform(sum_target)

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
