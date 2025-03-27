from typing import Union, Literal
from torch_geometric.loader import DataLoader
import torch
from torch.optim import AdamW, Adam, SGD

def weighted_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    weight: torch.Tensor,
    loss_type: Literal['MAE', 'MSE']
    ) -> torch.Tensor:
    if loss_type == 'MAE':
        return (weight * torch.abs(pred - target)).mean()
    elif loss_type == 'MSE':
        return (weight * (pred - target) ** 2).mean()

def train(
    model: torch.nn.Module,
    train_loader: DataLoader,
    optimizer: Union[AdamW, SGD, Adam],
    loss_fn: Literal['MAE', 'MSE'],
    device: torch.device,
) -> float:
    model.train()
    loss_all = 0

    for data in train_loader:
        data = data.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = weighted_loss(output, data.y, data.weight, loss_fn)
        loss.backward()
        loss_all += loss.item() * data.num_graphs
        optimizer.step()

    return loss_all / len(train_loader.dataset)
