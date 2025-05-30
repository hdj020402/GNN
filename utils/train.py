from typing import Union, Literal
from torch_geometric.loader import DataLoader
import torch
import torch.nn.functional as F
from torch.optim import AdamW, Adam, SGD

def weighted_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    weight: torch.Tensor,
    loss_type: Literal['MAE', 'MSE', 'Cosine']
    ) -> torch.Tensor:
    if loss_type == 'MAE':
        return (weight * torch.abs(pred - target)).mean()
    elif loss_type == 'MSE':
        return (weight * (pred - target) ** 2).mean()
    elif loss_type == 'Cosine':
        cosine_sim = F.cosine_similarity(pred, target, dim=1)
        loss = 1 - cosine_sim
        return (weight * loss).mean()

def unweighted_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    loss_type: Literal['MAE', 'MSE', 'Cosine']
    ) -> torch.Tensor:
    if loss_type == 'MAE':
        return torch.abs(pred - target).mean()
    elif loss_type == 'MSE':
        return ((pred - target) ** 2).mean()
    elif loss_type == 'Cosine':
        cosine_sim = F.cosine_similarity(pred, target, dim=1)
        loss = 1 - cosine_sim
        return loss.mean()

def train(
    model: torch.nn.Module,
    train_loader: DataLoader,
    optimizer: Union[AdamW, SGD, Adam],
    loss_fn: Literal['MAE', 'MSE', 'Cosine'],
    device: torch.device,
    accumulation_steps: int = 1
    ) -> float:
    model.train()
    loss_all = 0

    for i, data in enumerate(train_loader):
        data = data.to(device)
        output = model(data)
        loss = weighted_loss(output, data.y, data.weight, loss_fn)
        loss_all += loss.item() * data.num_graphs
        loss = loss / accumulation_steps
        loss.backward()

        if (i + 1) % accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad()

    if (i + 1) % accumulation_steps != 0:
        optimizer.step()
        optimizer.zero_grad()

    return loss_all / len(train_loader.dataset)

@torch.no_grad()
def validate(
    model: torch.nn.Module,
    loader: DataLoader,
    loss_fn: Literal['MAE', 'MSE', 'Cosine'],
    device: torch.device,
    ) -> float:
    model.eval()
    loss_all = 0

    for data in loader:
        data = data.to(device)
        output = model(data)
        loss = unweighted_loss(output, data.y, loss_fn)
        loss_all += loss.item() * data.num_graphs
    
    return loss_all / len(loader.dataset)
