import torch
from typing import Dict, Callable

class calc_error():
    def __init__(self, pred: torch.Tensor, target: torch.Tensor) -> None:
        self.pred = pred
        self.target = target

    def MAE(self, dim: int | None) -> torch.Tensor:
        return (self.pred - self.target).abs().mean(dim=dim)

    def MSE(self, dim: int | None) -> torch.Tensor:
        return ((self.pred - self.target) ** 2).mean(dim=dim)

    def RMSD(self, dim: int | None) -> torch.Tensor:
        return self.MSE(dim=dim) ** 0.5

    def S2(self, dim: int | None) -> torch.Tensor:
        return ((self.target - self.target.mean()) ** 2).mean(dim=dim)

    def R2(self, dim: int | None) -> torch.Tensor:
        return 1 - self.MSE(dim=dim) / self.S2(dim=dim)

    def AARD(self, dim: int | None) -> torch.Tensor:
        return ((self.target - self.pred) / self.target).abs().mean(dim=dim)

    def RD_each(self) -> torch.Tensor:
        return (self.pred - self.target) / self.target

    def MRD(self):
        if abs(max(self.RD_each())) > abs(min(self.RD_each())):
            return max(self.RD_each())
        else:
            return min(self.RD_each())
