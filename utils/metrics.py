import torch
import torch.nn.functional as F

class Metrics():
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
        return ((self.target - self.target.mean(dim=0)) ** 2).mean(dim=dim)

    def R2(self, dim: int | None) -> torch.Tensor:
        return 1 - self.MSE(dim=dim) / self.S2(dim=dim)

    def AARD(self, dim: int | None) -> torch.Tensor:
        return ((self.target - self.pred) / self.target).abs().mean(dim=dim)

    def RD_each(self) -> torch.Tensor:
        return (self.pred - self.target) / self.target

    def MRD(self) -> float:
        rd_each = self.RD_each()
        max_val = torch.max(rd_each).item()
        min_val = torch.min(rd_each).item()
        if abs(max_val) > abs(min_val):
            return max_val
        else:
            return min_val

    def Cosine(self, dim: int | None) -> torch.Tensor:
        return F.cosine_similarity(self.pred, self.target, dim=1).mean(dim=dim)
