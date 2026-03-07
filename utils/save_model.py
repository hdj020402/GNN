import torch
import math
from typing import Callable

from omegaconf import OmegaConf

from configs.schema import AppConfig


class SaveModel():
    def __init__(
        self,
        norm_dict: dict[str, tuple[torch.Tensor, torch.Tensor]],
        cfg: AppConfig,
        model_dir: str,
        ckpt_dir: str,
        trace_func: Callable
        ) -> None:
        self.best_val_loss = math.inf
        self.best_epoch = 0
        self.val_loss = math.inf
        self.norm_dict = norm_dict
        self.cfg = cfg
        self.model_dir = model_dir
        self.ckpt_dir = ckpt_dir
        self.early_stopping = EarlyStopping(
            **OmegaConf.to_container(cfg.training.early_stopping, resolve=True),
            trace_func=trace_func
        )

    def best_model(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        epoch: int,
        val_loss: float,
        ) -> None:
        self.val_loss = val_loss
        if val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            self.best_epoch = epoch
            state_dict = {
                'norm': self.norm_dict,
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': epoch,
                'val_loss': val_loss
            }
            torch.save(
                state_dict,
                f"{self.model_dir}/best_model_{self.cfg.timestamp}.pth"
                )

    def regular_model(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        epoch: int,
        ) -> None:
        if epoch % self.cfg.training.model_save_step == 0:
            state_dict = {
                'norm': self.norm_dict,
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': epoch,
                'val_loss': self.val_loss
            }
            digits = len(str(self.cfg.training.epoch_num))
            torch.save(
                state_dict,
                f"{self.ckpt_dir}/ckpt_{self.cfg.timestamp}_{epoch:0{digits}d}.pth"
                )

    def check_early_stopping(self) -> bool:
        self.early_stopping(self.val_loss)
        return self.early_stopping.early_stop


class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience: int = 7, delta: float = 0, trace_func: Callable = print) -> None:
        self.patience = patience
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.delta = delta
        self.trace_func = trace_func

    def __call__(self, val_loss: float) -> None:
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
                self.early_stop = True
        else:
            self.best_score = score
            self.counter = 0
