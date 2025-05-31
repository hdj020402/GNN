import torch
import math
from typing import Dict, Callable

class SaveModel():
    def __init__(
        self,
        norm_dict: dict[str, tuple[torch.Tensor, torch.Tensor]],
        param: Dict,
        model_dir: str,
        ckpt_dir: str,
        trace_func: Callable
        ) -> None:
        self.best_val_loss = math.inf
        self.best_epoch = 0
        self.norm_dict = norm_dict
        self.param = param
        self.model_dir = model_dir
        self.ckpt_dir = ckpt_dir
        self.early_stopping = EarlyStopping(**self.param['early_stopping'], trace_func=trace_func)

    def best_model(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        epoch: int,
        val_loss: float,
        ) -> Dict:
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
                f"{self.model_dir}/best_model_{self.param['time']}.pth"
                )

    def regular_model(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        epoch: int,
        ) -> None:
        if epoch % self.param['model_save_step'] == 0:
            state_dict = {
                'norm': self.norm_dict,
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': epoch,
                'val_loss': self.val_loss
            }
            torch.save(
                state_dict,
                f"{self.ckpt_dir}/ckpt_{self.param['time']}_{epoch:0{len(str(self.param['epoch_num']))}d}.pth"
                )

    def check_early_stopping(self):
        self.early_stopping(self.val_loss)
        return self.early_stopping.early_stop

class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=7, delta=0, trace_func=print) -> None:
        self.patience = patience
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.delta = delta
        self.trace_func = trace_func

    def __call__(self, val_loss) -> None:
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
