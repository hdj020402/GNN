import torch
from typing import Dict, Callable

class SaveModel():
    def __init__(
        self,
        mean: torch.Tensor,
        std: torch.Tensor,
        param: Dict,
        trace_func: Callable
        ) -> None:
        self.mean = mean
        self.std = std
        self.param = param
        self.early_stopping = EarlyStopping(**self.param['early_stopping'], trace_func=trace_func)

    def best_model(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        epoch: int,
        info: str,
        error_dict: Dict,
        criteria_info_dict: Dict,
        model_dir: str
        ) -> Dict:
        criteria = self.param['optim_criteria']
        self.val_loss = error_dict['Overall']['Val'][criteria]
        if self.val_loss < criteria_info_dict[criteria]['best_error']:
            criteria_info_dict[criteria]['best_error'] = error_dict['Overall']['Val'][criteria]
            criteria_info_dict[criteria]['best_epoch'] = epoch
            criteria_info_dict[criteria]['best_epoch_info'] = info
            state_dict = {
                'mean': self.mean,
                'std': self.std,
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': epoch
            }
            torch.save(
                state_dict,
                f"{model_dir}/best_model_{criteria}_{self.param['time']}.pth"
                )
        return criteria_info_dict

    def regular_model(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        epoch: int,
        ckpt_dir: str
        ) -> None:
        if epoch % self.param['model_save_step'] == 0:
            state_dict = {
                'mean': self.mean,
                'std': self.std,
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': epoch
            }
            torch.save(
                state_dict,
                f"{ckpt_dir}/ckpt_{self.param['time']}_{epoch:0{len(str(self.param['epoch_num']))}d}.pth"
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
