import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau

from models.factory import create_model


def gen_model(param: dict, dataset) -> torch.nn.Module:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = create_model(
        dataset=dataset,
        target_type=param['target_type'],
        num_targets=len(param['target_list']),
        backbone_name=param.get('backbone', 'mpnn'),
        backbone_cfg=param.get('backbone_cfg'),
        head_cfg=param.get('head_cfg'),
        device=device,
    )
    return model


def gen_optimizer(param: dict, model: torch.nn.Module) -> torch.optim.Optimizer:
    return getattr(torch.optim, param['optimizer'])(model.parameters(), lr=param['lr'])


def gen_scheduler(param: dict, optimizer: torch.optim.Optimizer) -> ReduceLROnPlateau | None:
    sched_cfg = param.get('scheduler', {})
    if sched_cfg.get('type') == 'ReduceLROnPlateau':
        return ReduceLROnPlateau(
            optimizer, mode='min',
            factor=sched_cfg['factor'],
            patience=sched_cfg['patience'],
            min_lr=sched_cfg['min_lr'],
        )
    return None
