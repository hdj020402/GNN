import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau

from omegaconf import OmegaConf

from configs.schema import AppConfig
from models.factory import create_model


def gen_model(cfg: AppConfig, dataset) -> torch.nn.Module:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    backbone_d: dict = OmegaConf.to_container(cfg.model.backbone, resolve=True)
    backbone_name = backbone_d.pop('name')
    head_d: dict = OmegaConf.to_container(cfg.model.head, resolve=True)
    head_d.pop('name')
    model = create_model(
        dataset=dataset,
        target_type=cfg.data.target_type,
        num_targets=len(cfg.data.target_list),
        backbone_name=backbone_name,
        backbone_cfg=backbone_d,
        head_cfg=head_d,
        device=device,
    )
    return model


def gen_optimizer(cfg: AppConfig, model: torch.nn.Module) -> torch.optim.Optimizer:
    return getattr(torch.optim, cfg.training.optimizer)(model.parameters(), lr=cfg.training.lr)


def gen_scheduler(cfg: AppConfig, optimizer: torch.optim.Optimizer) -> ReduceLROnPlateau | None:
    s = cfg.training.scheduler
    if s.type == 'ReduceLROnPlateau':
        return ReduceLROnPlateau(
            optimizer, mode='min',
            factor=s.factor,
            patience=s.patience,
            min_lr=s.min_lr,
        )
    return None
