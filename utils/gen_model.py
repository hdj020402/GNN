import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau
from typing import Dict
from nets.readout_add_graph_feature import GraphPredictionModel, NodePredictionModel

def gen_model(param: Dict, dataset, ) -> GraphPredictionModel | NodePredictionModel:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if param['target_type'] == 'graph':
        net = GraphPredictionModel(
            dataset, param['dim_linear'],
            param['dim_conv'],
            len(param['target_list']),
            param['processing_steps'],
            param['mp_times'],
            )
    elif param['target_type'] == 'node':
        net = NodePredictionModel(
            dataset, param['dim_linear'],
            param['dim_conv'],
            len(param['target_list']),
            param['mp_times'],
            )
    elif param['target_type'] == 'edge':
        ...   # TODO: To be implemented
    model = net.to(device)
    return model

def gen_optimizer(param: Dict, model: GraphPredictionModel | NodePredictionModel) -> torch.optim.Optimizer:
    optimizer = getattr(torch.optim, param['optimizer'])(model.parameters(), lr = param['lr'])
    return optimizer

def gen_scheduler(param: Dict, optimizer: torch.optim.Optimizer) -> ReduceLROnPlateau | None:
    if param['scheduler']['type'] == 'ReduceLROnPlateau':
        scheduler = ReduceLROnPlateau(
            optimizer, mode = 'min',
            factor = param['scheduler']['factor'], patience = param['scheduler']['patience'],
            min_lr = param['scheduler']['min_lr']
            )
    else:
        print('No scheduler')
        scheduler = None
    return scheduler
