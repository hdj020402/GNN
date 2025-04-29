import torch
import os
from typing import Dict
from functools import partial
from utils.gen_model import gen_model
from utils.data_processing import DataProcessing
from utils.evaluation import Evaluation
from utils.plot import scatter

def scatterFromModel(model_path: str, param: Dict, DATA: DataProcessing, output_dir: str):
    train_loader = DATA.train_loader
    val_loader = DATA.val_loader
    test_loader = DATA.test_loader

    model = gen_model(param, DATA.dataset)
    try:
        _model = torch.load(model_path, map_location = 'cuda' if torch.cuda.is_available() else 'cpu')
    except FileNotFoundError:
        return
    if model_path.endswith('pkl'):
        model = _model
    elif model_path.endswith('pth'):
        model.load_state_dict(_model['model'])
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.eval()

    len_target = len(param['target_list'])
    eval_class = partial(
        Evaluation,
        model = model,
        device = device,
        mean = DATA.mean[-len_target:],
        std = DATA.std[-len_target:],
        transform = param['target_transform']
        )
    train_eval = eval_class(train_loader)
    val_eval = eval_class(val_loader)
    test_eval = eval_class(test_loader)
    eval_dict = {
        'train': train_eval,
        'val': val_eval,
        'test': test_eval
        }

    file_name = os.path.splitext(os.path.basename(model_path))[0]
    for key, value in eval_dict.items():
        for target, pred, task in zip(torch.split(value.target, 1, dim=1), torch.split(value.pred, 1, dim=1), param['target_list']):
            scatter(
                [target, pred],
                scatter_label = [key],
                output_path = os.path.join(output_dir, f'{task}/{file_name}_{task}_{key}.png'),
                )
