import os, time, yaml, json
import torch, optuna
import pandas as pd
from functools import partial
from typing import Dict

from utils.data_processing import DataProcessing
from utils.reprocess import reprocess
from utils.gen_model import gen_model, gen_optimizer, gen_scheduler
from utils.setup_seed import setup_seed
from utils.plot import loss_epoch, scatter
from utils.plot_model import scatterFromModel
from utils.post_processing import ReadLog
from utils.evaluation import Evaluation
from utils.calc_error import calc_error
from utils.attr_filter import attr_filter
from utils.optuna_setup import OptunaSetup
from utils.train import train, validate
from utils.file_processing import FileProcessing
from utils.save_model import SaveModel
from utils.utils import extract_keys_and_lists, Timer
from utils.gpu_monitor import GPUMonitor

def training(param: Dict, ht_param: Dict | None = None, trial: optuna.Trial | None = None) -> float:
    fp = FileProcessing(param, ht_param, trial)
    fp.pre_make()
    plot_dir, model_dir, ckpt_dir = fp.plot_dir, fp.model_dir, fp.ckpt_dir
    log_file = fp.log_file
    training_logger = fp.training_logger
    gpu_logger = fp.gpu_logger
    gpu_monitor = GPUMonitor(gpu_logger)
    gpu_monitor.start()
    error_dict = fp.error_dict

    epoch_num = param['epoch_num']

    dp_timer = Timer()
    dp_timer.start()
    DATA = DataProcessing(param, reprocess = reprocess(param))
    dataset = DATA.dataset
    norm_dict = DATA.norm_dict
    train_loader = DATA.train_loader
    val_loader = DATA.val_loader
    test_loader = DATA.test_loader
    dp_timer.end()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # if device == torch.device('cuda'):
    #     torch.cuda.set_per_process_memory_fraction(param['GPU_memo_frac'], device = 0)
    model = gen_model(param, dataset)
    optimizer = gen_optimizer(param, model)
    scheduler = gen_scheduler(param, optimizer)

    fp.basic_info_log(
        dataset, train_loader, val_loader, test_loader, None,
        norm_dict, model, dp_timer
        )

    criteria_set = set(param['criteria_list'] + [param['loss_fn']])

    eval_class = partial(
        Evaluation, param = param, device = device, norm_dict=norm_dict,
        transform = param['target_transform'])

    model_saving = SaveModel(norm_dict, param, model_dir, ckpt_dir, training_logger.info)
    start_epoch = fp.pre_train(model, optimizer, device, model_saving)

    timer = Timer()
    timer.start()
    for epoch in range(start_epoch, epoch_num+1):
        try:
            lr = scheduler.optimizer.param_groups[0]['lr']
            loss = train(model, train_loader, optimizer, param['loss_fn'], device, param['accumulation_step'])
            val_loss = validate(model, val_loader, param['loss_fn'], device)
            error_dict['Overall']['LR'] = round(lr, 7)
            error_dict['Overall']['Loss'] = round(loss, 7)
            for phase, loader in zip(['Train', 'Val', 'Test'], [train_loader, val_loader, test_loader]):
                evaluation = eval_class(loader, model)
                pred, target = evaluation.pred, evaluation.target
                for criteria in criteria_set:
                    errors = torch.cat([
                        getattr(calc_error(pred, target), criteria)(dim=None).unsqueeze(0),
                        getattr(calc_error(pred, target), criteria)(dim=0).view(-1)
                        ])
                    for subtask, error in zip(error_dict.keys(), errors):
                        error_dict[subtask][phase][criteria] = round(float(error), 7)
            scheduler.step(val_loss)

            info = json.dumps({'Epoch': epoch} | error_dict)

            model_saving.best_model(model, optimizer, epoch, val_loss)
            model_saving.regular_model(model, optimizer, epoch)
            fp.training_log(epoch, info, model_saving.best_val_loss, model_saving.best_epoch)
            torch.cuda.empty_cache()

            if trial is not None:
                trial.report(val_loss, epoch)
                if trial.should_prune():
                    raise optuna.exceptions.TrialPruned()

            if model_saving.check_early_stopping():
                break

        except torch.cuda.OutOfMemoryError as e:
            training_logger.error(e)
            break
    timer.end()

    fp.ending_log(timer, epoch)
    gpu_monitor.stop()

    scatterFromModel(
        f'{model_dir}/best_model_{param["time"]}.pth',
        param, DATA, plot_dir
        )

    log_info_dict = ReadLog(log_file, param).get_performance()
    info_pairs = extract_keys_and_lists(log_info_dict)
    for item, data in info_pairs:
        if item == 'Epoch':
            continue
        loss_epoch(
            [[log_info_dict['Epoch'], data]],
            [f'{item}-Epoch'],
            ['#03658C'],
            'Epoch',
            f'{item}',
            f'{plot_dir}/{item.split("_")[0]}/{item}-Epoch_{param["time"]}.png'
            )
        pd.DataFrame({'Epoch': log_info_dict['Epoch'], item: data}).to_csv(
            f'{plot_dir}/{item.split("_")[0]}/{item}-Epoch_{param["time"]}.csv', index=False
        )

    return model_saving.best_val_loss

def prediction(param: Dict) -> None:
    fp = FileProcessing(param)
    fp.pre_make()
    plot_dir, data_dir, model_dir = fp.plot_dir, fp.data_dir, fp.model_dir
    prediction_logger = fp.prediction_logger
    gpu_logger = fp.gpu_logger
    gpu_monitor = GPUMonitor(gpu_logger)
    gpu_monitor.start()
    subtasks = fp.subtasks
    error_dict = fp.error_dict

    dp_timer = Timer()
    dp_timer.start()
    DATA = DataProcessing(param, reprocess = reprocess(param))
    dataset = DATA.dataset
    mean, std = DATA.mean, DATA.std
    train_loader = DATA.train_loader
    val_loader = DATA.val_loader
    test_loader = DATA.test_loader
    pred_loader = DATA.pred_loader
    dp_timer.end()

    loader_dict = {'train': train_loader, 'val': val_loader, 'test': test_loader, 'whole': pred_loader}
    try:
        loader = loader_dict[param['dataset_range']]
    except KeyError:
        loader = pred_loader

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = gen_model(param, dataset)
    optimizer = gen_optimizer(param, model)
    fp.pre_train(model, optimizer, device)

    len_target = len(param['target_list'])
    fp.basic_info_log(
        dataset, None, None, None, loader,
        mean[-len_target:], std[-len_target:], model, dp_timer
        )

    criteria_list = param['criteria_list']

    evaluation = Evaluation(loader, model, device, mean[-len_target:], std[-len_target:], param['target_transform'])
    pred, target = evaluation.pred, evaluation.target
    gpu_monitor.stop()

    for criteria in criteria_list:
        errors = torch.cat([
            getattr(calc_error(pred, target), criteria)(dim=None).unsqueeze(0),
            getattr(calc_error(pred, target), criteria)(dim=0)
            ])
        for subtask, error in zip(error_dict.keys(), errors):
            error_dict[subtask]['Pred'][criteria] = round(float(error), 7)
    fp.pred_log(error_dict)

    for subtask, idx in zip(subtasks, range(-1, len_target)):
        if subtask == 'Overall':
            torch.save(pred, f'{data_dir}/{subtask}/pred.pt')
            torch.save(target, f'{data_dir}/{subtask}/target.pt')
        else:
            torch.save(pred[:, idx], f'{data_dir}/{subtask}/pred.pt')
            torch.save(target[:, idx], f'{data_dir}/{subtask}/target.pt')

    for t, p, task in zip(torch.split(target, 1, dim=1), torch.split(pred, 1, dim=1), param['target_list']):
        scatter(
            [t, p],
            scatter_label = 'eval',
            output_path = f"{plot_dir}/{task}/{param['dataset_range']}.png",
            )

def hparam_tuning(param: Dict, ht_param: Dict[str, Dict]) -> None:
    jobtype = param['jobtype']
    fp = FileProcessing(param)
    fp.pre_make()
    storage_name = fp.optuna_db
    hptuning_logger = fp.hptuning_logger

    def hparam_optim(param: Dict, ht_param: Dict[str, Dict], trial: optuna.Trial) -> float:
        SUGGEST_METHOD_MAP = {
            'int': trial.suggest_int,
            'float': trial.suggest_float,
            'uniform': trial.suggest_uniform,
            'discrete_uniform': trial.suggest_discrete_uniform,
            'loguniform': trial.suggest_loguniform,
            'categorical': trial.suggest_categorical,
            }
        for hparam, attr in ht_param.items():
            if hparam == 'optuna':
                continue
            suggest_method = SUGGEST_METHOD_MAP[attr['type']]
            sm_kwargs = {k: v for k, v in attr.items() if k != 'type'}
            param[hparam] = suggest_method(hparam, **sm_kwargs)
        criteria = training(param, ht_param, trial)
        return criteria

    optuna_setup = OptunaSetup(param, ht_param)
    optuna_setup.logging_setup(hptuning_logger)
    study = optuna_setup.create_study(f'hptuning_{jobtype}', storage_name)
    study.optimize(
        lambda trial: hparam_optim(param, ht_param, trial),
        n_trials=ht_param['optuna']['n_trials']
        )

    fp.hptuning_log(study)

def main():
    """
    Main function to execute the script based on the mode specified in the configuration file.
    """
    TIME = time.strftime('%b_%d_%Y_%H%M%S', time.localtime())
    with open('model_parameters.yml', 'r', encoding='utf-8') as mp:
        param: Dict = yaml.full_load(mp)
    param['time'] = TIME

    seed = param['seed']
    setup_seed(seed)
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    torch.use_deterministic_algorithms(True)

    if param['mode'] in ['training', 'fine-tuning']:
        training(param)
    elif param['mode'] == 'hparam_tuning':
        with open('hparam_tuning.yml', 'r', encoding='utf-8') as ht:
            ht_param: Dict[str, Dict] = yaml.full_load(ht)
        hparam_tuning(param, ht_param)
    elif param['mode'] == 'feature_filtration':
        if param['feature_filter_mode'] == 'one_by_one':
            attr_filter(training, param)
        elif param['feature_filter_mode'] == 'file':
            with open('feature_filter.yml') as ff:
                feature_dict: Dict = yaml.full_load(ff)
            for idx, feature in feature_dict.items():
                param['node_attr_list'] = feature['node_attr_list']
                param['edge_attr_list'] = feature['edge_attr_list']
                param['graph_attr_list'] = feature['graph_attr_list']
                training(param)
        else:
            raise ValueError('Wrong feature_filter_mode! Please check `model_parameters.yml`.')
    elif param['mode'] == 'prediction':
        prediction(param)
    else:
        raise ValueError('Invalid mode specified in `model_parameters.yml`. Please check the configuration.')

if __name__ == '__main__':
    main()
