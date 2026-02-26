import os, time, yaml, json
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
os.environ["PYTHONHASHSEED"] = "0"
import torch, optuna
from functools import partial

import hydra
from omegaconf import DictConfig, OmegaConf
from torch.utils.tensorboard import SummaryWriter

from utils.data_processing import DataProcessing
from utils.gen_model import gen_model, gen_optimizer, gen_scheduler
from utils.setup_seed import setup_seed
from utils.plot import scatter
from utils.plot_model import scatterFromModel
from utils.evaluation import Evaluation
from utils.calc_error import calc_error
from utils.optuna_setup import create_study, redirect_optuna_log
from utils.train import train, validate
from utils.file_processing import FileProcessing
from utils.save_model import SaveModel
from utils.utils import Timer
from utils.gpu_monitor import GPUMonitor

def training(param: dict, trial: optuna.Trial | None = None) -> float:
    fp = FileProcessing(param, trial=trial)
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
    DATA = DataProcessing(param)
    dataset = DATA.dataset
    norm_dict = DATA.norm_dict
    train_loader = DATA.train_loader
    val_loader = DATA.val_loader
    test_loader = DATA.test_loader
    dp_timer.end()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = gen_model(param, dataset)
    optimizer = gen_optimizer(param, model)
    scheduler = gen_scheduler(param, optimizer)

    fp.basic_info_log(
        dataset, train_loader, val_loader, test_loader, None,
        norm_dict, model, dp_timer
        )

    criteria_set = set(param['criteria_list'] + [param['loss_fn']])
    # For vector targets only Cosine is meaningful; decide once outside the loop
    phase_criteria = {'Cosine'} if param['target_type'] == 'vector' else criteria_set

    eval_class = partial(
        Evaluation, param = param, device = device, norm_dict=norm_dict,
        transform = param['target_transform'])

    model_saving = SaveModel(norm_dict, param, model_dir, ckpt_dir, training_logger.info)
    start_epoch = fp.pre_train(model, optimizer, device, model_saving)

    writer = SummaryWriter(log_dir=fp.tensorboard_dir)

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
                for criteria in phase_criteria:
                    errors = torch.cat([
                        getattr(calc_error(pred, target), criteria)(dim=None).unsqueeze(0),
                        getattr(calc_error(pred, target), criteria)(dim=0).view(-1)
                        ])
                    for subtask, error in zip(error_dict.keys(), errors):
                        error_dict[subtask][phase][criteria] = round(float(error), 7)
            if scheduler is not None:
                scheduler.step(val_loss)

            # ── TensorBoard ────────────────────────────────────────────────
            writer.add_scalar('Loss/train', loss, epoch)
            writer.add_scalar('Loss/val', val_loss, epoch)
            writer.add_scalar('LR', lr, epoch)
            for criteria in phase_criteria:
                for phase in ['Train', 'Val', 'Test']:
                    writer.add_scalar(
                        f'{criteria}/{phase}',
                        error_dict['Overall'][phase][criteria],
                        epoch,
                    )
            # ───────────────────────────────────────────────────────────────

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
    writer.close()

    scatterFromModel(
        f'{model_dir}/best_model_{param["time"]}.pth',
        param, DATA, plot_dir
        )

    return model_saving.best_val_loss

def prediction(param: dict) -> None:
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
    DATA = DataProcessing(param)
    dataset = DATA.dataset
    norm_dict = DATA.norm_dict
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
        norm_dict, model, dp_timer
        )

    criteria_list = param['criteria_list']

    evaluation = Evaluation(loader, model, param, device, norm_dict, param['target_transform'])
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
            torch.save(pred[:, 0, idx], f'{data_dir}/{subtask}/pred.pt')
            torch.save(target[:, 0, idx], f'{data_dir}/{subtask}/target.pt')

    for t, p, task in zip(torch.split(target, 1, dim=-1), torch.split(pred, 1, dim=-1), param['target_list']):
        scatter(
            [t, p],
            scatter_label = 'eval',
            output_path = f"{plot_dir}/{task}/{param['dataset_range']}.png",
            )

def hparam_tuning(param: dict) -> None:
    """Run Optuna hyperparameter search.

    Study settings come from ``param['optuna']`` (cfg.optuna).
    The search space comes from ``param['search_space']`` (cfg.optuna.search_space).
    Each key in search_space is a flat param name (e.g. ``lr``, ``optimizer``).
    """
    import copy

    optuna_cfg = param['optuna']
    search_space = param.get('search_space') or {}

    jobtype = param['jobtype']
    fp = FileProcessing(param)
    fp.pre_make()
    storage_name = fp.optuna_db
    redirect_optuna_log(fp.hptuning_logger)

    # Map type strings to (suggest_method, extra_kwargs_transform) pairs.
    # Optuna 3+ deprecates suggest_loguniform / suggest_discrete_uniform.
    def _suggest(trial: optuna.Trial, name: str, spec: dict):
        stype = spec['type']
        kw = {k: v for k, v in spec.items() if k != 'type'}
        if stype == 'int':
            return trial.suggest_int(name, **kw)
        elif stype == 'float':
            return trial.suggest_float(name, **kw)
        elif stype == 'loguniform':
            return trial.suggest_float(name, kw['low'], kw['high'], log=True)
        elif stype == 'discrete_uniform':
            return trial.suggest_float(name, kw['low'], kw['high'], step=kw['q'])
        elif stype == 'categorical':
            return trial.suggest_categorical(name, kw['choices'])
        else:
            raise ValueError(f"Unknown suggest type '{stype}' for param '{name}'")

    def objective(trial: optuna.Trial) -> float:
        trial_param = copy.deepcopy(param)
        for key, spec in search_space.items():
            trial_param[key] = _suggest(trial, key, spec)
        return training(trial_param, trial=trial)

    study = create_study(optuna_cfg, f'hptuning_{jobtype}', storage_name)
    study.optimize(objective, n_trials=optuna_cfg['n_trials'])
    fp.hptuning_log(study)


def _build_param(cfg: DictConfig) -> dict:
    """Flatten the hierarchical Hydra config into a plain dict.

    All downstream utilities (DataProcessing, FileProcessing, …) expect a
    flat dict with keys like ``param['lr']``, ``param['target_list']``, etc.
    Hydra's DictConfig supports dict-style access, but some utilities call
    ``yaml.dump(param, …)`` which requires a plain Python dict.
    """
    param: dict = {}
    param.update(OmegaConf.to_container(cfg.data, resolve=True))
    param.update(OmegaConf.to_container(cfg.training, resolve=True))
    param.update(OmegaConf.to_container(cfg.output, resolve=True))

    # Root-level settings
    param['mode'] = cfg.mode
    param['seed'] = cfg.seed
    param['GPU_memo_frac'] = cfg.GPU_memo_frac
    param['pretrained_model'] = cfg.pretrained_model

    # Backbone: extract 'name' as the factory key; remaining keys are kwargs
    backbone_d: dict = OmegaConf.to_container(cfg.model.backbone, resolve=True)
    param['backbone'] = backbone_d.pop('name')
    param['backbone_cfg'] = backbone_d

    # Head: extract 'name' (matches target_type); remaining keys are kwargs
    head_d: dict = OmegaConf.to_container(cfg.model.head, resolve=True)
    param['head_name'] = head_d.pop('name')
    param['head_cfg'] = head_d

    # Optuna: study settings and search space (used only when mode=hparam_tuning)
    optuna_d: dict = OmegaConf.to_container(cfg.optuna, resolve=True)
    param['search_space'] = optuna_d.pop('search_space', None) or {}
    param['optuna'] = optuna_d

    return param


@hydra.main(config_path="configs", config_name="config", version_base="1.3")
def main(cfg: DictConfig) -> None:
    """Entry point.  Behaviour is determined by cfg.mode."""
    TIME = time.strftime('%b_%d_%Y_%H%M%S', time.localtime())

    param = _build_param(cfg)
    param['time'] = TIME

    setup_seed(param['seed'])
    torch.use_deterministic_algorithms(True)

    if param['mode'] in ['training', 'fine-tuning']:
        training(param)
    elif param['mode'] == 'hparam_tuning':
        hparam_tuning(param)
    elif param['mode'] == 'prediction':
        prediction(param)
    else:
        raise ValueError(
            f"Invalid mode '{param['mode']}'. Valid: training / hparam_tuning / "
            "prediction / fine-tuning"
        )

if __name__ == '__main__':
    main()
