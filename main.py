import os
import time
import json
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
os.environ["PYTHONHASHSEED"] = "0"
import torch
import optuna
from copy import deepcopy
from functools import partial
from typing import cast

import hydra
from omegaconf import DictConfig, OmegaConf
from torch.utils.tensorboard import SummaryWriter

from configs.schema import AppConfig
from data.data_processing import DataProcessing
from utils.gen_model import gen_model, gen_optimizer, gen_scheduler
from utils.setup_seed import setup_seed
from utils.visualization import scatter, scatterFromModel
from utils.evaluation import Evaluation
from utils.metrics import Metrics
from utils.optuna_setup import create_study, redirect_optuna_log
from utils.train import train, validate
from utils.file_processing import FileProcessing
from utils.save_model import SaveModel
from utils.timer import Timer
from utils.gpu_monitor import GPUMonitor


def _flatten_search_space(d: dict, prefix: str = '') -> dict:
    """Recursively flatten nested search-space dict to ``{dotted_path: spec}``."""
    result = {}
    for k, v in d.items():
        key = f"{prefix}.{k}" if prefix else str(k)
        if isinstance(v, dict) and 'type' not in v:
            result.update(_flatten_search_space(v, key))
        else:
            result[key] = v
    return result


def training(cfg: AppConfig, trial: optuna.Trial | None = None) -> float:
    fp = FileProcessing(cfg, trial=trial)
    fp.pre_make()
    plot_dir, model_dir, ckpt_dir = fp.plot_dir, fp.model_dir, fp.ckpt_dir
    log_file = fp.log_file
    training_logger = fp.training_logger
    gpu_logger = fp.gpu_logger
    gpu_monitor = GPUMonitor(gpu_logger)
    gpu_monitor.start()
    error_dict = fp.error_dict

    epoch_num = cfg.training.epoch_num

    dp_timer = Timer()
    dp_timer.start()
    DATA = DataProcessing(cfg)
    dataset = DATA.dataset
    norm_dict = DATA.norm_dict
    train_loader = DATA.train_loader
    val_loader = DATA.val_loader
    test_loader = DATA.test_loader
    dp_timer.end()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = gen_model(cfg, dataset, device)
    # Set normalization parameters for sum conservation constraint (if enabled)
    if hasattr(model.head, 'sum_conservation') and model.head.sum_conservation is not None:
        mean, std = norm_dict['y']
        model.head.sum_conservation.set_norm_params(mean, std)
    optimizer = gen_optimizer(cfg, model)
    scheduler = gen_scheduler(cfg, optimizer)

    fp.basic_info_log(
        dataset, train_loader, val_loader, test_loader, None,
        norm_dict, model, dp_timer
        )

    criteria_set = set(list(cfg.training.criteria_list) + [cfg.training.loss_fn])
    # For vector targets only Cosine is meaningful; decide once outside the loop
    phase_criteria = {'Cosine'} if cfg.data.target_type == 'vector' else criteria_set

    eval_class = partial(
        Evaluation, cfg=cfg, device=device, norm_dict=norm_dict)

    model_saving = SaveModel(norm_dict, cfg, model_dir, ckpt_dir, training_logger.info)
    start_epoch = fp.pre_train(model, optimizer, device, model_saving)

    writer = SummaryWriter(log_dir=fp.tensorboard_dir)

    timer = Timer()
    timer.start()
    for epoch in range(start_epoch, epoch_num+1):
        try:
            lr = scheduler.optimizer.param_groups[0]['lr']
            loss = train(model, train_loader, optimizer, cfg.training.loss_fn, device,
                         cfg.training.accumulation_step, cfg.training.use_amp,
                         cfg.training.grad_clip_norm)
            val_loss = validate(model, val_loader, cfg.training.loss_fn, device)
            error_dict['Overall']['LR'] = round(lr, 7)
            error_dict['Overall']['Loss'] = round(loss, 7)
            for phase, loader in zip(['Train', 'Val', 'Test'], [train_loader, val_loader, test_loader]):
                evaluation = eval_class(loader, model)
                pred, target = evaluation.pred, evaluation.target
                err = Metrics(pred, target)
                for criteria in phase_criteria:
                    errors = torch.cat([
                        getattr(err, criteria)(dim=None).unsqueeze(0),
                        getattr(err, criteria)(dim=0).view(-1)
                        ])
                    for subtask, error in zip(error_dict.keys(), errors):
                        error_dict[subtask][phase][criteria] = round(float(error), 7)

            # Denormalized val metric for model tracking (lower is better)
            loss_fn = cfg.training.loss_fn
            if loss_fn == 'Cosine':
                val_metric = 1 - error_dict['Overall']['Val']['Cosine']
            else:
                val_metric = error_dict['Overall']['Val'][loss_fn]

            if scheduler is not None:
                scheduler.step(val_metric)

            # ── TensorBoard ────────────────────────────────────────────────
            writer.add_scalar('Loss/train', loss, epoch)
            writer.add_scalar('Loss/val', val_loss, epoch)
            writer.add_scalar('LR', lr, epoch)
            for subtask in error_dict:
                for criteria in phase_criteria:
                    for phase in ['Train', 'Val', 'Test']:
                        writer.add_scalar(
                            f'{subtask}/{criteria}/{phase}',
                            error_dict[subtask][phase][criteria],
                            epoch,
                        )
            # ───────────────────────────────────────────────────────────────

            info = json.dumps({'Epoch': epoch} | error_dict)

            model_saving.best_model(model, optimizer, epoch, val_metric)
            model_saving.regular_model(model, optimizer, epoch)
            fp.training_log(epoch, info, model_saving.best_val_loss, model_saving.best_epoch)
            torch.cuda.empty_cache()

            if trial is not None:
                trial.report(val_metric, epoch)
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
        f'{model_dir}/best_model_{cfg.timestamp}.pth',
        cfg, DATA, plot_dir
        )

    return model_saving.best_val_loss


def prediction(cfg: AppConfig) -> None:
    fp = FileProcessing(cfg)
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
    DATA = DataProcessing(cfg)
    dataset = DATA.dataset
    norm_dict = DATA.norm_dict
    train_loader = DATA.train_loader
    val_loader = DATA.val_loader
    test_loader = DATA.test_loader
    pred_loader = DATA.pred_loader
    dp_timer.end()

    loader_dict = {'train': train_loader, 'val': val_loader, 'test': test_loader, 'whole': pred_loader}
    try:
        loader = loader_dict[cfg.data.dataset_range]
    except KeyError:
        loader = pred_loader

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = gen_model(cfg, dataset, device)
    optimizer = gen_optimizer(cfg, model)
    fp.pre_train(model, optimizer, device)
    # Set normalization parameters for sum conservation constraint (if enabled).
    # For checkpoints trained with the constraint, buffers are loaded automatically;
    # this ensures they are correct when using a fresh norm_dict.
    if hasattr(model.head, 'sum_conservation') and model.head.sum_conservation is not None:
        mean, std = norm_dict['y']
        model.head.sum_conservation.set_norm_params(mean, std)

    len_target = len(cfg.data.target_list)
    fp.basic_info_log(
        dataset, None, None, None, loader,
        norm_dict, model, dp_timer
        )

    criteria_list = list(cfg.training.criteria_list)
    phase_criteria = {'Cosine'} if cfg.data.target_type == 'vector' else set(criteria_list)

    evaluation = Evaluation(loader, model, cfg, device, norm_dict)
    pred, target = evaluation.pred, evaluation.target
    gpu_monitor.stop()

    err = Metrics(pred, target)
    for criteria in phase_criteria:
        errors = torch.cat([
            getattr(err, criteria)(dim=None).unsqueeze(0),
            getattr(err, criteria)(dim=0).view(-1)
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
    if cfg.data.target_type == 'node':
        batch = torch.cat([data.batch for data in loader])
        torch.save(batch, f'{data_dir}/batch.pt')

    for t, p, task in zip(torch.split(target, 1, dim=-1), torch.split(pred, 1, dim=-1), cfg.data.target_list):
        scatter(
            [t, p],
            scatter_label = ['eval'],
            output_path = f"{plot_dir}/{task}/{cfg.data.dataset_range}.png",
            )


def hparam_tuning(cfg: AppConfig) -> None:
    """Run Optuna hyperparameter search.

    Study settings come from ``cfg.optuna``.
    The search space comes from ``cfg.optuna.search_space`` (nested dict),
    which is flattened to dot-notation keys and applied via OmegaConf.update().
    """
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

    fp = FileProcessing(cfg)
    fp.pre_make()
    storage_name = fp.optuna_db
    redirect_optuna_log(fp.hptuning_logger)

    search_space_raw = OmegaConf.to_container(cfg.optuna.search_space, resolve=True) or {}
    search_space = _flatten_search_space(search_space_raw)

    def objective(trial: optuna.Trial) -> float:
        trial_cfg = cast(AppConfig, deepcopy(cfg))
        for dotted_key, spec in search_space.items():
            OmegaConf.update(trial_cfg, dotted_key, _suggest(trial, dotted_key, spec))
        return training(trial_cfg, trial=trial)

    optuna_cfg = OmegaConf.to_container(cfg.optuna, resolve=True)
    study = create_study(optuna_cfg, f'hptuning_{cfg.output.jobtype}', storage_name)
    study.optimize(objective, n_trials=cfg.optuna.n_trials)
    fp.hptuning_log(study)


@hydra.main(config_path="configs", config_name="config", version_base="1.3")
def main(cfg: DictConfig) -> None:
    """Entry point.  Behaviour is determined by cfg.mode."""
    TIME = time.strftime('%b_%d_%Y_%H%M%S', time.localtime())

    _cfg = cast(AppConfig, cfg)
    OmegaConf.update(_cfg, 'timestamp', TIME)

    setup_seed(_cfg.seed, _cfg.use_deterministic)
    if _cfg.use_deterministic:
        torch.use_deterministic_algorithms(True)

    if torch.cuda.is_available() and _cfg.GPU_memo_frac < 1.0:
        torch.cuda.set_per_process_memory_fraction(_cfg.GPU_memo_frac)

    if _cfg.mode in ['training', 'fine-tuning']:
        training(_cfg)
    elif _cfg.mode == 'hparam_tuning':
        hparam_tuning(_cfg)
    elif _cfg.mode == 'prediction':
        prediction(_cfg)
    else:
        raise ValueError(
            f"Invalid mode '{_cfg.mode}'. Valid: training / hparam_tuning / "
            "prediction / fine-tuning"
        )

if __name__ == '__main__':
    main()
