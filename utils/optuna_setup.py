"""Optuna study factory.

Reads study settings from the ``optuna`` section of the Hydra config
(cfg.optuna / param['optuna']).  The search-space is a separate dict
(loaded from configs/optuna/search_space.yaml) that maps dot-notation
param keys to suggest-method specs.
"""
import logging
import optuna


def create_pruner(optuna_cfg: dict) -> optuna.pruners.BasePruner:
    pruner_cfg = optuna_cfg['pruner']
    cls = getattr(optuna.pruners, pruner_cfg['type'])
    kwargs = {k: v for k, v in pruner_cfg.items() if k != 'type'}
    return cls(**kwargs)


def create_sampler(optuna_cfg: dict) -> optuna.samplers.BaseSampler:
    sampler_cfg = optuna_cfg['sampler']
    cls = getattr(optuna.samplers, sampler_cfg['type'])
    kwargs = {k: v for k, v in sampler_cfg.items() if k != 'type'}
    return cls(**kwargs)


def create_study(optuna_cfg: dict, study_name: str, storage: str) -> optuna.Study:
    """Create or resume an Optuna study from the config dict."""
    cont = optuna_cfg.get('continue_study', {})
    if cont.get('enabled', False):
        resume_storage = cont.get('storage') or storage
        resume_name = cont.get('study_name')
        if resume_name is None:
            summaries = optuna.get_all_study_summaries(storage=resume_storage)
            resume_name = summaries[0].study_name
        return optuna.load_study(study_name=resume_name, storage=resume_storage)

    return optuna.create_study(
        sampler=create_sampler(optuna_cfg),
        pruner=create_pruner(optuna_cfg),
        direction=optuna_cfg['direction'],
        study_name=study_name,
        storage=storage,
        load_if_exists=True,
    )


def redirect_optuna_log(logger: logging.Logger) -> None:
    """Route Optuna's own logger into the job's training logger."""
    optuna_logger = logging.getLogger('optuna')
    optuna_logger.handlers = []
    optuna_logger.addHandler(logger.handlers[0])
    optuna_logger.setLevel(logging.INFO)
    optuna_logger.propagate = False
