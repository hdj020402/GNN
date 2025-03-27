import optuna
import logging
from typing import Dict

class OptunaSetup:
    def __init__(self, param: Dict, ht_param: Dict[str, Dict]) -> None:
        self.param = param
        self.ht_param = ht_param

    def create_pruner(self) -> optuna.pruners.BasePruner:
        pruner = getattr(optuna.pruners, self.ht_param['optuna']['pruner']['type'])
        pruner_kwargs = {k: v for k, v in self.ht_param['optuna']['pruner'].items() if k != 'type'}
        return pruner(**pruner_kwargs)

    def create_sampler(self) -> optuna.samplers.BaseSampler:
        sampler = getattr(optuna.samplers, self.ht_param['optuna']['sampler']['type'])
        sampler_kwargs = {k: v for k, v in self.ht_param['optuna']['sampler'].items() if k != 'type'}
        return sampler(**sampler_kwargs)

    def create_study(self, study_name: str, storage: str) -> optuna.Study:
        if self.ht_param['optuna']['continue_trials']['continue'] is False:
            return optuna.create_study(
                sampler=self.create_sampler(),
                pruner=self.create_pruner(),
                direction=self.ht_param['optuna']['direction'],
                study_name=study_name,
                storage=storage,
                load_if_exists=True
                )
        else:
            return self.load_study()

    def logging_setup(self, hptuning_logger: logging.Logger) -> None:
        optuna_logger = logging.getLogger('optuna')
        optuna_logger.handlers = []

        optuna_logger.addHandler(hptuning_logger.handlers[0])
        optuna_logger.setLevel(logging.INFO)

        optuna_logger.propagate = False

    def load_study(self):
        storage = self.ht_param['optuna']['continue_trials']['storage']
        study_name = self.ht_param['optuna']['continue_trials']['study_name']
        if study_name is None:
            study_summaries = optuna.get_all_study_summaries(storage=storage)
            study_name = study_summaries[0].study_name
        return optuna.load_study(study_name=study_name, storage=storage)
