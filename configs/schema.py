"""Type schema for Hydra cfg — IDE autocompletion via cast(), zero runtime overhead.

Usage in main.py::

    from typing import cast
    from configs.schema import AppConfig

    @hydra.main(...)
    def main(cfg: DictConfig) -> None:
        _cfg = cast(AppConfig, cfg)   # narrows type for IDE; no runtime effect
        training(_cfg)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class SchedulerConfig:
    type: str = "ReduceLROnPlateau"
    factor: float = 0.7
    patience: int = 20
    min_lr: float = 0.00001


@dataclass
class EarlyStoppingConfig:
    patience: int = 50
    delta: float = 0.0


@dataclass
class TrainingConfig:
    loss_fn: str = "MSE"
    optimizer: str = "Adam"
    lr: float = 0.001
    scheduler: SchedulerConfig = field(default_factory=SchedulerConfig)
    accumulation_step: int = 1
    epoch_num: int = 200
    output_step: int = 1
    model_save_step: int = 5
    early_stopping: EarlyStoppingConfig = field(default_factory=EarlyStoppingConfig)
    criteria_list: list[str] = field(default_factory=list)
    use_amp: bool = False


@dataclass
class BondLengthConfig:
    power: list[str] = field(default_factory=lambda: ['r^-1'])
    threshold: float | None = None


@dataclass
class DefaultNodeAttrConfig:
    ele_type: bool = True
    atomic_number: bool = True
    aromatic: bool = False
    num_neighbors: bool = True
    num_hs: bool = True
    hybridization: bool = False
    in_ring: bool = False
    formal_charge: bool = False


@dataclass
class DefaultEdgeAttrConfig:
    edge_type: bool = True
    conjugated: bool = False
    bond_in_ring: bool = False
    bond_length: BondLengthConfig = field(default_factory=BondLengthConfig)


@dataclass
class DataConfig:
    path: str = "data"
    sdf_file: str = "data/all.sdf"
    node_attr_file: str | None = None
    edge_attr_file: str | None = None
    graph_attr_file: str | None = None
    vector_file: str | None = None
    weight_file: str | None = None
    atom_type: list[str] = field(default_factory=lambda: ['H', 'C', 'N', 'O', 'F', 'UNK'])
    default_node_attr: DefaultNodeAttrConfig = field(default_factory=DefaultNodeAttrConfig)
    default_edge_attr: DefaultEdgeAttrConfig = field(default_factory=DefaultEdgeAttrConfig)
    node_attr_list: list[str] = field(default_factory=list)
    edge_attr_list: list[str] = field(default_factory=list)
    graph_attr_list: list[str] = field(default_factory=list)
    node_attr_filter: list[str] = field(default_factory=list)
    edge_attr_filter: list[str] = field(default_factory=list)
    pos: bool = True
    sanitize: bool = True
    graph_type: str = "bond"
    target_type: str = "graph"
    target_list: list[str] = field(default_factory=lambda: ['target1'])
    target_transform: str | None = None
    batch_size: int = 32
    num_workers: int = 4
    split_method: str = "random"
    split_file: str | None = None
    train_size: float = 0.6
    val_size: float = 0.2
    dataset_range: str = "whole"


@dataclass
class OutputConfig:
    jobtype: str = "experiment"


@dataclass
class ModelBackboneConfig:
    name: str = "mpnn"
    # Additional backbone kwargs (node_dim, mp_times, …) come from the YAML config group.


@dataclass
class ModelHeadConfig:
    name: str = "graph"
    # Additional head kwargs come from the YAML config group.


@dataclass
class ModelConfig:
    backbone: ModelBackboneConfig = field(default_factory=ModelBackboneConfig)
    head: ModelHeadConfig = field(default_factory=ModelHeadConfig)


@dataclass
class AppConfig:
    """Top-level type schema for Hydra cfg.

    Use ``cast(AppConfig, cfg)`` in ``main()`` to narrow the DictConfig to
    this typed schema for IDE autocompletion.  No runtime validation is
    performed; attribute access still goes through OmegaConf's DictConfig.
    """
    mode: str = "training"
    seed: int = 42
    use_deterministic: bool = True
    GPU_memo_frac: float = 0.5
    pretrained_model: str | None = None
    # Set at runtime in main() via OmegaConf.update(cfg, 'timestamp', TIME)
    timestamp: str = ""
    data: DataConfig = field(default_factory=DataConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    # optuna has a complex nested structure; accessed via OmegaConf.to_container()
    optuna: Any = None
