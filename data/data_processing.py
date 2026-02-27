import torch
import torch.nn.functional as F
import numpy as np
import os, yaml

from torch.utils.data import random_split, Subset
from torch_geometric.loader import DataLoader
from typing import Dict

from omegaconf import OmegaConf

from data.graph_dataset import Graph
from configs.schema import AppConfig


# Keys compared between current and cached data config to decide if reprocessing is needed.
_REPROCESS_KEYS = [
    'sdf_file', 'node_attr_file', 'edge_attr_file', 'graph_attr_file',
    'weight_file', 'atom_type', 'default_node_attr', 'default_edge_attr',
    'node_attr_list', 'edge_attr_list', 'graph_attr_list',
    'node_attr_filter', 'edge_attr_filter', 'pos', 'target_list', 'target_transform',
]


class DataProcessing():
    def __init__(self, cfg: AppConfig) -> None:
        self.cfg = cfg
        self.reprocess = self._should_reprocess()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.dataset = self.gen_dataset()
        self.train_dataset, self.val_dataset, self.test_dataset, self.pred_dataset = self.split_dataset()
        self.norm_dict = self.get_mean_std()
        self.normalization()
        self.train_loader, self.val_loader, self.test_loader, self.pred_loader = self.gen_loader()

    def _should_reprocess(self) -> bool:
        """Return True if dataset-affecting params changed since last processing."""
        try:
            cache = os.path.join(self.cfg.data.path, 'processed/model_parameters.yml')
            with open(cache, 'r', encoding='utf-8') as f:
                cached: dict = yaml.full_load(f)
            current = OmegaConf.to_container(self.cfg.data, resolve=True)
            return {k: current.get(k) for k in _REPROCESS_KEYS} != \
                   {k: cached.get(k) for k in _REPROCESS_KEYS}
        except Exception:
            return True

    def gen_dataset(self) -> Graph:
        d = self.cfg.data
        dataset = Graph(
            root=d.path,
            transform=None,
            sdf_file=d.sdf_file,
            node_attr_file=d.node_attr_file,
            edge_attr_file=d.edge_attr_file,
            graph_attr_file=d.graph_attr_file,
            vector_file=d.vector_file,
            weight_file=d.weight_file,
            atom_type=list(d.atom_type),
            default_node_attr=OmegaConf.to_container(d.default_node_attr, resolve=True),
            default_edge_attr=OmegaConf.to_container(d.default_edge_attr, resolve=True),
            dist_thresh=d.default_edge_attr.bond_length.threshold,
            power_list=list(d.default_edge_attr.bond_length.power),
            node_attr_list=list(d.node_attr_list),
            edge_attr_list=list(d.edge_attr_list),
            graph_attr_list=list(d.graph_attr_list),
            target_type=d.target_type,
            target_list=list(d.target_list),
            node_attr_filter=list(d.node_attr_filter),
            edge_attr_filter=list(d.edge_attr_filter),
            pos=d.pos,
            reprocess=self.reprocess,
            )

        dataset = self._target_transform(dataset)

        OmegaConf.save(self.cfg.data, os.path.join(d.path, 'processed/model_parameters.yml'))
        return dataset

    def _target_transform(self, dataset: Graph) -> Graph:
        transform = self.cfg.data.target_transform
        if transform == 'LN':
            dataset.data.y = torch.log(dataset.y)
        elif transform == 'LG':
            dataset.data.y = torch.log10(dataset.y)
        elif transform == 'E^-x':
            dataset.data.y = torch.exp(-dataset.y)
        elif not transform:
            pass

        return dataset

    def split_dataset(self) -> tuple[Subset, Subset, Subset, Graph]:
        train_dataset = None
        val_dataset = None
        test_dataset = None
        pred_dataset = None

        pred_dataset = self.dataset

        if self.cfg.data.split_method == 'random':
            train_size = int(self.cfg.data.train_size * len(self.dataset))
            val_size = int(self.cfg.data.val_size * len(self.dataset))
            test_size = len(self.dataset) - train_size - val_size

            train_dataset, val_dataset, test_dataset = random_split(
                self.dataset,
                [train_size, val_size, test_size],
                generator=torch.Generator().manual_seed(self.cfg.seed)
                )

        elif self.cfg.data.split_method == 'manual':
            indices = np.load(self.cfg.data.split_file, allow_pickle=True)
            train_dataset = Subset(self.dataset, indices[0])
            val_dataset = Subset(self.dataset, indices[1])
            test_dataset = Subset(self.dataset, indices[2])

        else:
            raise NotImplementedError("Split method not implemented.")

        return train_dataset, val_dataset, test_dataset, pred_dataset

    def gen_loader(self) -> tuple[DataLoader, DataLoader, DataLoader, DataLoader]:
        batch_size = self.cfg.data.batch_size
        num_workers = self.cfg.data.num_workers
        train_loader = DataLoader(
            self.train_dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            shuffle=True,
            pin_memory=True
            )
        val_loader = DataLoader(
            self.val_dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            shuffle=False,
            pin_memory=True
            )
        test_loader = DataLoader(
            self.test_dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            shuffle=False,
            pin_memory=True
            )
        pred_loader = DataLoader(
            self.pred_dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            shuffle=False,
            pin_memory=True
            )

        return train_loader, val_loader, test_loader, pred_loader

    def normalization(self) -> None:
        for attr, norm_info in self.norm_dict.items():
            if attr == 'y':
                continue
            mean, std = norm_info
            data = getattr(self.dataset.data, attr)
            scaled_attr = torch.zeros_like(data)
            for i in range(data.shape[-1]):
                if std[i] == 0:
                    scaled_attr[:, i] = 0.0
                else:
                    scaled_attr[:, i] = (data[:, i] - mean[i]) / std[i]
            setattr(self.dataset.data, attr, scaled_attr)
        if self.cfg.data.target_type == 'vector':
            self.dataset.data.y = F.normalize(self.dataset.y, p=2, dim=1)
        else:
            mean, std = self.norm_dict['y']
            self.dataset.data.y = (self.dataset.y - mean) / std

    def get_mean_std(self) -> dict[str, tuple[torch.Tensor, torch.Tensor]]:
        if self.cfg.mode == 'prediction':
            pretrained_model = self.cfg.pretrained_model
            state_dict: Dict = torch.load(pretrained_model, map_location=torch.device('cpu'))
            norm_dict = state_dict['norm']
        else:
            train_dataset = self.dataset[self.train_dataset.indices]
            norm_dict = {}
            for attr in ['x', 'edge_attr', 'y', 'graph_attr', ]:
                data: torch.Tensor = getattr(train_dataset, attr)
                try:
                    data = data.view(-1, data.shape[-1])
                    mean = data.mean(dim=0)
                    std = data.std(dim=0)
                    for i in range(data.shape[-1]):
                        if ((data[:, i] == 0) | (data[:, i] == 1)).all():
                            mean[i] = 0.0
                            std[i] = 1.0
                    norm_dict[attr] = (mean, std)
                except RuntimeError:
                    pass
        return norm_dict
