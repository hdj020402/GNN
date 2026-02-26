import torch
import torch.nn.functional as F
import numpy as np
import os, yaml

from torch.utils.data import random_split, Subset
from torch_geometric.loader import DataLoader
from typing import Dict
from data.graph_dataset import Graph


# Keys compared between current and cached params to decide if reprocessing is needed.
_REPROCESS_KEYS = [
    'sdf_file', 'node_attr_file', 'edge_attr_file', 'graph_attr_file',
    'weight_file', 'atom_type', 'default_node_attr', 'default_edge_attr',
    'node_attr_list', 'edge_attr_list', 'graph_attr_list',
    'node_attr_filter', 'edge_attr_filter', 'pos', 'target_list', 'target_transform',
]


class DataProcessing():
    def __init__(self, param: Dict) -> None:
        self.param = param
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
            cache = os.path.join(self.param['path'], 'processed/model_parameters.yml')
            with open(cache, 'r', encoding='utf-8') as f:
                param_pre: dict = yaml.full_load(f)
            current  = {k: self.param[k] for k in _REPROCESS_KEYS if k in self.param}
            previous = {k: param_pre[k] for k in _REPROCESS_KEYS if k in param_pre}
            return current != previous
        except Exception:
            return True

    def gen_dataset(self) -> Graph:
        dataset = Graph(
            root = self.param['path'],
            transform = None,
            sdf_file = self.param['sdf_file'],
            node_attr_file = self.param['node_attr_file'],
            edge_attr_file = self.param['edge_attr_file'],
            graph_attr_file = self.param['graph_attr_file'],
            vector_file=self.param['vector_file'],
            weight_file = self.param['weight_file'],
            atom_type = self.param['atom_type'],
            default_node_attr = self.param['default_node_attr'],
            default_edge_attr = self.param['default_edge_attr'],
            dist_thresh = self.param['default_edge_attr']['bond_length']['threshold'],
            power_list = self.param['default_edge_attr']['bond_length']['power'],
            node_attr_list = self.param['node_attr_list'],
            edge_attr_list = self.param['edge_attr_list'],
            graph_attr_list = self.param['graph_attr_list'],
            target_type = self.param['target_type'],
            target_list = self.param['target_list'],
            node_attr_filter = self.param['node_attr_filter'],
            edge_attr_filter = self.param['edge_attr_filter'],
            pos = self.param['pos'],
            reprocess = self.reprocess,
            )

        dataset = self._target_transform(dataset)

        with open(os.path.join(self.param['path'], f'processed/model_parameters.yml'), 'w', encoding = 'utf-8') as mp:
            yaml.dump(self.param, mp, allow_unicode = True, sort_keys = False)
        return dataset

    def _target_transform(self, dataset: Graph) -> Graph:
        if self.param['target_transform'] == 'LN':
            dataset.data.y = torch.log(dataset.y)
        elif self.param['target_transform'] == 'LG':
            dataset.data.y = torch.log10(dataset.y)
        elif self.param['target_transform'] == 'E^-x':
            dataset.data.y = torch.exp(-dataset.y)
        elif not self.param['target_transform']:
            pass

        return dataset

    def split_dataset(self) -> tuple[Subset, Subset, Subset, Graph]:
        train_dataset = None
        val_dataset = None
        test_dataset = None
        pred_dataset = None

        pred_dataset = self.dataset

        if self.param['split_method'] == 'random':
            train_size = int(self.param['train_size'] * len(self.dataset))
            val_size = int(self.param['val_size'] * len(self.dataset))
            test_size = len(self.dataset) - train_size - val_size

            train_dataset, val_dataset, test_dataset = random_split(
                self.dataset,
                [train_size, val_size, test_size],
                generator = torch.Generator().manual_seed(self.param['seed'])
                )

        elif self.param['split_method'] == 'manual':
            indices = np.load(self.param['split_file'], allow_pickle=True)
            train_dataset = Subset(self.dataset, indices[0])
            val_dataset = Subset(self.dataset, indices[1])
            test_dataset = Subset(self.dataset, indices[2])

        else:
            raise NotImplementedError("Split method not implemented.")

        return train_dataset, val_dataset, test_dataset, pred_dataset

    def gen_loader(self) -> tuple[DataLoader, DataLoader, DataLoader, DataLoader]:
        train_loader = DataLoader(
            self.train_dataset,
            batch_size = self.param['batch_size'],
            num_workers = self.param['num_workers'],
            shuffle = True,
            pin_memory=True
            )
        val_loader = DataLoader(
            self.val_dataset,
            batch_size = self.param['batch_size'],
            num_workers = self.param['num_workers'],
            shuffle = False,
            pin_memory=True
            )
        test_loader = DataLoader(
            self.test_dataset,
            batch_size = self.param['batch_size'],
            num_workers = self.param['num_workers'],
            shuffle = False,
            pin_memory=True
            )
        pred_loader = DataLoader(
            self.pred_dataset,
            batch_size = self.param['batch_size'],
            num_workers = self.param['num_workers'],
            shuffle = False,
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
        if self.param['target_type'] == 'vector':
            self.dataset.data.y = F.normalize(self.dataset.y, p=2, dim=1)
        else:
            mean, std = self.norm_dict['y']
            self.dataset.data.y = (self.dataset.y - mean) / std

    def get_mean_std(self) -> dict[str, tuple[torch.Tensor, torch.Tensor]]:
        if self.param['mode'] == 'prediction':
            pretrained_model = self.param['pretrained_model']
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
