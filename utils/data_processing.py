import torch
import torch_geometric.transforms as T
import numpy as np
import os, yaml

from torch.utils.data import random_split, Subset
from torch_geometric.loader import DataLoader
from typing import Dict
from datasets.graph_dataset import Graph
from datasets.datasets import CustomSubset
from datasets.transform import Complete, CompleteWithDistanceFilter, PowerDistance

class data_processing():
    def __init__(self, param: Dict, reprocess: bool = True) -> None:
        self.param = param
        self.path = param['path']
        self.sdf_file = param['sdf_file']
        self.node_attr_file = param['node_attr_file']
        self.edge_attr_file = param['edge_attr_file']
        self.graph_attr_file = param['graph_attr_file']
        self.weight_file = param['weight_file']
        self.atom_type = param['atom_type']
        self.default_node_attr = param['default_node_attr']
        self.default_edge_attr = param['default_edge_attr']
        self.transform = self.Transform()
        self.node_attr_list = param['node_attr_list']
        self.edge_attr_list = param['edge_attr_list']
        self.graph_attr_list = param['graph_attr_list']
        self.target_type = param['target_type']
        self.target_list = param['target_list']
        self.target_transform = param['target_transform']
        self.node_attr_filter = param['node_attr_filter']
        self.edge_attr_filter = param['edge_attr_filter']
        self.pos = param['pos']
        self.seed = param['seed']
        self.split_method = param['split_method']
        self.split_path = param['SPLIT_file']
        self.train_size = param['train_size']
        self.val_size = param['val_size']
        self.batch_size = param['batch_size']
        self.num_workers = param['num_workers']
        self.reprocess = reprocess
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.dataset = self.gen_dataset()
        self.train_dataset, self.val_dataset, self.test_dataset, self.pred_dataset = self.split_dataset()
        self.mean, self.std = self.get_mean_std()
        self.normalization()
        self.train_dataset, self.val_dataset, self.test_dataset, self.pred_dataset = self.split_dataset()
        self.train_loader, self.val_loader, self.test_loader, self.pred_loader = self.gen_loader()

    def Transform(self) -> T.Compose:
        transforms = [CompleteWithDistanceFilter(distance_threshold=self.default_edge_attr['bond_length']['threshold'])]
        for length_type in self.default_edge_attr['bond_length']['power']:
            power = float(length_type.split('^')[1])
            transforms.append(PowerDistance(
                norm=False, power=power, distance_threshold=self.default_edge_attr['bond_length']['threshold']))
        transform = T.Compose(transforms)
        return transform

    def gen_dataset(self) -> Graph:
        dataset = Graph(
            root = self.path,
            transform = self.transform,
            sdf_file = self.sdf_file,
            node_attr_file = self.node_attr_file,
            edge_attr_file = self.edge_attr_file,
            graph_attr_file = self.graph_attr_file,
            weight_file = self.weight_file,
            atom_type = self.atom_type,
            default_node_attr = self.default_node_attr,
            default_edge_attr = self.default_edge_attr,
            node_attr_list = self.node_attr_list,
            edge_attr_list = self.edge_attr_list,
            graph_attr_list = self.graph_attr_list,
            target_type = self.target_type,
            target_list = self.target_list,
            node_attr_filter = self.node_attr_filter,
            edge_attr_filter = self.edge_attr_filter,
            pos = self.pos,
            reprocess = self.reprocess,
            )

        dataset = self._target_transform(dataset)

        with open(os.path.join(self.path, f'processed/model_parameters.yml'), 'w', encoding = 'utf-8') as mp:
            yaml.dump(self.param, mp, allow_unicode = True, sort_keys = False)
        return dataset

    def _target_transform(self, dataset):
        if self.target_transform == 'LN':
            dataset.data.y = torch.log(dataset.y)
        elif self.target_transform == 'LG':
            dataset.data.y = torch.log10(dataset.y)
        elif self.target_transform == 'E^-x':
            dataset.data.y = torch.exp(-dataset.y)
        elif not self.target_transform:
            pass

        return dataset

    def split_dataset(self):
        train_dataset = None
        val_dataset = None
        test_dataset = None
        pred_dataset = None


        pred_dataset = self.dataset

        if self.split_method == 'random':
            train_size = int(self.train_size * len(self.dataset))
            val_size = int(self.val_size * len(self.dataset))
            test_size = len(self.dataset) - train_size - val_size

            train_dataset, val_dataset, test_dataset = random_split(
                self.dataset,
                [train_size, val_size, test_size],
                generator = torch.Generator().manual_seed(self.seed)
                )

        elif self.split_method == 'manual':
            indices = np.load(self.split_path, allow_pickle=True)
            train_dataset = Subset(self.dataset, indices[0])
            val_dataset = Subset(self.dataset, indices[1])
            test_dataset = Subset(self.dataset, indices[2])

        else:
            raise NotImplementedError("Split method not implemented.")

        return train_dataset, val_dataset, test_dataset, pred_dataset

    def gen_loader(self):
        train_loader = DataLoader(
            self.train_dataset,
            batch_size = self.batch_size,
            num_workers = self.num_workers,
            shuffle = True,
            pin_memory=True
            )
        val_loader = DataLoader(
            self.val_dataset,
            batch_size = self.batch_size,
            num_workers = self.num_workers,
            shuffle = False,
            pin_memory=True
            )
        test_loader = DataLoader(
            self.test_dataset,
            batch_size = self.batch_size,
            num_workers = self.num_workers,
            shuffle = False,
            pin_memory=True
            )
        pred_loader = DataLoader(
            self.pred_dataset,
            batch_size = self.batch_size,
            num_workers = self.num_workers,
            shuffle = False,
            pin_memory=True
            )

        return train_loader, val_loader, test_loader, pred_loader

    def normalization(self):
        if self.param['graph_attr_list']:
            data_scaled = (torch.cat([self.dataset.graph_attr, self.dataset.y], dim = 1) - self.mean) / self.std
            self.dataset.data.graph_attr = data_scaled[:, 0:len(self.graph_attr_list)]
            self.dataset.data.y = data_scaled[:, len(self.graph_attr_list):len(self.graph_attr_list) + len(self.target_list)]
        else:
            self.dataset.data.y = (self.dataset.y - self.mean) / self.std

    def get_mean_std(self):
        if self.param['mode'] == 'prediction':
            pretrained_model = self.param['pretrained_model']
            state_dict: Dict = torch.load(pretrained_model, map_location=torch.device('cpu'))
            mean = state_dict['mean']
            std = state_dict['std']
        else:
            train_dataset = CustomSubset(self.dataset, self.train_dataset.indices)
            if self.param['graph_attr_list']:
                combined_data = torch.cat(
                    [
                        train_dataset.graph_attr,
                        train_dataset.y
                    ],
                    dim=1
                )
                mean = combined_data.mean(dim=0)
                std = combined_data.std(dim=0, unbiased=False)
            else:
                mean = train_dataset.y.mean(dim=0)
                std = train_dataset.y.std(dim=0, unbiased=False)
        return mean, std