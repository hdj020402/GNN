import json, os, shutil, pickle
from typing import Callable, List, Optional, Union, Dict, Literal

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data, InMemoryDataset

from rdkit import Chem, RDLogger
from rdkit.Chem.rdchem import BondType as BT
RDLogger.DisableLog('rdApp.*')

from datasets.attr_generator import get_adj_mat, get_edge_attr, get_node_attr
from datasets.utils import read_attr

# definition atom and bond type for one hot repr
_BOND_TYPE = {BT.SINGLE: 0, BT.DOUBLE: 1, BT.TRIPLE: 2, BT.AROMATIC: 3}


class Graph(InMemoryDataset):
    def __init__(self, root: str, transform: Optional[Callable]=None,
                 pre_transform: Optional[Callable]=None,
                 pre_filter: Optional[Callable]=None,
                 sdf_file: str=None,
                 node_attr_file: str=None,
                 edge_attr_file: str=None,
                 graph_attr_file: str=None,
                 vector_file: str=None,
                 weight_file: str=None,
                 atom_type: List[str]=None,
                 default_node_attr: Dict=None,
                 default_edge_attr: Dict=None,
                 node_attr_list: List[str]=[],
                 edge_attr_list: List[str]=[],
                 graph_attr_list: List[str]=[],
                 target_type: Literal['graph', 'node', 'edge']='graph',
                 target_list: List[str]=[],
                 node_attr_filter: Union[int, List]=[],
                 edge_attr_filter: Union[int, List]=[],
                 pos: bool=True,
                 reprocess: bool=False,
                 ):
        '''graph dataset for polymers

        Args:
            root: base dir to store raw and processed data,
            transform: data transformer applied when generating graph(default to None),
            pre_transform: inherited from InMemoryDataset, not used(default to None),
            pre_filter: inherited from InMemoryDataset, not used(default to None),
            sdf_file: path to sdf file containing all molecules 3D structures (MUST HAVE),
            node_attr_file: path to json file containing all extra node attributes,
            edge_attr_file: path to json file containing all extra edge attributes,
            graph_attr_file: path to csv file containing all target(s) and graph_attrs (MUST HAVE),
            node_attr_list: list containing node attr(s) demanding consideration,
            edge_attr_list: list containing edge attr(s) demanding consideration,
            graph_attr_list: list containing graph attr(s) demanding consideration,
            target_list: list containing the target(s),
            node_attr_filter: index of node attr(s) to be filtered(default to []),
            edge_attr_filter: index of edge attr(s) to be filtered(default to []),
            reprocess: whether to force reprocess the dataset.

        Return:
            A InMemoryDataset of polymer graph data.
        '''
        self.root = root
        self.sdf_file = sdf_file
        self.node_attr_file = node_attr_file
        self.edge_attr_file = edge_attr_file
        self.graph_attr_file = graph_attr_file
        self.vector_file  = vector_file
        self.weight_file = weight_file
        self.atom_type = atom_type
        self.default_node_attr = default_node_attr
        self.default_edge_attr = default_edge_attr
        self.node_attr_list = node_attr_list
        self.edge_attr_list = edge_attr_list
        self.graph_attr_list = graph_attr_list
        self.target_type = target_type
        self.target_list = target_list
        self.node_attr_filter = [node_attr_filter] if isinstance(node_attr_filter, int) else node_attr_filter  # TODO: Implement this
        self.edge_attr_filter = [edge_attr_filter] if isinstance(edge_attr_filter, int) else edge_attr_filter  # TODO: Implement this
        self.pos = pos
        self.reprocess = reprocess
        if self.reprocess:
            self._reprocess()
        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0], weights_only=False)

    def mean(self, target: int) -> float:
        '''calculate mean value of all targets.
        '''
        y = torch.cat([self.get(i).y for i in range(len(self))], dim=0)
        return float(y[:, target].mean())

    def std(self, target: int) -> float:
        '''calculate standard deviation value of all targets.
        '''
        y = torch.cat([self.get(i).y for i in range(len(self))], dim=0)
        return float(y[:, target].std())

    @property
    def raw_file_names(self) -> List[str]:
        '''define default path to input data files.
        '''
        return ['merged_mol.sdf', 'smiles.csv']

    @property
    def processed_file_names(self) -> str:
        '''define default path to processed data file.
        '''
        return 'graph_data.pt'

    @property
    def num_graph_features(self) -> int:
        '''number of graph attributes.
        '''
        return len(self.graph_attr_list)

    @property
    def num_total_features(self) -> int:
        '''number of all(node, edge, graph) attributes.
        '''
        return (self.num_node_features +
                self.num_edge_features +
                self.num_graph_features)

    def _reprocess(self):
        if os.path.exists(os.path.join(self.root, 'processed/')):
            shutil.rmtree(os.path.join(self.root, 'processed/'))

    def process(self):
        '''process raw data to generate dataset
        '''
        # read graph_attr_file
        database = pd.read_csv(self.graph_attr_file)

        # read sdf file
        suppl = Chem.SDMolSupplier(self.sdf_file,
                                   removeHs=False,
                                   sanitize=False,
                                   )

        # read node & edge attribute
        node_attr_dict = read_attr(self.node_attr_file, 'node', len(suppl))
        edge_attr_dict = read_attr(self.edge_attr_file, 'edge', len(suppl))
        vector_dict = read_attr(self.vector_file, 'vector', len(suppl))

        # extract target and graph_attr from csv
        if self.target_type == 'graph':
            target = torch.tensor(
                np.array(database.loc[:, self.target_list]),
                dtype=torch.float
                ).reshape(-1, 1, len(self.target_list)).unsqueeze(1)
        elif self.target_type == 'vector':
            target = torch.tensor(
                np.array([vector_dict[key] for key in self.target_list]),
                dtype=torch.float
            ).reshape(len(suppl), -1, len(self.target_list)).unsqueeze(1)
        elif self.target_type == 'node':
            target: List[torch.Tensor] = []
            node_target_list = [node_attr_dict[key] for key in self.target_list]
            for i in zip(*node_target_list):
                target.append(
                    torch.tensor(
                        np.dstack(i), dtype=torch.float
                        ).reshape(-1, len(self.target_list))
                    )
        elif self.target_type == 'edge':
            target: List[torch.Tensor] = []
            edge_target_list = [edge_attr_dict[key] for key in self.target_list]
            for i in zip(*edge_target_list):
                target.append(
                    torch.tensor(
                        np.dstack(i), dtype=torch.float
                        ).reshape(-1, len(self.target_list))
                    )
        if self.graph_attr_list:
            graph_attr = torch.tensor(
                np.array(database.loc[:, self.graph_attr_list]),
                dtype=torch.float
                ).reshape(-1, len(self.graph_attr_list)).unsqueeze(1)
        else:
            graph_attr = torch.empty(len(target), 1, 0)

        if self.weight_file is None:
            if self.target_type == 'graph':
                weights = torch.ones_like(target)
            else:
                weights = [torch.ones_like(t) for t in target]
        else:
            with open(self.weight_file) as wf:
                weights = json.load(wf)

        # extract raw attrs from structures and generate graph
        data_list = []
        for i, mol in enumerate(suppl):
            # node attr
            _node_attr_dict = {k: v[i] for k, v in node_attr_dict.items()} if node_attr_dict else None
            x, pos = get_node_attr(mol,
                                   self.default_node_attr,
                                   self.atom_type,
                                   _node_attr_dict,
                                   self.node_attr_list,
                                   self.node_attr_filter,
                                   )
            if not self.pos:
                pos = None

            # edge attr
            _edge_attr_dict = {k: v[i] for k, v in edge_attr_dict.items()} if edge_attr_dict else None
            edge_attr = get_edge_attr(mol,
                                      self.default_edge_attr,
                                      _BOND_TYPE,
                                      _edge_attr_dict,
                                      self.edge_attr_list,
                                      self.edge_attr_filter,
                                      )
            # edge index(adj matrix)
            edge_index = get_adj_mat(mol)
            # target
            y = target[i]
            # weight
            weight = weights[i]
            # graph name
            name = mol.GetProp('_Name')
            # graph attr
            g_a = graph_attr[i]
            # create mol graph
            data = Data(x=x, pos=pos, edge_index=edge_index,
                        edge_attr=edge_attr, y=y, graph_attr=g_a, weight = weight,
                        name=name, idx=i)
            # pre filter and pre transform
            if self.pre_filter is not None and not self.pre_filter(data):
                continue
            if self.pre_transform is not None:
                data = self.pre_transform(data)
            # graph generated, add to data list
            data_list.append(data)

        torch.save(self.collate(data_list), self.processed_paths[0])
