import json, os, shutil, pickle
from typing import Callable, Literal

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data, InMemoryDataset

from rdkit import Chem, RDLogger
RDLogger.DisableLog('rdApp.*')

from data.featurizer import get_adj_mat, get_edge_attr, get_node_attr
from data.graph_transforms import complete_with_dist_filter, power_dist


def _read_attr(file: str | None, attr_type: str, length: int) -> dict | None:
    '''Load a per-molecule attribute file (.json or .pkl) and validate its length.

    Returns None when file is None; raises ValueError for unsupported formats
    or mismatched lengths.
    '''
    if file is None:
        return None
    _, ext = os.path.splitext(file)
    if ext == '.json':
        with open(file, 'r') as f:
            attr_dict: dict = json.load(f)
    elif ext == '.pkl':
        with open(file, 'rb') as f:
            attr_dict: dict = pickle.load(f)
    else:
        raise ValueError(f"Unsupported attribute file format: {ext}")
    lengths = [len(v) for v in attr_dict.values()]
    if min(lengths) != max(lengths):
        raise ValueError(f'{attr_type} attributes have inconsistent lengths.')
    if min(lengths) != length:
        raise ValueError(
            f'{attr_type} attribute file has {min(lengths)} entries '
            f'but SDF has {length}.'
        )
    return attr_dict


class Graph(InMemoryDataset):
    def __init__(self, root: str, transform: Callable | None=None,
                 pre_transform: Callable | None=None,
                 pre_filter: Callable | None=None,
                 sdf_file: str=None,
                 node_attr_file: str=None,
                 edge_attr_file: str=None,
                 graph_attr_file: str=None,
                 vector_file: str=None,
                 weight_file: str=None,
                 atom_type: list[str]=None,
                 default_node_attr: dict=None,
                 default_edge_attr: dict=None,
                 dist_thresh: float | None=None,
                 power_list: list | None=None,
                 node_attr_list: list[str]=[],
                 edge_attr_list: list[str]=[],
                 graph_attr_list: list[str]=[],
                 target_type: Literal['graph', 'node', 'edge']='graph',
                 target_list: list[str]=[],
                 node_attr_filter: int | list=[],
                 edge_attr_filter: int | list=[],
                 pos: bool=True,
                 graph_type: str='bond',
                 sanitize: bool=True,
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
        self.dist_thresh = dist_thresh
        self.power_list = power_list
        self.node_attr_list = node_attr_list
        self.edge_attr_list = edge_attr_list
        self.graph_attr_list = graph_attr_list
        self.target_type = target_type
        self.target_list = target_list
        self.node_attr_filter = [node_attr_filter] if isinstance(node_attr_filter, int) else node_attr_filter  # TODO: Implement this
        self.edge_attr_filter = [edge_attr_filter] if isinstance(edge_attr_filter, int) else edge_attr_filter  # TODO: Implement this
        self.pos = pos
        self.graph_type = graph_type
        self.sanitize = sanitize
        self.reprocess = reprocess
        if self.reprocess:
            self._reprocess()
        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0], weights_only=False)

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
        if self.graph_attr_file is None:
            database = None
        else:
            database = pd.read_csv(self.graph_attr_file)

        # read sdf file
        suppl = Chem.SDMolSupplier(self.sdf_file,
                                   removeHs=False,
                                   sanitize=self.sanitize,
                                   )

        # read node & edge attribute
        node_attr_dict = _read_attr(self.node_attr_file, 'node', len(suppl))
        edge_attr_dict = _read_attr(self.edge_attr_file, 'edge', len(suppl))
        vector_dict = _read_attr(self.vector_file, 'vector', len(suppl))

        # extract target and graph_attr from csv
        try:
            if self.target_type == 'graph':
                target = torch.tensor(
                    np.array(database.reindex(columns=self.target_list)),
                    dtype=torch.float
                    ).reshape(-1, len(self.target_list))
            elif self.target_type == 'vector':
                target = torch.tensor(
                    np.array([vector_dict[key] for key in self.target_list]),
                    dtype=torch.float
                ).reshape(len(suppl), -1)
            elif self.target_type == 'node':
                target: list[torch.Tensor] = []
                node_target_list = [node_attr_dict[key] for key in self.target_list]
                for i in zip(*node_target_list):
                    target.append(
                        torch.tensor(
                            np.dstack(i), dtype=torch.float
                            ).reshape(-1, len(self.target_list))
                        )
            elif self.target_type == 'edge':
                target: list[torch.Tensor] = []
                edge_target_list = [edge_attr_dict[key] for key in self.target_list]
                for i in zip(*edge_target_list):
                    target.append(
                        torch.tensor(
                            np.dstack(i), dtype=torch.float
                            ).reshape(-1, len(self.target_list))
                        )
        except TypeError:
            target = torch.empty(len(suppl), 0)

        if self.graph_attr_list:
            graph_attr = torch.tensor(
                np.array(database.loc[:, self.graph_attr_list]),
                dtype=torch.float
                ).reshape(-1, len(self.graph_attr_list))
        else:
            graph_attr = torch.empty(len(target), 0)

        if self.weight_file is None:
            if self.target_type in ['graph', 'vector']:
                weights = torch.ones([len(suppl), len(self.target_list)])
            else:
                weights = [torch.ones_like(t) for t in target]
        else:
            with open(self.weight_file) as wf:
                weights = json.load(wf)

        # extract raw attrs from structures and generate graph
        data_list = []
        for i, mol in enumerate(suppl):
            if mol is None:
                continue
            if not self.sanitize:
                # Partial sanitization: skip only the valence-property check so that
                # molecules with unusual valences (e.g. hypervalent P/S/Cl) are kept,
                # while hybridization, aromaticity, ring info, and conjugation are
                # all correctly assigned.
                try:
                    Chem.SanitizeMol(mol, Chem.SanitizeFlags.SANITIZE_ALL ^ Chem.SanitizeFlags.SANITIZE_PROPERTIES)
                except Exception:
                    # Rare: e.g. truly un-kekulizable aromatic systems.
                    # Treat the same as a failed strict sanitization — skip the entry.
                    continue

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
                                      _edge_attr_dict,
                                      self.edge_attr_list,
                                      self.edge_attr_filter,
                                      )
            # edge index(adj matrix)
            edge_index = get_adj_mat(mol)
            # atomic numbers as long tensor (required by equivariant backbones)
            z = torch.tensor(
                [atom.GetAtomicNum() for atom in mol.GetAtoms()],
                dtype=torch.long
            )
            # target
            y = target[i]
            # weight
            weight = weights[i]
            # graph name
            name = mol.GetProp('_Name')
            # graph attr
            g_a = graph_attr[i]
            # create mol graph
            data = Data(x=x, z=z, pos=pos, edge_index=edge_index,
                        edge_attr=edge_attr, y=y, graph_attr=g_a, weight=weight,
                        name=name, idx=i)

            # Dispatch based on graph_type
            if self.graph_type == 'bond':
                # Keep only chemical bonds, optionally add distance features
                if self.pos and self.power_list:
                    for length_type in self.power_list:
                        power = float(length_type.split('^')[1])
                        data = power_dist(data, cat=True, power=power)

            elif self.graph_type in ('radius', 'complete'):
                # Build extended graphs (distance-based or complete)
                if not self.pos:
                    raise ValueError(
                        f"graph_type='{self.graph_type}' requires pos=true."
                    )
                data = complete_with_dist_filter(data, self.dist_thresh)
                for length_type in self.power_list:
                    power = float(length_type.split('^')[1])
                    data = power_dist(data, cat=True, power=power)

            else:
                raise ValueError(f"Unknown graph_type: '{self.graph_type}'")

            # pre filter and pre transform
            if self.pre_filter is not None and not self.pre_filter(data):
                continue
            if self.pre_transform is not None:
                data = self.pre_transform(data)
            # graph generated, add to data list
            data_list.append(data)

        torch.save(self.collate(data_list), self.processed_paths[0])
