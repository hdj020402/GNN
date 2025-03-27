'''Attribute generator in gnn_attribute.
'''
import numpy as np
import torch
from rdkit import Chem
from rdkit.Chem.rdchem import BondType as BT
from typing import Dict, List

from datasets.utils import one_hot


def get_node_attr(mol,
                  default_node_attr: Dict,
                  atom_type: List,
                  _node_attr_dict: Dict[str, List],
                  node_attr_list: List,
                  filter: List,
                  ) -> tuple[torch.Tensor, torch.Tensor]:
    '''Generator of node attributes.

    Args:
        mol: rdkit mol object, has N atoms.
        node_attr_mat: (multiple) extra attributes need to add to the nodes.
            Maybe in form of np.ndarray (N atoms * M attributes),
            or a list (length=M) of list (length=N).
        atom_type: dict of element symbol and index in one hot repr.
        filter: list of attr index that need to be filtered.

    Return:
        mol_node_attr(np.array, shape=N*(len(_ATOM_TYPE)+4+X)): including
            one hot repr of atom type,
            atomic number,
            whether is aromatic,
            number of atom neighbors,
            number of Hs in neighbors,
            extra attributes (if exists).
        pos(np.array, shape=N*3):
            atom coord.
    '''
    # default
    conf = mol.GetConformer()
    pos = torch.tensor(conf.GetPositions(), dtype=torch.float)

    type_idx_hot = []
    atomic_number = []
    aromatic = []
    num_neighbors = []
    num_hs = []

    for atom in mol.GetAtoms():
        type_idx = atom_type.index(atom.GetSymbol())
        type_idx_hot.append(one_hot(type_idx, len(atom_type)))
        atomic_number.append(atom.GetAtomicNum())
        aromatic.append(1 if atom.GetIsAromatic() else 0)
        neighbors = atom.GetNeighbors()
        num_neighbors.append(len(neighbors))
        nhs = 0
        for neighbor in neighbors:
            if neighbor.GetAtomicNum() == 1:
                nhs += 1
        num_hs.append(nhs)

    default_node_attr_dict = {
        'ele_type': type_idx_hot,
        'atomic_number': atomic_number,
        'aromatic': aromatic,
        'num_neighbors': num_neighbors,
        'num_hs': num_hs
    }

    mol_node_attr_list = []
    for key, value in default_node_attr.items():
        if not value:
            continue
        if key == 'ele_type':
            mol_node_attr_list.append(np.array(default_node_attr_dict[key]))
        else:
            mol_node_attr_list.append(np.array(default_node_attr_dict[key])[:, np.newaxis])

    # additional
    node_attr_mat = []
    for node_attr in node_attr_list:
        node_attr_mat.append(
            np.array(_node_attr_dict[node_attr]).reshape(len(_node_attr_dict[node_attr]), -1)
            )
    try:
        node_attr_mat = np.concatenate(node_attr_mat, axis=1, dtype=float)
    except ValueError:
        node_attr_mat = np.array(node_attr_mat, dtype=float)

    if node_attr_mat.size:
        mol_node_attr_list.append(node_attr_mat)
    mol_node_attr = np.concatenate(mol_node_attr_list, axis=1, dtype=float)

    # add filter
    node_filter = [1] * mol_node_attr.shape[1]
    for nfi in filter:
        node_filter[nfi] = 0
    node_filter = np.array(node_filter, dtype=bool)
    mol_node_attr = mol_node_attr[:, node_filter]

    return torch.tensor(mol_node_attr, dtype=torch.float), pos


def get_adj_mat(mol, tot_atom: int=0):
    '''Generator of adj matrix.

    Args:
        mol: rdkit mol object, has M bonds.
        tot_atom: number of atoms already in this graph.

    Return:
        mol_adj_mat(np.array, shape=2M*2): adj matrix of this mol.
    '''
    start_list = []
    end_list = []
    for bond in mol.GetBonds():
        start_list.append(bond.GetBeginAtomIdx())
        end_list.append(bond.GetEndAtomIdx())

        start_list.append(bond.GetEndAtomIdx())
        end_list.append(bond.GetBeginAtomIdx())

    mol_adj_mat = torch.tensor([start_list,
                                end_list])

    return mol_adj_mat


def get_edge_attr(mol,
                  default_edge_attr: Dict,
                  bond_type: Dict,
                  _edge_attr_dict: Dict[str, List],
                  edge_attr_list: List,
                  filter: List,
                  ):
    '''Generator of edge type.

    Args:
        mol: rdkit mol object, has M bonds.
        bond_type: dict of bond type in rdkit(key) to one hot index(value).
        filter: list of attr index that need to be filtered.
        extra_attr: (multiple) extra attributes need to add to the edges.
            Maybe in form of np.ndarray (N bonds * M attributes),
            or a list (length=M) of list (length=N).

    Return:
        mol_edge_type(torch.tensor, shape=2M*len(_BOND_TYPE)): including
            one hot repr of bond type,
            extra attributes (if exists).
            # remember to double the extra attributes (twice for each bond)
    '''
    # default
    edge_type = []
    for bond in mol.GetBonds():
        edge_type += 2 * [bond_type[bond.GetBondType()]]

    default_node_attr_dict = {
        'edge_type': [one_hot(t, len(bond_type)) for t in edge_type],
    }

    mol_edge_attr_list = []
    for key, value in default_edge_attr.items():
        if not value:
            continue
        if key == 'edge_type':
            mol_edge_attr_list.append(np.array(default_node_attr_dict[key]))
        else:
            ...
    try:
        mol_edge_attr = np.concatenate(mol_edge_attr_list, axis=1, dtype=float)
    except ValueError:
        return None

    # additional
    edge_attr_mat = []
    for edge_attr in edge_attr_list:
        edge_attr_mat.append(
            np.array(_edge_attr_dict[edge_attr]).reshape(len(_edge_attr_dict[edge_attr]), -1)
            )
    try:
        edge_attr_mat = np.concatenate(edge_attr_mat, axis=1, dtype=float)
    except ValueError:
        edge_attr_mat = np.array(edge_attr_mat, dtype=float)

    if edge_attr_mat.size:
        mol_edge_attr_list.append(edge_attr_mat)
    mol_edge_attr = np.concatenate(mol_edge_attr_list, axis=1, dtype=float)

    # add filter
    edge_filter = [1] * mol_edge_attr.shape[1]
    for efi in filter:
        edge_filter[efi] = 0
    edge_filter = np.array(edge_filter, dtype=bool)
    mol_edge_attr = mol_edge_attr[:, edge_filter]

    return torch.tensor(mol_edge_attr, dtype=torch.float)
