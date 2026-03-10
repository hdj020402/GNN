'''Molecular featurizer: converts RDKit Mol objects to node/edge attribute tensors.
'''
import warnings
import numpy as np
import torch
from rdkit import Chem
from rdkit.Chem.rdchem import BondType as BT
from rdkit.Chem.rdchem import HybridizationType


# Bond type mapping: rdkit BondType → one-hot index
_BOND_TYPE = {BT.SINGLE: 0, BT.DOUBLE: 1, BT.TRIPLE: 2, BT.AROMATIC: 3}

# Hybridization mapping: rdkit HybridizationType → one-hot index (OTHER maps to index 5)
_HYBRIDIZATION_TYPE = {
    HybridizationType.SP: 0,
    HybridizationType.SP2: 1,
    HybridizationType.SP3: 2,
    HybridizationType.SP3D: 3,
    HybridizationType.SP3D2: 4,
}


def one_hot(hot_idx: int, total_len: int) -> np.ndarray:
    '''Return a 1-D float32 numpy array of length total_len
    with position hot_idx set to 1.0 and all others 0.0.
    '''
    arr = np.zeros(total_len, dtype=np.float32)
    arr[hot_idx] = 1.0
    return arr


def get_node_attr(
    mol: Chem.rdchem.Mol,
    default_node_attr: dict,
    atom_type: list,
    _node_attr_dict: dict[str, list],
    node_attr_list: list,
    ) -> tuple[torch.Tensor, torch.Tensor]:
    '''Generate node feature matrix and 3-D coordinates for one molecule.

    Args:
        mol: RDKit Mol object with N atoms.
        default_node_attr: dict of {feature_name: bool} controlling which
            default features to include.
        atom_type: list of element symbols used for one-hot encoding.
            Unknown elements are mapped to the last slot (UNK).
        _node_attr_dict: per-molecule custom node attributes keyed by column name.
        node_attr_list: column names to read from _node_attr_dict.

    Returns:
        x   (torch.Tensor, shape [N, D]): node feature matrix.
        pos (torch.Tensor, shape [N, 3]): atomic coordinates.
    '''
    conf = mol.GetConformer()
    pos = torch.tensor(conf.GetPositions(), dtype=torch.float)

    type_idx_hot = []
    atomic_number = []
    aromatic = []
    num_neighbors = []
    num_hs = []
    hybridization = []
    in_ring = []
    formal_charge = []

    for atom in mol.GetAtoms():
        atom: Chem.rdchem.Atom
        try:
            type_idx = atom_type.index(atom.GetSymbol())
        except ValueError:
            type_idx = len(atom_type) - 1
            warnings.warn(f"Atom '{atom.GetSymbol()}' not in atom_type — mapped to UNK.")
        type_idx_hot.append(one_hot(type_idx, len(atom_type)))
        atomic_number.append(atom.GetAtomicNum())
        aromatic.append(1 if atom.GetIsAromatic() else 0)
        neighbors = atom.GetNeighbors()
        num_neighbors.append(len(neighbors))
        nhs = sum(1 for nb in neighbors if nb.GetAtomicNum() == 1)
        num_hs.append(nhs)
        hyb_idx = _HYBRIDIZATION_TYPE.get(atom.GetHybridization(), len(_HYBRIDIZATION_TYPE))
        hybridization.append(one_hot(hyb_idx, len(_HYBRIDIZATION_TYPE) + 1))
        in_ring.append(1 if atom.IsInRing() else 0)
        formal_charge.append(atom.GetFormalCharge())

    default_node_attr_dict = {
        'ele_type':       type_idx_hot,
        'atomic_number':  atomic_number,
        'aromatic':       aromatic,
        'num_neighbors':  num_neighbors,
        'num_hs':         num_hs,
        'hybridization':  hybridization,
        'in_ring':        in_ring,
        'formal_charge':  formal_charge,
    }

    mol_node_attr_list = []
    for key, enabled in default_node_attr.items():
        if not enabled:
            continue
        raw = np.array(default_node_attr_dict[key])
        # Multi-dim features (ele_type, hybridization) are already 2-D after np.array;
        # scalar features need an explicit column axis.
        if raw.ndim == 1:
            raw = raw[:, np.newaxis]
        mol_node_attr_list.append(raw)

    # custom per-molecule attributes
    node_attr_mat = []
    for col in node_attr_list:
        node_attr_mat.append(
            np.array(_node_attr_dict[col]).reshape(len(_node_attr_dict[col]), -1)
        )
    try:
        node_attr_mat = np.concatenate(node_attr_mat, axis=1, dtype=float)
    except ValueError:
        node_attr_mat = np.array(node_attr_mat, dtype=float)

    if node_attr_mat.size:
        mol_node_attr_list.append(node_attr_mat)

    mol_node_attr = np.concatenate(mol_node_attr_list, axis=1, dtype=float)

    return torch.tensor(mol_node_attr, dtype=torch.float), pos


def get_adj_mat(mol: Chem.rdchem.Mol) -> torch.Tensor:
    '''Return the directed edge index tensor (shape [2, 2M]) for one molecule.'''
    src, dst = [], []
    for bond in mol.GetBonds():
        u, v = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        src += [u, v]
        dst += [v, u]
    return torch.tensor([src, dst])


def get_edge_attr(
    mol: Chem.rdchem.Mol,
    default_edge_attr: dict,
    _edge_attr_dict: dict[str, list],
    edge_attr_list: list,
    ) -> torch.Tensor | None:
    '''Generate edge feature matrix for one molecule.

    Args:
        mol: RDKit Mol object with M bonds (→ 2M directed edges).
        default_edge_attr: dict of {feature_name: bool} controlling which
            default features to include.  The nested bond_length sub-dict is
            handled separately in graph_dataset (distance appended after graph
            construction), so it is ignored here.
        _edge_attr_dict: per-molecule custom edge attributes keyed by column name.
        edge_attr_list: column names to read from _edge_attr_dict.

    Returns:
        edge_attr (torch.Tensor, shape [2M, D]) or None if no features are enabled.
    '''
    edge_type = []
    conjugated = []
    bond_in_ring = []
    for bond in mol.GetBonds():
        bond: Chem.rdchem.Bond
        edge_type     += 2 * [_BOND_TYPE[bond.GetBondType()]]
        conjugated    += 2 * [1 if bond.GetIsConjugated() else 0]
        bond_in_ring  += 2 * [1 if bond.IsInRing() else 0]

    default_edge_attr_dict = {
        'edge_type':    [one_hot(t, len(_BOND_TYPE)) for t in edge_type],
        'conjugated':   conjugated,
        'bond_in_ring': bond_in_ring,
    }

    mol_edge_attr_list = []
    for key, enabled in default_edge_attr.items():
        if not enabled or key not in default_edge_attr_dict:
            continue
        raw = np.array(default_edge_attr_dict[key])
        if raw.ndim == 1:
            raw = raw[:, np.newaxis]
        mol_edge_attr_list.append(raw)

    try:
        mol_edge_attr = np.concatenate(mol_edge_attr_list, axis=1, dtype=float)
    except ValueError:
        return None

    # custom per-molecule attributes
    edge_attr_mat = []
    for col in edge_attr_list:
        edge_attr_mat.append(
            np.array(_edge_attr_dict[col]).reshape(len(_edge_attr_dict[col]), -1)
        )
    try:
        edge_attr_mat = np.concatenate(edge_attr_mat, axis=1, dtype=float)
    except ValueError:
        edge_attr_mat = np.array(edge_attr_mat, dtype=float)

    if edge_attr_mat.size:
        mol_edge_attr = np.concatenate([mol_edge_attr, edge_attr_mat], axis=1)

    return torch.tensor(mol_edge_attr, dtype=torch.float)
