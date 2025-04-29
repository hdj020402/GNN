import yaml, os
from typing import Dict

KEYS = [
    'sdf_file',
    'node_attr_file',
    'edge_attr_file',
    'graph_attr_file',
    'weight_file',
    'atom_type',
    'default_node_attr',
    'default_edge_attr',
    'node_attr_list',
    'edge_attr_list',
    'graph_attr_list',
    'node_attr_filter',
    'edge_attr_filter',
    'pos',
    'target_list',
    'target_transform',
]

def reprocess(param: Dict) -> bool:
    try:
        with open(os.path.join(param['path'], 'processed/model_parameters.yml'), 'r', encoding = 'utf-8') as mp:
            param_pre: Dict = yaml.full_load(mp)
        sub_dict1 = {key: param[key] for key in KEYS if key in param}
        sub_dict2 = {key: param_pre[key] for key in KEYS if key in param_pre}
        return not sub_dict1 == sub_dict2
    except Exception:
        return True

