import os, json, pickle
from typing import Dict, Optional

'''utils for gnn attribute.
'''

def one_hot(hot_idx: int, total_len: int):
    '''generate one hot repr according to selected index and
    total length.
    
    Args:
        hot_idx: the index chosen to be 1.
        total_len: how long should the repr be.
        
    Return:
        one_hot_list: [0, 0, 1, 0] for hot_idx=2, total_len=4.
    '''
    one_hot_list = []
    for i in range(total_len):
        if i == hot_idx:
            one_hot_list.append(1)
        else:
            one_hot_list.append(0)
    
    return one_hot_list

def read_attr(file: Optional[str], attr_type: str, length: int) -> Optional[Dict]:
    if file is None:
        attr_dict = None
    else:
        _, ext = os.path.splitext(file)
        if ext == '.json':
            with open(file, 'r') as f:
                attr_dict: Dict = json.load(f)
        elif ext == '.pkl':
            with open(file, 'rb') as f:
                attr_dict: Dict = pickle.load(f)
        else:
            raise ValueError(f"Unsupported file format: {ext}")
        length_list = [len(value) for value in attr_dict.values()]
        assert min(length_list) == max(length_list), f'{attr_type} attributes are not of the same length.'
        assert min(length_list) == length, f'{attr_type} attributes and SDF file are not of the same length.'
    return attr_dict
