from typing import Dict, List

def extract_keys_and_lists(d: Dict, parent_key='', sep='_') -> List: 
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(extract_keys_and_lists(v, new_key, sep))
        elif isinstance(v, list):
            items.append((new_key, v))
    return items

def recursive_merge(dicts: List[Dict]) -> Dict:
    if not dicts:
        return {}

    merged = {}

    for key in dicts[0].keys():
        if isinstance(dicts[0][key], dict):
            sub_dicts = [d[key] for d in dicts if key in d]
            merged[key] = recursive_merge(sub_dicts)
        else:
            merged[key] = [d.get(key) for d in dicts]

    return merged
