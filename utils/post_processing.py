import json, re
from typing import Dict, List
from utils.utils import recursive_merge

class ReadLog():
    def __init__(self, log_file: str, param: Dict) -> None:
        self.log_file = log_file
        self.param = param

    def get_performance(self) -> Dict:
        with open(self.log_file) as lf:
            text = lf.readlines()
        pattern = r'\{.*\}'
        dicts = []
        for line in text:
            if 'EarlyStopping' in line or 'Ending...' in line:
                break
            if not '"Epoch"' in line:
                continue
            match = re.search(pattern, line)
            dicts.append(json.loads(match.group(0)))
        log_info_dict = recursive_merge(dicts)

        return log_info_dict

    def get_feature(self) -> Dict:
        with open(self.log_file) as lf:
            text = lf.readlines()
        param = json.loads(text[1])
        feature = {
            'node': param['node_attr_list'],
            'edge': param['edge_attr_list'],
            'graph': param['graph_attr_list']
            }
        return feature

    def restart(self, start_epoch: int) -> List:
        with open(self.log_file) as lf:
            text = lf.readlines()
        pre_log_text = []
        i = 1
        for line in text:
            if i == start_epoch:
                break
            if '"Epoch"' in line:
                pre_log_text.append(line)
                i += 1
        return pre_log_text
