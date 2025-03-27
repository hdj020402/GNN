from typing import Dict, Callable

def attr_filter(main: Callable[[Dict], None], param: Dict):
    graph_attr_list = param['graph_attr_list']
    for i, attr in enumerate(graph_attr_list):
        param['graph_attr_list'] = [x for x in graph_attr_list if x != attr]
        main(param)
