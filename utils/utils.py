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

import time

class Timer:
    def __init__(self) -> None:
        self.start_time = None
        self.end_time = None

    def start(self) -> None:
        self.start_time = time.perf_counter()

    def end(self) -> None:
        self.end_time = time.perf_counter()

    def get_tot_time(self) -> tuple[int, int, int, float]:
        if self.start_time is None or self.end_time is None:
            return (-1, -1, -1, -1.0)
        
        tot_time = self.end_time - self.start_time
        
        return Timer.convert_time(tot_time)

    def get_average_time(self, divisor: int) -> tuple[int, int, int, float]:
        if divisor <= 0:
            raise ValueError("Divisor must be greater than 0.")
            
        if self.start_time is None or self.end_time is None:
            return (-1, -1, -1, -1.0)
        
        tot_time = self.end_time - self.start_time
        avg_time = tot_time / divisor

        return Timer.convert_time(avg_time)

    @staticmethod
    def convert_time(seconds: float) -> tuple[int, int, int, float]:
        days, remainder = divmod(seconds, 86400)
        hours, remainder = divmod(remainder, 3600)
        minutes, seconds = divmod(remainder, 60)
        return int(days), int(hours), int(minutes), seconds
