import sys
import json
import os
from typing import Mapping, TypeVar, List, Callable, Union, Tuple, Set, Optional
import tqdm
import numpy as np
import ray
import multiprocessing as mp

T = TypeVar('T')
Primitive = Union[str, bool, int, float]
Config = List[Tuple[str, Primitive]]
Data = Mapping[str, List[float]]
GroupedExp = Mapping[Config, List[Data]]


def _parse_file(
        filepath: str,
        keys: List[str],
) -> Optional[Tuple[Config, Data]]:
    data = {key: [] for key in keys}
    config = None
    with open(filepath, 'r') as f:
        for line in f:
            json_line = json.loads(line)
            if config is None:
                config = json_line['config']
            for key in keys:
                try:
                    data[key].append(json_line[key])
                except KeyError:
                    continue
    if config is None:
        return None
    return tuple(sorted(list(config.items()))), data


@ray.remote
def _parse_file_ray(
        filepath: str,
        keys: List[str]
) -> Optional[Tuple[Config, Data]]:
    return _parse_file(filepath, keys)


def _collect_results_files(
        experiment_dir: str,
) -> List[str]:
    folders = [f for d in os.listdir(experiment_dir)
               if os.path.isdir((f := os.path.join(experiment_dir, d)))]
    results = []
    for f in folders:
        if os.path.isfile((result := os.path.join(f, 'result.json'))):
            results.append(result)
    return results

def _parse_file_mp(args):
    filepath, keys = args
    return _parse_file(filepath, keys)


def tuplize(obj):
    if isinstance(obj, list) or isinstance(obj, tuple):
        return tuple(tuplize(c) for c in obj)
    else:
        return obj


def load_data(
        experiment_dir: str,
        keys: List[str],
        no_ray: bool = False,
) -> GroupedExp:
    grouped_exp = dict()
    results_files = _collect_results_files(experiment_dir)
    pool = mp.Pool(mp.cpu_count())
    finished_files = []
    if no_ray:
        for results_file in tqdm.tqdm(results_files):
            finished_files.append(_parse_file(results_file, keys))
    else:
        res_and_keys = [(results_file, keys) for results_file in results_files]
        finished_files = list(tqdm.tqdm(pool.imap_unordered(_parse_file_mp, res_and_keys), total=len(res_and_keys)))
    for opt_config_data in finished_files:
        if opt_config_data is None:
            continue
        config, data = opt_config_data
        config = tuplize(config)
        if config in grouped_exp:
            grouped_exp[config].append(data)
        else:
            grouped_exp[config] = [data]
    return grouped_exp


def get_varying_fields(
        data: GroupedExp,
        exclude: List[str],
) -> Mapping[str, List[Primitive]]:
    fields = dict()
    for config in data:
        for field, value in config:
            if field in fields:
                fields[field].add(value)
            else:
                fields[field] = {value}
    fields = {k: v for k, v in fields.items() if len(v) > 1}
    return fields


def print_varying_fields(
        data: GroupedExp,
        exclude: List[str]
) -> None:
    for field, value in get_varying_fields(data, exclude).items():
        if field in exclude:
            continue
        print(f'{field} : {value}')


def select_by(
        data: GroupedExp,
        **field_values: Union[Primitive, Callable[[Primitive], bool]],
) -> GroupedExp:
    selected_data = dict()

    def satisfied(field, value):
        if callable(value):
            return value(dict_config[field])
        else:
            return dict_config[field] == value

    for config, exps in data.items():
        dict_config = dict(config)
        if any(not satisfied(field, value) for field, value in field_values.items()):
            continue
        selected_data[config] = exps
    return selected_data
        

def group_by(
        data: GroupedExp,
        *fields: str,
) -> GroupedExp:
    grouped_exps = dict()
    for config, exps in data.items():
        filtered_config = tuple([(field, value) for field, value in config if field in fields])
        if filtered_config not in grouped_exps:
            grouped_exps[filtered_config] = []
        grouped_exps[filtered_config].extend(exps)
    return grouped_exps


def map_groups(
        data: GroupedExp,
        fn: Callable[[List[List[float]]], T]
) -> Mapping[Config, T]:
    aggreged_data = dict()
    for config, exps in data.items():
        combined_data = dict()
        for exp in exps:
            for key, key_data in exp.items():
                if key in combined_data:
                    combined_data[key].append(key_data)
                else:
                    combined_data[key] = [key_data]
        for key in combined_data:
            combined_data[key] = fn(combined_data[key])
        aggreged_data[config] = combined_data
    return aggreged_data 


def map_individuals(
        data: GroupedExp,
        fn: Callable[[List[float]], T]
) -> Mapping[Config, List[T]]:
    return map_groups(data, lambda x: [fn(xx) for xx in x])


