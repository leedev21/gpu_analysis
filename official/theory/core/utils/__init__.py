import os
import json
from functools import wraps
from typing import Callable, Optional, TypeVar, overload
from typing_extensions import ParamSpec
from omegaconf import OmegaConf
from theory.core.utils.op_info_parser import Parser


def each_file(dir, endwith='', with_path=False):
    paths = os.walk(dir)
    for path, dir_lst, file_lst in paths:
        for file_name in file_lst:
            need_to_run = file_name.endswith(endwith) if isinstance(endwith, str) else any(file_name.endswith(k) for k in endwith)
            if need_to_run:
                if with_path:
                    yield path, os.path.join(path, file_name)
                else:
                    yield os.path.join(path, file_name)


def read_json(path, json_file):
    load_dict = None
    with open(os.path.join(path, json_file), 'r', encoding='utf-8') as f:
        load_dict = json.load(f)
        # print('=======================' + 'json_file' + '=======================>')
        json_report = json.dumps(load_dict, indent=4, sort_keys=False)
        # print("{}".format(json_report))
        # for k, v in load_dict.items():
        #     print(k, v)
    return load_dict


def read_file(path, file):
    lines = None
    with open(os.path.join(path, file), 'r', encoding='utf-8') as f:
        lines = f.readlines()
        # print('=======================' + 'read_file' + '=======================>')
        # for k, v in enumerate(lines):
        #     print(k, v)
    return lines


def read_yaml_by_omegaconf(path):
    return OmegaConf.to_container(OmegaConf.load(path), resolve=True)


def save_file(path, file, lines):
    with open(os.path.join(path, file), 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))


def collection_path():
    return os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'core/collections')

def backend_path():
    return os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'core/backend')


def _get_rank() -> Optional[int]:
    # SLURM_PROCID can be set even if SLURM is not managing the multiprocessing,
    # therefore LOCAL_RANK needs to be checked first
    rank_keys = ("RANK", "LOCAL_RANK", "SLURM_PROCID", "JSM_NAMESPACE_RANK")
    for key in rank_keys:
        rank = os.environ.get(key)
        if rank is not None:
            return int(rank)
    # None to differentiate whether an environment variable was set at all
    return None


T = TypeVar("T")
P = ParamSpec("P")


@overload
def rank_zero_only(fn: Callable[P, T]) -> Callable[P, Optional[T]]:
    """Rank zero only."""

@overload
def rank_zero_only(fn: Callable[P, T], default: T) -> Callable[P, T]:
    """Rank zero only."""

def rank_zero_only(fn: Callable[P, T], default: Optional[T] = None) -> Callable[P, Optional[T]]:
    @wraps(fn)
    def wrapped_fn(*args: P.args, **kwargs: P.kwargs) -> Optional[T]:
        rank = getattr(rank_zero_only, "rank", None)
        if rank is None:
            raise RuntimeError("The `rank_zero_only.rank` needs to be set before use")
        if rank == 0:
            return fn(*args, **kwargs)
        return default

    return wrapped_fn

# add the attribute to the function but don't overwrite in case Trainer has already set it
rank_zero_only.rank = getattr(rank_zero_only, "rank", _get_rank() or 0)