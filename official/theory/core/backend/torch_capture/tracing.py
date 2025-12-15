from __future__ import annotations
from typing import Dict, List, Union, Callable, Optional, Any
from types import GeneratorType
from dataclasses import dataclass, field
from operator import getitem
from contextlib import contextmanager, nullcontext
from itertools import count
from threading import local

import re
import os
import sys
import shutil
import traceback
import datetime
import getpass
import logging
import importlib
import importlib.metadata
import yaml

import torch

from torch.fx import GraphModule, Node
from torch.fx.operator_schemas import get_signature_for_torch_op

from .utils import IndentedBuffer, load_yaml_file, safe_json_dumps
from . import compute_graph as cg

THREAD_LOCAL = local()
SCOPE = 'SCOPE'

def capture(f: Union[None, torch.nn.Module, Callable],
    *,
    call_stack_depth: int = 3,
    dynamic: bool = False,
    enable_flops_counter: bool = True,
    enable_faster_cpu_mixed_precision_mode: bool = True,
    verbose: bool = False,
    decompositions: Optional[Callable] = None,
    forward_func = None,
):
    PYTHON_VERSION_CURRENT  = sys.version_info[:3]
    PYTHON_VERSION_REQUIRED = (3, 8, 10)
    if PYTHON_VERSION_CURRENT < PYTHON_VERSION_REQUIRED:
        raise RuntimeError(
            f'required python version >= {PYTHON_VERSION_REQUIRED}, '
            f'got: {PYTHON_VERSION_CURRENT}')

    TORCH_VERSION_REQUIRED = torch.torch_version.TorchVersion("2.0.0")
    TORCH_VERSION_CURRENT  = torch.torch_version.__version__

    # if TORCH_VERSION_CURRENT < TORCH_VERSION_REQUIRED:
    #     raise NotImplementedError(
    #         f'required torch version is {TORCH_VERSION_REQUIRED}, '
    #         f'got: {TORCH_VERSION_CURRENT}')

    from .dynamo import make_simple_dynamo_backend
    callable = TracedCallable(f, verbose=verbose)

    # delay the import of core_aten_decompositions to allow for lower-versioned torch
    from torch._decomp import core_aten_decompositions
    decompositions = decompositions or core_aten_decompositions

    @make_simple_dynamo_backend(decompositions=decompositions)
    def _backend(gm: GraphModule, example_inputs: List[torch.Tensor]):
        id = len(callable.graphs)
        callable.graphs.append(gm)

        def wrapped(*args):
            if enable_flops_counter:
                # delay the import of stats to allow other components such as
                # trace loader to work with torch with lower version (e.g. 1.x)
                from .stats import OpCounter
                counter = OpCounter(display=False)
            else:
                counter = nullcontext()

            if enable_faster_cpu_mixed_precision_mode:
                from .mixed_precision import MixedPrecisionMode
                mixed_p = MixedPrecisionMode()
            else:
                mixed_p = nullcontext()

            with counter, mixed_p:
                if forward_func:
                    result = gm.forward_func(*args)
                else:
                    result = gm.forward(*args)

            callable.record_call(id, args, result,
                depth=call_stack_depth,
                stats=counter.get_counts())

            return result
        return wrapped

    callable.compiled_fn = torch._dynamo.optimize(_backend, dynamic=dynamic)(f)
    return callable

@contextmanager
def scope(name: str):
    orig = getattr(THREAD_LOCAL, SCOPE, None)
    setattr(THREAD_LOCAL, SCOPE, name)
    try:
        yield
    except:
        raise
    finally:
        setattr(THREAD_LOCAL, SCOPE, orig)

# {{{ data structures and friends for tracing

def _compute_type_name(t: type):
    return (t.__qualname__ if t.__module__ == 'builtins' else
            t.__module__ + '.' + t.__qualname__)

REGEX_TARGET = re.compile(r'<built-in function .*>')
def _compute_target_name(t: Any):
    s = str(t)
    if REGEX_TARGET.match(s):
        return t.__qualname__
    return s

@dataclass
class TracedCallable:
    @dataclass
    class CallRecord:
        id: int
        args: List[Any]
        result: Any
        scope: Optional[str] = None
        stats: Optional[Any] = None
        callsite: Optional[str] = None

        def values(self):
            yield from self.args
            yield self.result

    fn: Callable
    compiled_fn: Optional[Callable] = None
    graphs: List[GraphModule] = field(default_factory=list)

    started: bool = True
    records: List[CallRecord] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    verbose: bool = False

    def __post_init__(self):
        import inspect
        self.__signature__ = inspect.signature(self.fn)

    def __call__(self, *args, **kwargs):
        return self.compiled_fn(*args, **kwargs)

    def stop_capture(self):
        self.started = False

    def start_capture(self):
        self.started = True

    @contextmanager
    def capture(self, scope_name: Optional[str] = None):
        orig = self.started
        self.started = True
        if scope_name is not None:
            with scope(scope_name):
                yield
        else:
            yield
        self.started = orig

    def record_call(self, id_: int, args: List[Any], result: Any, *,
        depth: int = 3,
        stats: Optional[Any] = None,
    ):
        if not self.started:
            return

        callsite = list(self._find_call_site(depth=depth))
        self.records.append(self.CallRecord(
            id_, args, result,
            callsite=callsite,
            stats=stats,
            scope=getattr(THREAD_LOCAL, SCOPE, None)))

    def _find_call_site(self, depth):
        remaining_depth = -1
        for f, lineno in traceback.walk_stack(None):
            co = f.f_code
            filename = co.co_filename
            name = co.co_name

            if remaining_depth > 0:
                yield f'{filename}:{lineno} in {name}'
                remaining_depth -= 1
            if remaining_depth == 0:
                return

            if filename.endswith('torch/_dynamo/eval_frame.py'):
                remaining_depth = depth
                continue

    def add_metadata(self, key: str, value: Any):
        self.metadata[key] = value

    def get_metadata(self):
        result = {}
        result['time'] = datetime.datetime.now().isoformat(timespec="seconds")
        result['user'] = getpass.getuser()
        result['cwd']  = os.getcwd()
        for package in [
            'torch_capture',
            'torch', 'torchvision', 'torchaudio',
            'transformers', 'diffusers',
        ]:
            try:
                module = importlib.import_module(package)
                result[package] = getattr(module, '__version__', importlib.metadata.version(package))
            except:
                pass

        if self.metadata:
            result['user_data'] = self.metadata.copy()
        return result

    def get_stats(self):
        return [{'name': f'call_{i}', **record.stats} for i, record in enumerate(self.records)
                if record.stats is not None]

    def get_workload(self):
        subgraphs  = self.get_compute_subgraphs()
        submodules = [self.graphs[g.attrs['id']] for g in subgraphs]

        return TracedWorkload(
            graph=self.get_compute_graph(),
            subgraphs=subgraphs,
            submodules=submodules,
            stats=self.get_stats(),
            metadata=self.get_metadata())

    def get_compute_subgraphs(self):
        convert = FxGraphConverter(verbose=self.verbose)
        graph_ids = {record.id for record in self.records}
        return [convert(name=f'g_{id}', gm=self.graphs[id], attrs={'id': id})
                for id in graph_ids]

    def get_compute_graph(self):
        result = cg.Graph(name='trace')
        values = {}

        def compute_value(value: Any):
            if id(value) in values:
                return values[id(value)]

            kwargs = {}
            kwargs['dtype'] = _compute_type_name(type(value))
            kwargs['kind']  = cg.ValueKind(0)

            if isinstance(value, (int, float, str, bool)):
                kwargs['value'] = value
                kwargs['kind']  = cg.ValueKind.scalar
            elif isinstance(value, (list, tuple)):
                kwargs['value'] = value
                kwargs['attrs'] = {'length': len(value)}
            elif isinstance(value, torch.Tensor):
                kwargs['dtype'] = value.dtype
                kwargs['shape'] = value.shape
                kwargs['stride'] = value.stride()
                kwargs['kind']  = cg.ValueKind.tensor

            name = f'v_{len(values)}'
            result = cg.Value(name=name, **kwargs)
            values[name] = result
            return result

        for i, record in enumerate(self.records):
            record_args = [compute_value(arg) for arg in record.args]
            if isinstance(record.result, (list, tuple)):
                record_result = [compute_value(value) for value in record.result]
            else:
                record_result = compute_value(record.result)

            result.nodes.append(cg.Node(
                name=f'call_{i}',
                target=f'g_{record.id}',
                args=record_args,
                kwargs={},
                result=record_result,
                scope=record.scope,
                from_=record.callsite))

        result.values = values.values()
        return result

    def to_folder(self, path: str):
        self.get_workload().to_folder(path)

@dataclass
class TracedWorkload:
    ''':class:`TracedWorkload` is the data structure that can be exported and imported.
    '''
    graph: cg.Graph
    subgraphs: List[cg.Graph]
    metadata: Dict[str, str] = field(default_factory=dict)
    attrs: Dict[str, Any] = field(default_factory=dict)
    stats: List[Any] = field(default_factory=list)
    submodules: List[GraphModule] = field(default_factory=list)

    def __post_init__(self):
        if not self.submodules:
            self.submodules = [None] * len(self.subgraphs)
        assert len(self.submodules) == len(self.subgraphs)

    def to_folder(self, path: str):
        shutil.rmtree(path, ignore_errors=True)
        os.makedirs(path, exist_ok=True)

        self._export_stats(os.path.join(path, 'stats.yaml'))
        self._export_metadata(os.path.join(path, 'meta.yaml'))
        self._export_env(os.path.join(path, 'env.txt'))
        cg.export_graph_to_yaml(self.graph, os.path.join(path, 'trace.yaml'))

        for graph, gm in zip(self.subgraphs, self.submodules):
            graph_output = os.path.join(path, f'{graph.name}')
            if gm is not None:
                gm.to_folder(graph_output)
            cg.export_graph_to_yaml(graph, os.path.join(graph_output, 'compute_graph.yaml'))

    def _export_metadata(self, path: str):
        buf = IndentedBuffer(tabwidth=2)
        buf.writeline('metadata:')
        with buf.indent():
            for k, v in self.metadata.items():
                if isinstance(v, dict):
                    buf.writeline(f'{k}:')
                    with buf.indent():
                        for kk, vv in v.items():
                            buf.writeline(f'{kk}: {safe_json_dumps(vv)}')
                else:
                    buf.writeline(f'{k}: {safe_json_dumps(v)}')

        with open(path, 'w') as fh:
            fh.write(buf.getvalue())

    def _export_env(self, path: str):
        try:
            from torch.utils.collect_env import get_pretty_env_info
            with open(path, 'w') as fh:
                fh.write(get_pretty_env_info())
        except Exception as e:
            logging.warning('failed to collect env info: %s', e)

    def _export_stats(self, path: str):
        buf = IndentedBuffer(tabwidth=2)
        buf.writeline('stats:')
        for stats in self.stats:
            buf.writeline(f'- name: {stats["name"]}')
            with buf.indent():
                for key, data in stats.items():
                    if key == 'name':
                        continue
                    buf.writeline(f'{key}:')
                    with buf.indent():
                        buf.writelines(yaml.dump(data).strip().split('\n'))

        with open(path, 'w') as fh:
            fh.write(buf.getvalue())

def create_workload_from_folder(path: str):
    if not os.path.isdir(path):
        raise RuntimeError(f'not a valid directory: {path}')

    metadata_file = os.path.join(path, 'meta.yaml')
    metadata = load_yaml_file(metadata_file) if os.path.exists(metadata_file) else {}

    trace_file = os.path.join(path, 'trace.yaml')
    if not os.path.exists(trace_file):
        raise RuntimeError(f'no trace file found in: {path}')
    compute_graph = cg.create_graph_from_dict(load_yaml_file(trace_file))

    ok = True
    compute_subgraphs = []
    for subgraph in {node.target for node in compute_graph.nodes}:
        subgraph_file = os.path.join(path, subgraph, 'compute_graph.yaml')
        if not os.path.exists(subgraph_file):
            logging.error('subgraph compute graph file not found: %s', subgraph_file)
            ok = False
            continue
        compute_subgraphs.append(cg.create_graph_from_dict(load_yaml_file(subgraph_file)))

    if not ok:
        raise RuntimeError(f'errors encountered when loading trace from: {path}')

    compute_subgraphs.sort(key=lambda g: g.attrs['id'])

    return TracedWorkload(
        graph=compute_graph,
        subgraphs=compute_subgraphs,
        metadata=metadata,
        attrs={'directory': path})

# }}}

# {{{ utils for torch.fx graph conversion

class FxGraphConverter:
    def __init__(self, *, verbose: bool = False):
        self.verbose = verbose

        self.nodes : Dict[str, cg.Node] = {}
        self.tensors : Dict[str, cg.Value] = {}
        self.getitems: Dict[str, str] = {}

        self.value_counter = count()

    def reset(self):
        self.nodes = {}
        self.tensors = {}
        self.getitems = {}
        self.value_counter = count()

    def __call__(self,
        name: str,
        gm: GraphModule,
        attrs: Optional[Dict[str, Any]] = None,
    ):
        self.reset()

        for node in gm.graph.nodes:
            getattr(self, f'visit_{node.op}', self.visit)(node, gm)

        return cg.Graph(
            name=name,
            nodes=list(self.nodes.values()),
            values=list(self.tensors.values()),
            attrs=attrs or {})

    def visit(self, node: Node, gm: GraphModule):
        raise RuntimeError(f'node op {node.op} is not expected. name: {node.name}')

    def visit_placeholder(self, node: Node, gm: GraphModule):
        node_name = self.compute_full_name(node)
        assert node_name not in self.tensors
        self.create_value(node_name, node=node, kind=cg.ValueKind.input)
        self.getitems[node_name] = node_name

    def visit_output(self, node: Node, gm: GraphModule):
        for output in self.flatten_args(node.args):
            if not output:
                continue

            tensor_name = self.compute_full_name(output)
            tensor_name = self.getitems.get(tensor_name, tensor_name)
            assert tensor_name in self.tensors, \
                f'tensor {tensor_name} not found'
            self.tensors[tensor_name].kind |= cg.ValueKind.output

    def visit_get_attr(self, node: Node, gm: GraphModule):
        value = getattr(gm, node.target, None)
        if (isinstance(value, GraphModule) or
            type(value).__qualname__.split(".")[-1] == 'GraphModuleImpl'):
            raise NotImplementedError(
                'nested graph module is not supported: '
                'getting module attribute value is GraphModuleImpl type')

        node_name = self.compute_full_name(node)
        assert node_name not in self.tensors

        self.create_value(node_name,
            node=node,
            kind=cg.ValueKind.parameter,
            attrs={'target': node.target})
        self.getitems[node_name] = node_name

    def visit_call_function(self, node: Node, gm: GraphModule):
        node_name = self.compute_full_name(node)
        if node.target == getitem:
            target, index = node.args
            target_name = self.compute_full_name(target)
            self.getitems[node_name] = f'{self.getitems[target_name]}.{index}'
            return

        node_args = self.convert_args(node.args)
        node_kwargs = self.convert_args(node.kwargs)

        bound_args = None
        signatures = get_signature_for_torch_op(node.target) or []
        for signature in signatures:
            try:
                bound_args = signature.bind(*node_args, **node_kwargs)
                break
            except TypeError:
                continue

        if bound_args is not None:
            node_args = tuple()
            node_kwargs = bound_args.arguments

        node_args = {
            'name': node_name,
            'target': _compute_target_name(node.target),
            'args': node_args,
            'kwargs': node_kwargs,
            'attrs': {},
        }

        if self.verbose:
            node_args['from_'] = node.meta.get('stack_trace', '').strip().split('\n')

        user_is_get_item = [user.target == getitem for user in node.users]
        assert sum(user_is_get_item) in (0, len(user_is_get_item)), \
            f'not all users are getitem. node: {node_name} : {node.users}'

        name = self.new_value_name()
        if any(user_is_get_item):
            results = []
            users = {user.args[1]: user for user in node.users}
            for index in sorted(users.keys()):
                results.append(self.create_value(
                    name=f'{name}.{index}',
                    node=users[index]))
            node_args['result'] = results
            node_args['result_mask'] = self.compute_result_mask(node)
        else:
            node_args['result'] = self.create_value(name=name, node=node)

        self.getitems[node_name] = name
        self.nodes[node_name] = cg.Node(**node_args)

    def new_value_name(self):
        while True:
            name = f'v{next(self.value_counter)}'
            if name not in self.tensors:
                return name

    def compute_result_mask(self, node: Node):
        value = node.meta.get('val')
        if not isinstance(value, tuple):
            return

        mask = [False] * len(value)
        for user in node.users:
            _, id = user.args
            mask[id] = True
        return mask

    def create_value(self,
        name: str, *,
        node: Node,
        kind: cg.ValueKind = cg.ValueKind(0),
        attrs: Optional[Dict[str, Any]] = None,
    ):
        meta = node.meta.get('tensor_meta')
        if meta is not None:
            result = cg.Value(
                name=name,
                kind=kind | cg.ValueKind.tensor,
                dtype=meta.dtype,
                shape=self.convert_args(meta.shape),
                stride=self.convert_args(meta.stride),
                attrs=attrs or {})
        else:
            value = node.meta.get('val')
            if value is None:
                logging.warning('no value found for node: %s meta: %s', node.name, node.meta)

            result = cg.Value(
                name=name,
                kind=kind | cg.ValueKind.scalar,
                dtype=_compute_type_name(type(value)),
                value=value,
                attrs=attrs or {})

        self.tensors[name] = result
        return result

    def convert_args(self, data):
        if isinstance(data, Node):
            node_name = self.compute_full_name(data)
            return self.getitems.get(node_name, node_name)
        if isinstance(data, (tuple, list, set, GeneratorType)):
            return list([self.convert_args(d) for d in data])
        if isinstance(data, dict):
            return dict({k: self.convert_args(v) for k, v in data.items()})
        if isinstance(data, (torch.dtype, torch.memory_format, torch.device)):
            return str(data)
        if isinstance(data, (torch.SymInt, torch.SymFloat, torch.SymBool)):
            return str(data)

        return data

    def flatten_args(self, args):
        for arg in args:
            if isinstance(arg, (tuple, list)):
                yield from arg
            else:
                yield arg

    @staticmethod
    def compute_full_name(node: Node):
        stack = node.meta.get('nn_module_stack')
        if not stack:
            return node.name

        prev = ''
        hierarchy = []
        try:
            for key in stack.keys():
                assert not prev or key.startswith(prev), \
                    f'strange stack: {key}, prev: {prev}'
                hierarchy.append(key[len(prev):])
                prev = key
        except AssertionError:
            hierarchy = list(stack.keys())[-1:]

        hierarchy.append(node.name)
        return '/'.join(hierarchy)

# }}}
