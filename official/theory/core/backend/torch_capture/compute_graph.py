from __future__ import annotations
from typing import List, Dict, Any, Union, Optional
from dataclasses import dataclass, field, fields
from enum import IntFlag, auto
import sympy

__all__ = [
    'Graph', 'Node', 'Value','ValueKind',
    'export_graph_to_yaml',
    'create_graph_from_dict',
]

@dataclass
class Graph:
    name: str
    nodes: List[Node] = field(default_factory=list)
    values: List[Value] = field(default_factory=list)
    attrs: Dict[str, Any] = field(default_factory=dict)

@dataclass
class Node:
    ''':class:`Node` is essentially a record of function call.
    '''
    name: str
    target: str
    args: List[Value]
    kwargs: Dict[str, Value]
    result: Union[List[Value], Value]
    result_mask: Optional[List[bool]] = field(default=None)
    attrs: Dict[str, Any] = field(default_factory=dict)
    from_: Optional[List[str]] = field(default=None)
    scope: Optional[str] = field(default=None)

class ValueKind(IntFlag):
    ''':class:`ValueKind` denotes the kind of a tensor (which is an N-dimensional
        value, where the rank of a scalar is zero):

    - `input` value is an input of a compute graph
    - `output` value is an output of a compute graph
    - `parameter` value is a constant parameter / attribute, which may be
        considered as a special kind of `input` tensor
    - `tensor` value is a ranked tensor value
    - `scalar` value is an un-ranked scalar value
    '''
    input  = auto()
    output = auto()
    parameter = auto()
    tensor = auto()
    scalar = auto()

    @classmethod
    def _missing_(cls, value):
        new_value = value
        if isinstance(value, str):
            new_value = 0
            for field in value.split('|'):
                try:
                    new_value |= cls.__members__[field]
                except KeyError:
                    raise ValueError(f'unknown enum value: {field} for {cls.__name__}')
        elif isinstance(value, type(None)):
            new_value = 0

        return super()._missing_(new_value)

    def __repr__(self):
        if not self.value:
            return ''
        return '|'.join(m.name for m in self.__class__ if m.value & self.value)

    __str__ = __repr__

@dataclass
class Value:
    ''':class:`Tensor` records the information of values that got passed along in the compute graph.
    '''
    name: str
    kind: ValueKind
    dtype: str
    shape: List[Union[int, sympy.Expr]] = field(default_factory=list)
    stride: List[Union[int, sympy.Expr]] = field(default_factory=list)
    value: Optional[Any] = field(default=None)
    attrs: Dict[str, Any] = field(default_factory=dict)

def export_graph_to_yaml(g: Graph, path: Optional[str] = None):
    from .utils import IndentedBuffer, safe_json_dumps

    buf = IndentedBuffer(tabwidth=2)
    buf.writeline(f'name: {g.name}')

    if g.attrs:
        buf.writeline('attrs:')
        with buf.indent():
            for k, v in g.attrs.items():
                buf.writeline(f'{k}: {safe_json_dumps(v)}')

    if g.nodes:
        buf.writeline('nodes:')
        for node in g.nodes:
            buf.writeline(f'- name: {node.name}')
            with buf.indent():
                buf.writeline(f'target: {node.target}')
                if node.args:
                    buf.writeline(f'args: {safe_json_dumps(node.args)}')
                if node.kwargs:
                    buf.writeline(f'kwargs:')
                    with buf.indent():
                        for k, v in node.kwargs.items():
                            buf.writeline(f'{k}: {safe_json_dumps(v)}')
                if node.result is not None:
                    buf.writeline(f'result: {safe_json_dumps(node.result)}')
                if node.result_mask:
                    buf.writeline(f'result_mask: {safe_json_dumps(node.result_mask)}')
                if node.attrs:
                    buf.writeline(f'attrs: {safe_json_dumps(node.attrs)}')
                if node.scope is not None:
                    buf.writeline(f'scope: {node.scope}')
                if node.from_ is not None:
                    buf.writeline(f'from: |')
                    with buf.indent():
                        buf.writelines(node.from_)

    if g.values:
        buf.writeline('values:')
        for value in g.values:
            buf.writeline(f'- name: {value.name}')
            with buf.indent():
                buf.writeline(f'kind: {value.kind}')
                buf.writeline(f'dtype: {value.dtype}')
                if value.value is not None:
                    buf.writeline(f'value: {value.value}')
                if value.shape:
                    buf.writeline(f'shape: {safe_json_dumps(value.shape)}')
                if value.stride:
                    buf.writeline(f'stride: {safe_json_dumps(value.stride)}')
                if value.attrs:
                    buf.writeline(f'attrs: {safe_json_dumps(value.attrs)}')

    result = buf.getvalue()
    if path is not None:
        with open(path, 'w') as fh:
            fh.write(result)
    return result

def create_graph_from_dict(data: Dict[str, Any]):
    NODE_KEYS  = {'nodes', 'calls'}
    VALUE_KEYS = {'values', 'tensors'}
    ATTRS_KEYS = {'attrs'}
    VALID_KEYS = ATTRS_KEYS | NODE_KEYS | VALUE_KEYS | {'name'}

    keys = set(data.keys())

    assert keys < VALID_KEYS, \
        f'invalid keys: {keys - VALID_KEYS}'

    result = Graph(name=data.get('name', 'unnamed'))

    value_keys = keys & VALUE_KEYS
    if value_keys:
        assert len(value_keys) == 1, \
            f'more than one valid value keys are present: {value_keys}'

        for value in data[value_keys.pop()]:
            result.values.append(_create_value_from_dict(value))

    _update_symbolic_shapes(result.values)

    node_keys = keys & NODE_KEYS
    if node_keys:
        assert len(node_keys) == 1, \
            f'more than one valid node keys are present: {node_keys}'

        named_values = {value.name: value for value in result.values}

        for node in data[node_keys.pop()]:
            result.nodes.append(_create_node_from_dict(node, named_values))

    attrs_keys = keys & ATTRS_KEYS
    if attrs_keys:
        assert len(attrs_keys) == 1, \
            f'more than one valid attribute keys are present: {attrs_keys}'
        result.attrs = {k: v for k, v in data[attrs_keys.pop()].items()}

    return result

class VisitorBase:
    def _visit_default(self, key: str, value: Any, result: Any):
        raise NotImplementedError

    def visit(self, key: str, value: Any, result: Any):
        getattr(self, f'_visit_{key}', self._visit_default)(key, value, result)

class ValueVisitor(VisitorBase):
    def _visit_default(self, key: str, value: Any, result: Dict):
        result[key] = value
    def _visit_kind(self, key: str, value: Any, result: Dict):
        result[key] = ValueKind(value)

def _create_value_from_dict(data: Dict[str, Any]):
    VALID_KEYS = {f.name for f in fields(Value)}
    keys = set(data.keys())
    assert keys < VALID_KEYS, \
        f'invalid keys: {keys - VALID_KEYS}'

    result  = {}
    visitor = ValueVisitor()
    for key, value in data.items():
        visitor.visit(key, value, result)

    # handle symbolic shapes
    assert 'dtype' in result, f'dtype is not defined. value: {result}'
    if result['dtype'] == 'torch.SymInt':
        assert 'value' in result, f'value not defined for a SymInt: {result}'
        result['value'] = sympy.simplify(result['value'])

    return Value(**result)

class NodeVisitor(VisitorBase):
    def __init__(self, values: Dict):
        super().__init__()
        self.values = values

    def _visit_default(self, key: str, value: Any, result: Dict):
        result[key] = value

    def _visit_args(self, key: str, value: Any, result: Dict):
        result[key] = [self._convert_value(v) for v in value]

    def _visit_kwargs(self, key: str, value: Any, result: Dict):
        k, v = zip(*value.items())
        result[key] = dict(zip(k, [self._convert_value(vv) for vv in v]))

    def _visit_result(self, key: str, value: Any, result: Dict):
        key = 'result'
        if isinstance(value, (list, tuple)):
            result[key] = [self._convert_value(v) for v in value]
        else:
            result[key] = self._convert_value(value)

    def _visit_result_mask(self, key: str, value: Any, result: Dict):
        if not value:
            return
        assert isinstance(value, (list, tuple))
        assert all(isinstance(vv, bool) for vv in value)
        result[key] = value

    def _visit_from(self, key: str, value: Any, result: Dict):
        result['from_'] = value.strip().split('\n')

    _visit_results = _visit_result
    _visit_from_   = _visit_from

    def _convert_value(self, value: Any):
        if isinstance(value, str):
            return self.values.get(value, value)
        elif isinstance(value, (list, tuple)) \
            and all(isinstance(v, str) for v in value if v is not None):
            return [self.values.get(v, v) for v in value]
        else:
            return value

def _create_node_from_dict(data: Dict[str, Any], values: Dict[str, Value]):
    RESULT_KEYS = {'result', 'results'}
    FROM_KEYS   = {'from', 'from_'}
    VALID_KEYS  = {f.name for f in fields(Node)} | RESULT_KEYS | FROM_KEYS

    keys = set(data.keys())
    assert keys < VALID_KEYS, \
        f'invalid keys: {keys - VALID_KEYS}'
    assert len(keys & RESULT_KEYS) <= 1, \
        f'invalid result keys: {keys & RESULT_KEYS}'
    assert len(keys & FROM_KEYS) <= 1, \
        f'invalid from keys: {keys & FROM_KEYS}'

    result  = {'args': tuple(), 'kwargs': {}}
    visitor = NodeVisitor(values)
    for key, value in data.items():
        visitor.visit(key, value, result)
    return Node(**result)

def _update_symbolic_shapes(values: List[Value]):
    def transform(expr):
        if isinstance(expr, str):
            result = sympy.simplify(expr)
        else:
            result = expr
        return result

    for value in values:
        value.shape  = [transform(dim) for dim in value.shape]
        value.stride = [transform(dim) for dim in value.stride]
