from __future__ import annotations
from typing import NamedTuple, Any, Optional, Iterable
from dataclasses import dataclass
from io import StringIO

import re
import math
import time
import textwrap
import contextlib
import json
import yaml

import torch

from . import compute_graph

class LineContext(NamedTuple):
    context: Any

class IndentedBuffer:
    def __init__(self, initial_indent=0, tabwidth=4):
        self._lines = []
        self._indent = initial_indent
        self._tabwidth = tabwidth

    def getvaluewithlinemap(self) -> tuple[str, list[tuple[int, LineContext]]]:
        buf = StringIO()
        p = 1
        linemap = []
        for line in self._lines:
            if isinstance(line, DeferredLineBase):
                line = line()
                if line is None:
                    continue
            elif isinstance(line, LineContext):
                linemap.append((p, line.context))
                continue
            assert isinstance(line, str)
            buf.write(line)
            buf.write("\n")
            p += 1 + line.count("\n")
        return buf.getvalue(), linemap

    def getvalue(self) -> str:
        v, _ = self.getvaluewithlinemap()
        return v

    def getrawvalue(self) -> str:
        buf = StringIO()
        for line in self._lines:
            if isinstance(line, DeferredLineBase):
                line = line()
                if line is None:
                    continue
            elif isinstance(line, LineContext):
                continue
            assert isinstance(line, str)
            # backslash implies line continuation
            if line.endswith("\\"):
                buf.write(line[:-1])
            else:
                buf.write(line)
                buf.write("\n")
        return buf.getvalue()

    def clear(self):
        self._lines.clear()

    def __bool__(self):
        return bool(self._lines)

    def prefix(self):
        return " " * (self._indent * self._tabwidth)

    def writeline(self, line):
        if isinstance(line, LineContext):
            self._lines.append(line)
        elif isinstance(line, DeferredLineBase):
            self._lines.append(line.with_prefix(self.prefix()))
        elif line.strip():
            self._lines.append(f"{self.prefix()}{line}")
        else:
            self._lines.append("")

    def writelines(self, lines):
        for line in lines:
            self.writeline(line)

    def indent(self, offset=1):
        @contextlib.contextmanager
        def ctx():
            self._indent += offset
            try:
                yield
            finally:
                self._indent -= offset

        return ctx()

    def splice(self, other_code, strip=False):
        if isinstance(other_code, IndentedBuffer):
            dedent = float("inf")
            for line in other_code._lines:
                if not isinstance(line, LineContext) and line:
                    dedent = min(dedent, len(line) - len(line.lstrip()))
            if math.isinf(dedent):
                dedent = 0
            for line in other_code._lines:
                if isinstance(line, LineContext):
                    self._lines.append(line)
                else:
                    IndentedBuffer.writeline(self, line[int(dedent) :])
        else:
            other_code = textwrap.dedent(other_code)
            if strip:
                other_code = other_code.lstrip()
            if not other_code:
                return
            other_code = other_code.rstrip()
            for line in other_code.split("\n"):
                self.writeline(line)


class DeferredLineBase:
    """A line that can be 'unwritten' at a later time"""

    def __init__(self, line):
        if not line.strip():
            line = ""
        self.line = line

    def __call__(self) -> Optional[str]:
        """Returns either self.line or None to indicate the line has been 'unwritten'"""
        raise NotImplementedError()

    def _new_line(self, line: str) -> DeferredLineBase:
        """Returns a new deferred line with the same condition"""
        raise NotImplementedError()

    def with_prefix(self, prefix):
        return self._new_line(f"{prefix}{self.line}")

    def lstrip(self):
        return self._new_line(self.line.lstrip())

    def __getitem__(self, index):
        return self._new_line(self.line[index])

    def __bool__(self):
        return bool(self.line)

    def __len__(self):
        return len(self.line)

class _SafeEncoder(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, (dict, list, tuple, str, float, int, bool, type(None))):
            return o
        elif isinstance(o, torch.Tensor):
            return f'Tensor[%s]' % 'x'.join(list(map(str, o.shape)) + [str(o.dtype)])
        elif isinstance(o, Iterable):
            return list(o)
        elif isinstance(o, compute_graph.Value):
            return o.name
        else:
            result = str(o)
            if len(result) > 20:
                t = type(o)
                return f'{t.__module__}.{t.__qualname__}'
            return str(o)

def safe_json_dumps(data: Any):
    return json.dumps(data, cls=_SafeEncoder)

# https://stackoverflow.com/questions/30458977/yaml-loads-5e-6-as-string-and-not-a-number
YAML_LOADER = yaml.CLoader
YAML_LOADER.add_implicit_resolver(
    u'tag:yaml.org,2002:float',
    re.compile(u'''^(?:
     [-+]?(?:[0-9][0-9_]*)\\.[0-9_]*(?:[eE][-+]?[0-9]+)?
    |[-+]?(?:[0-9][0-9_]*)(?:[eE][-+]?[0-9]+)
    |\\.[0-9_]+(?:[eE][-+][0-9]+)?
    |[-+]?[0-9][0-9_]*(?::[0-5]?[0-9])+\\.[0-9_]*
    |[-+]?\\.(?:inf|Inf|INF)
    |\\.(?:nan|NaN|NAN))$''', re.X),
    list(u'-+0123456789.'))

def load_yaml_file(filename: str):
    with open(filename) as fh:
        return yaml.load(fh, Loader=YAML_LOADER)

def register_dispatch_mode(op):
    from torch.utils._python_dispatch import (
        _get_current_dispatch_mode,
        _pop_mode_temporarily,
    )

    def decorator(cls):
        def inner(*args, **kwargs):
            mode = _get_current_dispatch_mode()
            assert mode is not None, "Mode should always be enabled for python fallback key"
            with _pop_mode_temporarily() as mode:
                return op(*args, **kwargs)

        op.py_impl(cls)(inner)
        return cls

    return decorator

@dataclass
class TimingResult:
    ts:  int
    te:  int = None
    dur: int = None
    name: str = None

@contextlib.contextmanager
def timing(name: str = None):
    result = TimingResult(ts=time.time(), name=name)
    try:
        yield result
    finally:
        result.te = time.time()
        result.dur = result.te - result.ts
