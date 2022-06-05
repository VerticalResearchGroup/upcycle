from dataclasses import dataclass
from typing import Iterator
import numpy as np
import itertools

from ..common import *
from .. import ops

@dataclass
class TimeSpan:
    ti : int
    tf : int

    def move(self, o : int):
        self.ti += o
        self.tf += o

@dataclass
class WorkBlock:
    issue : TimeSpan
    read : TimeSpan
    exe : TimeSpan

    def __iadd__(self, other):
        other : WorkBlock
        off = other.issue.ti - self.issue.tf
        other.issue.move(off)
        other.read.move(off)
        other.exe.move(off)

@dataclass(frozen=True)
class Tensor:
    oid : int
    dtype : Dtype
    shape : tuple
    strides : tuple = None

    def __post_init__(self):
        object.__setattr__(
            self,
            'strides',
            tuple(
                int(np.prod(self.shape[i+1:]) * Dtype.sizeof(self.dtype))
                for i in range(len(self.shape))))

    def _gen_offs(self, i, d):
        if isinstance(d, int):
            yield d * self.strides[i]
        elif isinstance(d, slice):
            start = 0 if d.start is None else d.start
            stop = self.shape[i] if d.stop is None else d.stop
            yield from map(
                lambda x: x * self.strides[i], range(start, stop, d.step))


    def _gen_ids_rec(self, di, idx):
        d = idx[di]
        if di == len(self.shape) - 1:
            yield from self._gen_offs(di, d)
        else:
            for off in self._gen_offs(di, d):
                yield from map(
                    lambda i: i + off, self._gen_ids_rec(di + 1, idx))

    def __getitem__(self, idx):
        assert len(self.shape) == len(idx)
        upper = self.oid << 32
        last = None

        for off in self._gen_ids_rec(0, idx):
            line = (upper | off) & ~0x3F
            if last is not None and line == last: continue
            last = line
            yield line


@dataclass(frozen=True)
class WorkItem:
    arch : Arch
    dtype : Dtype

    @property
    def exec_lat(self): raise NotImplementedError()

    @property
    def flops(self): raise NotImplementedError()

    @property
    def read_trace(self)  -> Iterator[int]: raise NotImplementedError()

    @property
    def write_trace(self)  -> Iterator[int]: raise NotImplementedError()


@dataclass(frozen=True)
class WorkItemPerfectCompute(WorkItem):
    @property
    def exec_lat(self): return self.flops / self.arch.peak_opc(self.dtype)


@dataclass
class GlobalWorkList:
    tiles : list[list[WorkItem]]

    @staticmethod
    def from_arch(arch : Arch):
        return GlobalWorkList([list() for _ in range(arch.ntiles)])

    @property
    def nsteps(self): return max(map(len, self.tiles))

placement_funcs = {}

def register_placement(mode, archclass, opclass):
    def decorator(x):
        placement_funcs[(mode, archclass, opclass)] = x
        return x
    return decorator

def place_op(mode : str, arch : Arch, op : ops.Operator) -> GlobalWorkList:
    global placement_funcs
    return placement_funcs[(mode, type(arch), type(op))](arch, op)

class Soc:
    def __init__(self, arch : Arch):
        self.arch = arch

    def simulate(self, op : ops.Operator): raise NotImplementedError()

    def noc(self, step : int): raise NotImplementedError()

    @property
    def nsteps(self): raise NotImplementedError()

    @property
    def cycles(self): raise NotImplementedError()

    @property
    def placement_mode(self): return 'naive'


socs = {}

def register_soc(archclass):
    def decorator(xclass):
        global socs
        socs[archclass] = xclass
        return xclass
    return decorator

def make_soc(arch : Arch, *args, **kwargs) -> Soc:
    global socs
    return socs[type(arch)](arch, *args, **kwargs)
