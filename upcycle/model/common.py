from dataclasses import dataclass

from ..common import *
from .. import ops
from .. import apps

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
class AffineTile:
    oid : int
    addr : int
    stride : int
    w : int
    h : int

    @property
    def lines(self):
        upper = self.oid << 32
        for hi in range(self.h):
            base = (self.addr + self.stride * hi) & ~0x3F
            for line in range(base, base + self.w, 64):
                yield upper | line

class WorkItem:
    @property
    def exec_lat(self): raise NotImplementedError()

    @property
    def flops(self): raise NotImplementedError()

    @property
    def read_trace(self): raise NotImplementedError()

@dataclass
class GlobalWorkList:
    tiles : list[list[WorkItem]]

    @staticmethod
    def from_arch(arch : Arch):
        return GlobalWorkList([list() for _ in range(arch.ntiles)])

placement_funcs = {}

def register_placement(mode, opclass):
    def decorator(x):
        placement_funcs[(mode, opclass)] = x
        return x
    return decorator

def place_op(mode : str, arch : Arch, op : ops.Operator) -> GlobalWorkList:
    global placement_funcs
    return placement_funcs[(mode, type(op))](arch, op)

