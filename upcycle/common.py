from dataclasses import dataclass, field
from enum import IntEnum
import numpy as np
import random
import logging

logger = logging.getLogger(__name__)


@dataclass(frozen=True, order=True)
class Slice:
    start : int
    stop : int
    step : int = 1

    @staticmethod
    def from_pyslice(s : slice, n : int) -> 'Slice':
        return Slice(
            s.start if s.start is not None else 0,
            s.stop if s.stop is not None else n,
            s.step if s.step is not None else 1)

    @staticmethod
    def blk(start : int, n : int, blk : int):
        return Slice(start, start + min(n - start, blk), 1)

    def __mul__(self, c : int):
        return Slice(self.start * c, self.stop * c, self.step)

    def __add__(self, c : int):
        return Slice(self.start + c, self.stop + c, self.step)

    def _div(self, c : int):
        return Slice(self.start // c, int(np.ceil(self.stop / c)), self.step)

    def __div__(self, c : int): return self._div(c)
    def __truediv__(self, c : int): return self._div(c)
    def __floordiv__(self, c : int): return self._div(c)
    def __contains__(self, i : int): return self.start <= i < self.stop

    def __len__(self):
        return (self.stop - self.start) // self.step

    @property
    def indices(self): yield from range(self.start, self.stop, self.step)

class Dtype(IntEnum):
    I8 = 1
    FP16 = 2

    @staticmethod
    def sizeof(dt): return int(dt)

    @staticmethod
    def from_str(s):
        if s == 'I8': return Dtype.I8
        if s == 'FP16': return Dtype.FP16
        raise ValueError(f'Invalid dtype: {s}')


@dataclass(order=True, frozen=True)
class Arch:
    freq : float
    vbits : int
    macs : int
    nrows : int
    ncols : int
    noc_ports_per_dir : int = 1

    def __post_init__(self):
        if self.noc_ports_per_dir > 1:
            logger.warn(f'Arch has {self.noc_ports_per_dir} ports per direction (>1)')

    @property
    def ntiles(self): return self.nrows * self.ncols

    def vlen(self, dtype : Dtype): return self.vbits / 8 / Dtype.sizeof(dtype)

    def peak_opc(self, dtype : Dtype):
        return self.vlen(dtype) * self.macs * 2


@dataclass(frozen=True)
class Operator:
    dtype : Dtype
    train : bool

    @property
    def flops(self): raise NotImplementedError()

@dataclass(order=True, frozen=True)
class FlatMeshArch(Arch): pass

@dataclass(order=True, frozen=True)
class BgroupArch(Arch):
    grows : int = 4
    gcols : int = 8


@dataclass(order=True, frozen=True)
class OracleArch(Arch): pass

