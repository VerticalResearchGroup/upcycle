from dataclasses import dataclass, field
from enum import IntEnum
import random
import logging

logger = logging.getLogger(__name__)


@dataclass(frozen=True, order=True)
class Slice:
    start : int
    stop : int

    @staticmethod
    def from_pyslice(s : slice, n : int) -> 'Slice':
        return Slice(
            s.start if s.start is not None else 0,
            s.stop if s.stop is not None else n)

    @staticmethod
    def blk(start : int, n : int, blk : int):
        return Slice(start, start + min(n - start, blk))

    def __mul__(self, c : int):
        return Slice(self.start * c, self.stop * c)

    def __add__(self, c : int):
        return Slice(self.start + c, self.stop + c)

    def __len__(self):
        return self.stop - self.start

class Dtype(IntEnum):
    I8 = 1
    FP16 = 2

    @staticmethod
    def sizeof(dt): return int(dt)


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
class OracleArch(Arch): pass

