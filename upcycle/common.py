from dataclasses import dataclass, field
from enum import IntEnum
import random
import logging

logger = logging.getLogger(__name__)

def slice_mul(s, c):
    start = None if s.start is None else s.start * c
    stop = None if s.stop is None else s.stop * c
    return slice(start, stop, None)

def slice_add(s, c):
    start = None if s.start is None else s.start + c
    stop = None if s.stop is None else s.stop + c
    return slice(start, stop, None)

def slice_len(s, n):
    start = 0 if s.start is None else s.start
    stop = n if s.stop is None else s.stop
    return stop - start

def slice_blk(start, n, blk):
    return slice(start, start + min(n - start, blk))

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

