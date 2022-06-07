from dataclasses import dataclass, field
from enum import IntEnum
import random

def slice_mul(s, c):
    start = None if s.start is None else s.start * c
    stop = None if s.stop is None else s.stop * c
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

