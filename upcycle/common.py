from dataclasses import dataclass, field
from enum import IntEnum
import random


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

    @property
    def ntiles(self): raise NotImplementedError()

    def vlen(self, dtype : Dtype): return self.vbits / 8 / Dtype.sizeof(dtype)

    def peak_opc(self, dtype : Dtype):
        return self.ntiles * self.vlen(dtype) * self.macs * 2

@dataclass(order=True, frozen=True)
class FlatMeshArch(Arch):
    nrow : int
    ncol : int

    @property
    def ntiles(self): return self.nrow * self.ncol
