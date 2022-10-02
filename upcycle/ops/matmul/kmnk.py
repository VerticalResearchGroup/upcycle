from dataclasses import dataclass
import logging

from ...common import *
from ..common import *
from .op import *

logger = logging.getLogger(__name__)

@dataclass(frozen=True)
class MatmulTileKMNK(MatmulTile):
    tr_a = True
    tr_b = True
    ttk = None

    def intrinsic(self, mss, nss, kss):
        assert len(mss) <= self.arch.vlen(self.dtype)
        assert len(mss) <= self.tm
        assert len(nss) <= self.tn
        assert len(kss) <= self.tk

        num_loads = \
            M.nloads(self.arch, self.dtype, mss, self.op.m, kss, self.op.k, transpose=True) + \
            M.nloads(self.arch, self.dtype, kss, self.op.k, nss, self.op.n, transpose=True)

        exec_cyc = len(nss) * cld(len(kss), self.ttk)
        return num_loads, exec_cyc

@dataclass(frozen=True)
class MatmulTile256KMNKFP16(MatmulTileKMNK):
    vbits = 256
    dtype = Dtype.FP16
    tm = 8
    tn = 2
    tk = 2
    ttk = 2

@dataclass(frozen=True)
class MatmulTile512KMNKFP16(MatmulTileKMNK):
    vbits = 512
    dtype = Dtype.FP16
    tm = 16
    tn = 2
    tk = 2
    ttk = 2

@dataclass(frozen=True)
class MatmulTile1024KMNKFP16(MatmulTileKMNK):
    vbits = 1024
    dtype = Dtype.FP16
    tm = 32
    tn = 2
    tk = 2
    ttk = 2
