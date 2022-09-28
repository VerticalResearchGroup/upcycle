from dataclasses import dataclass
import logging

from ...common import *
from ..common import *
from .op import *

logger = logging.getLogger(__name__)

@dataclass(frozen=True)
class MatmulTileMKKN(MatmulTile):
    tr_a = False
    tr_b = False
    ttk = None

    def intrinsic(self, mss, nss, kss):
        assert len(mss) <= self.arch.vlen(self.dtype) // self.ttk
        assert len(mss) <= self.tm
        assert len(nss) <= self.tn
        assert len(kss) <= self.tk

        num_loads = \
            M.nloads(self.arch, self.dtype, mss, self.op.m, kss, self.op.k) + \
            sum(M.nloads(self.arch, self.dtype, ksss, self.op.k, nss, self.op.n)
                for ksss in kss.subslice(self.ttk))

        exec_cyc = len(nss) * cld(len(kss), self.ttk)
        return num_loads, exec_cyc

@dataclass(frozen=True)
class MatmulTile256MKKNI8(MatmulTileMKKN):
    vbits = 256
    dtype = Dtype.I8
    tm = 8
    tn = 2
    tk = 32
    ttk = 4

@dataclass(frozen=True)
class MatmulTile512MKKNI8(MatmulTileMKKN):
    vbits = 512
    dtype = Dtype.I8
    tm = 16
    tn = 2
    tk = 64
    ttk = 4

@dataclass(frozen=True)
class MatmulTile1024MKKNI8(MatmulTileMKKN):
    vbits = 1024
    dtype = Dtype.I8
    tm = 32
    tn = 2
    tk = 128
    ttk = 4

@dataclass(frozen=True)
class MatmulTile256MKKNFP16(MatmulTileMKKN):
    vbits = 256
    dtype = Dtype.FP16
    tm = 8
    tn = 2
    tk = 16
    ttk = 2

@dataclass(frozen=True)
class MatmulTile512MKKNFP16(MatmulTileMKKN):
    vbits = 512
    dtype = Dtype.FP16
    tm = 16
    tn = 2
    tk = 32
    ttk = 2

@dataclass(frozen=True)
class MatmulTile1024MKKNFP16(MatmulTileMKKN):
    vbits = 1024
    dtype = Dtype.FP16
    tm = 32
    tn = 2
    tk = 64
    ttk = 2
