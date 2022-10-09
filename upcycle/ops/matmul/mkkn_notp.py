from dataclasses import dataclass
import logging

from ...common import *
from ..common import *
from .op import *

logger = logging.getLogger(__name__)

@dataclass(frozen=True)
class MatmulTileMKKN_NT(MatmulTile):
    tr_a = False
    tr_b = False

    def intrinsic(self, mss, nss, kss):
        assert len(mss) <= self.tm
        assert len(nss) <= self.tn
        assert len(kss) <= self.tk
        num_loads = len(mss) + len(nss) * len(kss)
        exec_cyc = len(mss) * len(nss)
        return num_loads, exec_cyc

@dataclass(frozen=True)
class MatmulTile256MKKNI8_NT(MatmulTileMKKN_NT):
    vbits = 256
    dtype = Dtype.I8
    tm = 32
    tn = 1
    tk = 32

@dataclass(frozen=True)
class MatmulTile512MKKNI8_NT(MatmulTileMKKN_NT):
    vbits = 512
    dtype = Dtype.I8
    tm = 64
    tn = 1
    tk = 64

@dataclass(frozen=True)
class MatmulTile1024MKKNI8_NT(MatmulTileMKKN_NT):
    vbits = 1024
    dtype = Dtype.I8
    tm = 128
    tn = 1
    tk = 128

@dataclass(frozen=True)
class MatmulTile256MKKNFP16_NT(MatmulTileMKKN_NT):
    vbits = 256
    dtype = Dtype.FP16
    tm = 16
    tn = 1
    tk = 16

@dataclass(frozen=True)
class MatmulTile512MKKNFP16_NT(MatmulTileMKKN_NT):
    vbits = 512
    dtype = Dtype.FP16
    tm = 32
    tn = 1
    tk = 32

@dataclass(frozen=True)
class MatmulTile1024MKKNFP16_NT(MatmulTileMKKN_NT):
    vbits = 1024
    dtype = Dtype.FP16
    tm = 64
    tn = 1
    tk = 64
