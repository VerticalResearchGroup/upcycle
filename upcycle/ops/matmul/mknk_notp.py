from dataclasses import dataclass
import logging

from ...common import *
from ..common import *
from .op import *

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class MatmulTileMKNK_NT(MatmulTile):
    tr_a = False
    tr_b = True

    def intrinsic(self, mss, nss, kss):
        assert len(mss) <= self.tm
        assert len(nss) <= self.tn
        assert len(kss) <= self.tk
        num_loads = len(mss) + len(nss)
        exec_cyc = len(mss) * len(nss)
        return num_loads, exec_cyc

@dataclass(frozen=True)
class MatmulTile256MKNKI8_NT(MatmulTileMKNK_NT):
    vbits = 256
    dtype = Dtype.I8
    tm = 2
    tn = 2
    tk = 32

@dataclass(frozen=True)
class MatmulTile512MKNKI8_NT(MatmulTileMKNK_NT):
    vbits = 512
    dtype = Dtype.I8
    tm = 2
    tn = 2
    tk = 64

@dataclass(frozen=True)
class MatmulTile1024MKNKI8_NT(MatmulTileMKNK_NT):
    vbits = 1024
    dtype = Dtype.I8
    tm = 2
    tn = 2
    tk = 128

@dataclass(frozen=True)
class MatmulTile256MKNKFP16_NT(MatmulTileMKNK_NT):
    vbits = 256
    dtype = Dtype.FP16
    tm = 2
    tn = 2
    tk = 16

@dataclass(frozen=True)
class MatmulTile512MKNKFP16_NT(MatmulTileMKNK_NT):
    vbits = 512
    dtype = Dtype.FP16
    tm = 2
    tn = 2
    tk = 32

@dataclass(frozen=True)
class MatmulTile1024MKNKFP16_NT(MatmulTileMKNK_NT):
    vbits = 1024
    dtype = Dtype.FP16
    tm = 2
    tn = 2
    tk = 64
