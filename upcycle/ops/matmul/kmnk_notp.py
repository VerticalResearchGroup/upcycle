from dataclasses import dataclass
import logging

from ...common import *
from ..common import *
from .op import *

logger = logging.getLogger(__name__)

@dataclass(frozen=True)
class MatmulTileKMNK_NT(MatmulTile):
    tr_a = True
    tr_b = True

    def intrinsic(self, mss, nss, kss):
        assert len(mss) <= self.tm
        assert len(nss) <= self.tn
        assert len(kss) <= self.tk
        num_loads = len(kss) + len(nss) * len(kss)
        exec_cyc = cld(len(mss), 32) * len(nss) * len(kss)
        return num_loads, exec_cyc

@dataclass(frozen=True)
class MatmulTile256KMNKFP16_NT(MatmulTileKMNK_NT):
    vbits = 256
    dtype = Dtype.FP16
    tm = 16
    tn = 2
    tk = 2

@dataclass(frozen=True)
class MatmulTile512KMNKFP16_NT(MatmulTileKMNK_NT):
    vbits = 512
    dtype = Dtype.FP16
    tm = 32
    tn = 2
    tk = 2

@dataclass(frozen=True)
class MatmulTile1024KMNKFP16_NT(MatmulTileKMNK_NT):
    vbits = 1024
    dtype = Dtype.FP16
    tm = 64
    tn = 2
    tk = 2
