from dataclasses import dataclass
import numpy as np
import logging
import functools
import itertools

from ...common import *
from ..common import *
from .. import matmul

from .op import *

from .. import matmul
from ..reduce import Reduce

from .di import *

logger = logging.getLogger(__name__)

@dataclass(frozen=True)
class ConvDiTile_NT(ConvDiTile):
    def intrinsic(self, mss, nss, kss):
        assert len(mss) <= self.arch.vlen(self.dtype)
        assert len(mss) <= self.tk
        assert len(nss) <= self.tq
        assert len(kss) <= self.tc

        return self.gemm(
            self.arch, None, None, None, False, 0, None, None, None) \
            .intrinsic(mss, nss, kss)

@dataclass(frozen=True)
class ConvDiTile256FP16KC_NT(ConvDiTile_NT):
    gemm = matmul.MatmulTile256KMNKFP16_NT
    vbits = 256
    dtype = Dtype.FP16
    tc = matmul.MatmulTile256KMNKFP16_NT.tm # M
    tk = matmul.MatmulTile256KMNKFP16_NT.tk # K
    tn = matmul.MatmulTile256KMNKFP16_NT.tn # N

@dataclass(frozen=True)
class ConvDiTile512FP16KC_NT(ConvDiTile_NT):
    gemm = matmul.MatmulTile512KMNKFP16_NT
    vbits = 512
    dtype = Dtype.FP16
    tc = matmul.MatmulTile512KMNKFP16_NT.tm # M
    tk = matmul.MatmulTile512KMNKFP16_NT.tk # K
    tn = matmul.MatmulTile512KMNKFP16_NT.tn # N

@dataclass(frozen=True)
class ConvDiTile1024FP16KC_NT(ConvDiTile_NT):
    gemm = matmul.MatmulTile1024KMNKFP16_NT
    vbits = 1024
    dtype = Dtype.FP16
    tc = matmul.MatmulTile1024KMNKFP16_NT.tm # M
    tk = matmul.MatmulTile1024KMNKFP16_NT.tk # K
    tn = matmul.MatmulTile1024KMNKFP16_NT.tn # N
