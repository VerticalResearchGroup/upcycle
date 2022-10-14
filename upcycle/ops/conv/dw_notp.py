from dataclasses import dataclass
import numpy as np
import logging
import functools

from ...common import *
from ..common import *
from .. import matmul

from .op import *

from .. import matmul
from ..reduce import Reduce

from .dw import *

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ConvDwTile_NT(ConvDwTile):
    def intrinsic(self, mss, nss, kss):
        assert len(mss) <= self.arch.vlen(self.dtype)
        assert len(mss) <= self.tc
        assert len(nss) <= self.tq
        assert len(kss) <= self.tk

        return self.gemm(
            self.arch, None, None, None, False, 0, None, None, None) \
            .intrinsic(mss, nss, kss)

@dataclass(frozen=True)
class ConvDwTile256FP16KC_NT(ConvDwTile_NT):
    gemm = matmul.MatmulTile256KMKNFP16_NT
    vbits = 256
    dtype = Dtype.FP16
    tc = matmul.MatmulTile256KMKNFP16_NT.tm # M
    tk = matmul.MatmulTile256KMKNFP16_NT.tk # N
    tq = matmul.MatmulTile256KMKNFP16_NT.tn # K

@dataclass(frozen=True)
class ConvDwTile512FP16KC_NT(ConvDwTile_NT):
    gemm = matmul.MatmulTile512KMKNFP16_NT
    vbits = 512
    dtype = Dtype.FP16
    tc = matmul.MatmulTile512KMKNFP16_NT.tm # M
    tk = matmul.MatmulTile512KMKNFP16_NT.tk # N
    tq = matmul.MatmulTile512KMKNFP16_NT.tn # K

@dataclass(frozen=True)
class ConvDwTile1024FP16KC_NT(ConvDwTile_NT):
    gemm = matmul.MatmulTile1024KMKNFP16_NT
    vbits = 1024
    dtype = Dtype.FP16
    tc = matmul.MatmulTile1024KMKNFP16_NT.tm # M
    tk = matmul.MatmulTile1024KMKNFP16_NT.tk # N
    tq = matmul.MatmulTile1024KMKNFP16_NT.tn # K
