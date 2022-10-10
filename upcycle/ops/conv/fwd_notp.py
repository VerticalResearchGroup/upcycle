from dataclasses import dataclass
import numpy as np
import logging
import functools
import itertools

from ...common import *
from ..common import *
from .. import matmul

from .op import *
from .fwd import *

@dataclass(frozen=True)
class ConvTile_NT(ConvTile):
    def intrinsic(self, mss, nss, kss):
        assert len(mss) <= self.arch.vlen(self.dtype)
        assert len(mss) <= self.tk
        assert len(nss) <= self.tq
        assert len(kss) <= self.tc

        return self.gemm(
            self.arch, None, None, None, False, 0, None, None, None) \
            .intrinsic(mss, nss, kss)

#
# Int8, KC
#

@dataclass(frozen=True)
class ConvTile256I8KC_NT(ConvTile_NT):
    gemm = matmul.MatmulTile256MKNKI8_NT
    vbits = 256
    dtype = Dtype.I8
    tr_w = False
    tk = matmul.MatmulTile256MKNKI8_NT.tm
    tq = matmul.MatmulTile256MKNKI8_NT.tn
    tc = matmul.MatmulTile256MKNKI8_NT.tk

@dataclass(frozen=True)
class ConvTile512I8KC_NT(ConvTile_NT):
    gemm = matmul.MatmulTile512MKNKI8_NT
    vbits = 512
    dtype = Dtype.I8
    tr_w = False
    tk = matmul.MatmulTile512MKNKI8_NT.tm
    tq = matmul.MatmulTile512MKNKI8_NT.tn
    tc = matmul.MatmulTile512MKNKI8_NT.tk

@dataclass(frozen=True)
class ConvTile1024I8KC_NT(ConvTile_NT):
    gemm = matmul.MatmulTile1024MKNKI8_NT
    vbits = 1024
    dtype = Dtype.I8
    tr_w = False
    tk = matmul.MatmulTile1024MKNKI8_NT.tm
    tq = matmul.MatmulTile1024MKNKI8_NT.tn
    tc = matmul.MatmulTile1024MKNKI8_NT.tk

#
# Float16, KC
#

@dataclass(frozen=True)
class ConvTile256FP16KC_NT(ConvTile_NT):
    gemm = matmul.MatmulTile256MKNKFP16_NT
    vbits = 256
    dtype = Dtype.FP16
    tr_w = False
    tk = matmul.MatmulTile256MKNKFP16_NT.tm
    tq = matmul.MatmulTile256MKNKFP16_NT.tn
    tc = matmul.MatmulTile256MKNKFP16_NT.tk

@dataclass(frozen=True)
class ConvTile512FP16KC_NT(ConvTile_NT):
    gemm = matmul.MatmulTile512MKNKFP16_NT
    vbits = 512
    dtype = Dtype.FP16
    tr_w = False
    tk = matmul.MatmulTile512MKNKFP16_NT.tm
    tq = matmul.MatmulTile512MKNKFP16_NT.tn
    tc = matmul.MatmulTile512MKNKFP16_NT.tk

@dataclass(frozen=True)
class ConvTile1024FP16KC_NT(ConvTile_NT):
    gemm = matmul.MatmulTile1024MKNKFP16_NT
    vbits = 1024
    dtype = Dtype.FP16
    tr_w = False
    tk = matmul.MatmulTile1024MKNKFP16_NT.tm
    tq = matmul.MatmulTile1024MKNKFP16_NT.tn
    tc = matmul.MatmulTile1024MKNKFP16_NT.tk

#
# Float16, CK
#

@dataclass(frozen=True)
class ConvTileCK_NT(ConvTile):
    def intrinsic(self, mss, nss, kss):
        assert len(mss) <= self.arch.vlen(self.dtype)
        assert len(mss) <= self.tk
        assert len(nss) <= self.tq
        assert len(kss) <= self.tc

        return self.gemm(
            self.arch, None, None, None, False, 0, None, None, None) \
            .intrinsic(mss, nss, kss)

@dataclass(frozen=True)
class ConvTile256FP16CK_NT(ConvTileCK_NT):
    gemm = matmul.MatmulTile256KMNKFP16_NT
    vbits = 256
    dtype = Dtype.FP16
    tr_w = True
    tk = matmul.MatmulTile256KMNKFP16_NT.tm
    tq = matmul.MatmulTile256KMNKFP16_NT.tn
    tc = matmul.MatmulTile256KMNKFP16_NT.tk

@dataclass(frozen=True)
class ConvTile512FP16CK_NT(ConvTileCK_NT):
    gemm = matmul.MatmulTile512KMNKFP16_NT
    vbits = 512
    dtype = Dtype.FP16
    tr_w = True
    tk = matmul.MatmulTile512KMNKFP16_NT.tm
    tq = matmul.MatmulTile512KMNKFP16_NT.tn
    tc = matmul.MatmulTile512KMNKFP16_NT.tk

@dataclass(frozen=True)
class ConvTile1024FP16CK_NT(ConvTileCK_NT):
    gemm = matmul.MatmulTile1024KMNKFP16_NT
    vbits = 1024
    dtype = Dtype.FP16
    tr_w = True
    tk = matmul.MatmulTile1024KMNKFP16_NT.tm
    tq = matmul.MatmulTile1024KMNKFP16_NT.tn
    tc = matmul.MatmulTile1024KMNKFP16_NT.tk
