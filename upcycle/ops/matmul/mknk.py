from dataclasses import dataclass
import logging

from ...common import *
from ..common import *
from .op import *

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class MatmulTileMKNK_LargeK(MatmulTile):
    """MKNK matmul with large K

    When K is large and B is contig. in K, we can use traditional SIMD loads and
    FMAs and still achieve high utilization. This allows us to support much
    smaller m and n tiles without hurting performance / utilization.
    """
    tr_a = False
    tr_b = True
    tm = 2
    tn = 2

    def intrinsic(self, mss, nss, kss):
        assert len(kss) <= self.arch.vlen(self.dtype)
        assert len(mss) <= self.tm
        assert len(nss) <= self.tn
        assert len(kss) <= self.tk

        num_loads = 0

        # For each tk chunk, we issue a single VLD4T for A
        num_loads += M.nloads(
            self.arch, self.dtype, mss, self.op.m, kss, self.op.k)

        # For MKNK, B is contig. in K, so we just load as many lines
        # as are needed to cover the kslice we are operating on.
        num_loads += M.nloads(
            self.arch,
            self.dtype,
            kss, self.op.k,
            nss, self.op.n,
            transpose=True)

        exec_cyc = len(mss) * len(nss)
        return num_loads, exec_cyc


@dataclass(frozen=True)
class MatmulTile256MKNKI8_LargeK(MatmulTileMKNK_LargeK):
    vbits = 256
    dtype = Dtype.I8
    tk = 32

@dataclass(frozen=True)
class MatmulTile512MKNKI8_LargeK(MatmulTileMKNK_LargeK):
    vbits = 512
    dtype = Dtype.I8
    tk = 64

@dataclass(frozen=True)
class MatmulTile1024MKNKI8_LargeK(MatmulTileMKNK_LargeK):
    vbits = 1024
    dtype = Dtype.I8
    tk = 128

@dataclass(frozen=True)
class MatmulTile256MKNKFP16_LargeK(MatmulTileMKNK_LargeK):
    vbits = 256
    dtype = Dtype.FP16
    tk = 16

@dataclass(frozen=True)
class MatmulTile512MKNKFP16_LargeK(MatmulTileMKNK_LargeK):
    vbits = 512
    dtype = Dtype.FP16
    tk = 32

@dataclass(frozen=True)
class MatmulTile1024MKNKFP16_LargeK(MatmulTileMKNK_LargeK):
    vbits = 1024
    dtype = Dtype.FP16
    tk = 64

@dataclass(frozen=True)
class MatmulTileMKNK_SmallK(MatmulTile):
    tr_a = False
    tr_b = True
    ttk = None

    def intrinsic(self, mss, nss, kss):
        assert len(mss) <= self.arch.vlen(self.dtype) // self.ttk
        assert len(mss) <= self.tm
        assert len(nss) <= self.tn
        assert len(kss) <= self.tk

        num_loads = 0

        # For each tk chunk, we issue a single VLD4T for A
        num_loads += M.nloads(
            self.arch, self.dtype, mss, self.op.m, kss, self.op.k)

        # For MKNK, B is contig. in K, so we just load as many lines
        # as are needed to cover the kslice we are operating on.
        num_loads += M.nloads(
            self.arch,
            self.dtype,
            kss, self.op.k,
            nss, self.op.n,
            transpose=True)

        exec_cyc = len(nss) * cld(len(kss), self.ttk)
        return num_loads, exec_cyc

@dataclass(frozen=True)
class MatmulTile256MKNKI8_SmallK(MatmulTileMKNK_SmallK):
    vbits = 256
    dtype = Dtype.I8
    tm = 8
    tn = 4
    tk = 32
    ttk = 4

@dataclass(frozen=True)
class MatmulTile512MKNKI8_SmallK(MatmulTileMKNK_SmallK):
    vbits = 512
    dtype = Dtype.I8
    tm = 16
    tn = 4
    tk = 64
    ttk = 4

@dataclass(frozen=True)
class MatmulTile1024MKNKI8_SmallK(MatmulTileMKNK_SmallK):
    vbits = 1024
    dtype = Dtype.I8
    tm = 32
    tn = 4
    tk = 128
    ttk = 4

@dataclass(frozen=True)
class MatmulTile256MKNKFP16_SmallK(MatmulTileMKNK_SmallK):
    vbits = 256
    dtype = Dtype.I8
    tm = 8
    tn = 4
    tk = 16
    ttk = 2

@dataclass(frozen=True)
class MatmulTile512MKNKFP16_SmallK(MatmulTileMKNK_SmallK):
    vbits = 512
    dtype = Dtype.I8
    tm = 16
    tn = 4
    tk = 32
    ttk = 2

@dataclass(frozen=True)
class MatmulTile1024MKNKFP16_SmallK(MatmulTileMKNK_SmallK):
    vbits = 1024
    dtype = Dtype.I8
    tm = 32
    tn = 4
    tk = 64
    ttk = 2
