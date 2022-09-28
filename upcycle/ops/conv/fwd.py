from dataclasses import dataclass
import numpy as np
import logging
import functools
import itertools

from ...common import *
from ..common import *
from .. import matmul

from .op import *

@dataclass(frozen=True)
class ConvTile(M.WorkItem):
    write : bool
    ns : Slice
    ss : tuple[Slice]
    cs : Slice
    ks : Slice

    vbits = None
    dtype = None
    tr_w = None
    to = 1
    tp = 1
    tk = None # M
    tq = None # N
    tc = None # K
    ttc = None

    def __post_init__(self):
        assert self.arch.vbits == self.vbits
        assert self.op.dtype == self.dtype
        assert self.op.tr_w == self.tr_w

    @property
    def i(self): return self.inputs[0]

    @property
    def w(self): return self.inputs[1]

    @property
    def o(self): return self.outputs[0]

    @functools.cached_property
    def flops(self):
        return np.prod([len(x) for x in self.ss]) * len(self.ks) * len(self.cs) * np.prod(self.op.filsize) * 2

    @property
    def read_trace(self):
        st = self.op.stride
        if self.op.d == 2:
            yield from self.i[self.ns, self.ss[0] * st, self.ss[1] * st, self.cs]
            if not self.op.tr_w: yield from self.w[:, :, self.ks, self.cs]
            else: yield from self.w[:, :, self.cs, self.ks]
        elif self.op.d == 3:
            yield from self.i[self.ns, self.ss[0] * st, self.ss[1] * st, self.ss[2] * st, self.cs]
            if not self.op.tr_w: yield from self.w[:, :, :, self.ks, self.cs]
            else: yield from self.w[:, :, :, self.cs, self.ks]
        else:
            raise NotImplementedError()

    @functools.cached_property
    def write_trace(self):
        if not self.write: return
        raise NotImplementedError()

    def intrinsic(self, mss, nss, kss):
        m, n, k = self.op.k, self.op.so[-1], self.op.c
        assert len(mss) <= self.arch.vlen(self.dtype) // self.ttc
        assert len(mss) <= self.tk
        assert len(nss) <= self.tq
        assert len(kss) <= self.tc

        num_loads = 0

        # For each tk chunk, we issue a single VLD4T for A
        num_loads += M.nloads(
            self.arch, self.dtype, mss, m, kss, k, transpose=self.tr_w)

        # For MKNK, B is contig. in K, so we just load as many lines
        # as are needed to cover the kslice we are operating on.
        num_loads += M.nloads(
            self.arch, self.dtype, kss, k, nss, n,
            transpose=True, contig=self.op.stride == 1)

        exec_cyc = len(nss) * cld(len(kss), self.ttc)
        return num_loads, exec_cyc

    @functools.cached_property
    def exec_lat(self):
        num_loads = 0
        exec_cyc = 0
        for _ in self.ns.indices:
            for s in itertools.product(*self.ss[:-1]):
                for qss in self.ss[-1].subslice(self.tq): # N
                    for rst in range(self.op.filsize):
                        for kss in self.ks.subslice(self.tk): # M
                            for css in self.cs.subslice(self.tc): # K
                                # logger.debug(f'{s}, {qss}, {rst}, {kss}, {css}')
                                inner_loads, inner_cyc = self.intrinsic(kss, qss, css)
                                num_loads += inner_loads
                                exec_cyc += inner_cyc
                                # logger.debug('Done')

        return max(num_loads / self.arch.l1.rports, exec_cyc)

#
# Int8, KC
#

@dataclass(frozen=True)
class ConvTile256I8KC(ConvTile):
    vbits = 256
    dtype = Dtype.I8
    tr_w = False
    tk = matmul.MatmulTile256MKNKI8_SmallK.tm
    tq = matmul.MatmulTile256MKNKI8_SmallK.tn
    tc = matmul.MatmulTile256MKNKI8_SmallK.tk
    ttc = matmul.MatmulTile256MKNKI8_SmallK.ttk

@dataclass(frozen=True)
class ConvTile512I8KC(ConvTile):
    vbits = 512
    dtype = Dtype.I8
    tr_w = False
    tk = matmul.MatmulTile512MKNKI8_SmallK.tm
    tq = matmul.MatmulTile512MKNKI8_SmallK.tn
    tc = matmul.MatmulTile512MKNKI8_SmallK.tk
    ttc = matmul.MatmulTile512MKNKI8_SmallK.ttk

@dataclass(frozen=True)
class ConvTile1024I8KC(ConvTile):
    vbits = 1024
    dtype = Dtype.I8
    tr_w = False
    tk = matmul.MatmulTile1024MKNKI8_SmallK.tm
    tq = matmul.MatmulTile1024MKNKI8_SmallK.tn
    tc = matmul.MatmulTile1024MKNKI8_SmallK.tk
    ttc = matmul.MatmulTile1024MKNKI8_SmallK.ttk

#
# Float16, KC
#

@dataclass(frozen=True)
class ConvTile256FP16KC(ConvTile):
    vbits = 256
    dtype = Dtype.FP16
    tr_w = False
    tk = matmul.MatmulTile256MKNKFP16_SmallK.tm
    tq = matmul.MatmulTile256MKNKFP16_SmallK.tn
    tc = matmul.MatmulTile256MKNKFP16_SmallK.tk
    ttc = matmul.MatmulTile256MKNKFP16_SmallK.ttk

@dataclass(frozen=True)
class ConvTile512FP16KC(ConvTile):
    vbits = 512
    dtype = Dtype.FP16
    tr_w = False
    tk = matmul.MatmulTile512MKNKFP16_SmallK.tm
    tq = matmul.MatmulTile512MKNKFP16_SmallK.tn
    tc = matmul.MatmulTile512MKNKFP16_SmallK.tk
    ttc = matmul.MatmulTile512MKNKFP16_SmallK.ttk

@dataclass(frozen=True)
class ConvTile1024FP16KC(ConvTile):
    vbits = 1024
    dtype = Dtype.FP16
    tr_w = False
    tk = matmul.MatmulTile1024MKNKFP16_SmallK.tm
    tq = matmul.MatmulTile1024MKNKFP16_SmallK.tn
    tc = matmul.MatmulTile1024MKNKFP16_SmallK.tk
    ttc = matmul.MatmulTile1024MKNKFP16_SmallK.ttk

#
# Float16, CK
#

@dataclass(frozen=True)
class ConvTile256FP16CK(ConvTile):
    vbits = 256
    dtype = Dtype.FP16
    tr_w = False
    tk = matmul.MatmulTile256KMNKFP16.tm
    tq = matmul.MatmulTile256KMNKFP16.tn
    tc = matmul.MatmulTile256KMNKFP16.tk
    ttc = matmul.MatmulTile256KMNKFP16.ttk

@dataclass(frozen=True)
class ConvTile512FP16CK(ConvTile):
    vbits = 512
    dtype = Dtype.FP16
    tr_w = False
    tk = matmul.MatmulTile256KMNKFP16.tm
    tq = matmul.MatmulTile256KMNKFP16.tn
    tc = matmul.MatmulTile256KMNKFP16.tk
    ttc = matmul.MatmulTile256KMNKFP16.ttk

@dataclass(frozen=True)
class ConvTile1024FP16CK(ConvTile):
    vbits = 1024
    dtype = Dtype.FP16
    tr_w = False
    tk = matmul.MatmulTile256KMNKFP16.tm
    tq = matmul.MatmulTile256KMNKFP16.tn
    tc = matmul.MatmulTile256KMNKFP16.tk
    ttc = matmul.MatmulTile256KMNKFP16.ttk
