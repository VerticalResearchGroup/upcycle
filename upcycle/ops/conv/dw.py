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

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ConvDwTile(M.WorkItem):
    write : bool

    ns : Slice
    os : tuple[Slice]
    fs : tuple[Slice]

    ks : Slice
    cs : Slice

    vbits = None
    dtype = None
    tr_w = False
    tn = 1
    to = 1
    tp = 1
    tk = None # M
    tq = None # K
    tc = None # N
    ttq = None

    def __post_init__(self):
        assert self.arch.vbits == self.vbits
        assert self.op.dtype == self.dtype
        assert self.op.tr_w == self.tr_w

    @property
    def i(self): return self.inputs[1]

    @property
    def do(self): return self.inputs[0]

    @property
    def dw(self): return self.outputs[0]

    @property
    def read_trace(self):
        st = self.op.stride
        xs = tuple(map(lambda s: s * st, self.os))

        if self.op.d == 2:
            yield from self.i[self.ns, xs[0], xs[1], self.cs]
            yield from self.do[self.ns, self.os[0], self.os[1], self.ks]
        elif self.op.d == 3:
            yield from self.i[self.ns, xs[0], xs[1], xs[2], self.cs]
            yield from self.do[self.ns, self.os[0], self.os[1], self.os[2], self.ks]
        else:
            raise NotImplementedError()

    @property
    def write_trace(self):
        if not self.write: return
        raise NotImplementedError()

    def intrinsic(self, mss, nss, kss):
        m, n, k = self.op.c, self.op.so[-1], self.op.k
        assert len(mss) <= self.tc
        assert len(nss) <= self.tq
        assert len(kss) <= self.tk

        num_loads = M.nloads(
            self.arch, self.dtype, mss, m, kss, k, transpose=not self.tr_w)

        num_loads += M.nloads(
            self.arch, self.dtype, kss, k, nss, n,
            transpose=False, contig=self.op.stride == 1)

        exec_cyc = len(nss) * cld(len(kss), self.ttq)
        return num_loads, exec_cyc

    def _small_gemm(self, n):
        ns = Slice(0, n)

        num_loads = 0
        exec_cyc = 0
        for kss in self.ks.subslice(self.tk):
            for css in self.cs.subslice(self.tc):
                for qss in ns.subslice(self.tq):
                    inner_loads, inner_cyc = self.intrinsic(css, qss, kss)
                    num_loads += inner_loads
                    exec_cyc += inner_cyc

        return num_loads, exec_cyc

    @functools.cached_property
    def _gemms(self):
        gemms = []

        for sf in itertools.product(*self.fs):
            for ni in self.ns.indices:
                for s in itertools.product(*self.os[:-1]):
                    gemms.append(len(self.os[-1]))

        return gemms

    @functools.cached_property
    def flops(self): return sum(self._gemms) * len(self.cs) * len(self.ks) * 2

    @functools.cached_property
    def exec_lat(self):
        num_loads = 0
        exec_cyc = 0
        for n in self._gemms:
            _num_loads, _exec_cyc = self._small_gemm(n)
            num_loads += _num_loads
            exec_cyc += _exec_cyc
        lat = max(num_loads / self.arch.l1.rports, exec_cyc)
        return lat

@dataclass(frozen=True)
class ConvDwTile256FP16KC(ConvDwTile):
    vbits = 256
    dtype = Dtype.FP16
    tc = matmul.MatmulTile256KMKNFP16.tm # M
    tk = matmul.MatmulTile256KMKNFP16.tk # N
    tq = matmul.MatmulTile256KMKNFP16.tn # K
    ttq = matmul.MatmulTile256KMKNFP16.ttk

@dataclass(frozen=True)
class ConvDwTile512FP16KC(ConvDwTile):
    vbits = 512
    dtype = Dtype.FP16
    tc = matmul.MatmulTile512KMKNFP16.tm # M
    tk = matmul.MatmulTile512KMKNFP16.tk # N
    tq = matmul.MatmulTile512KMKNFP16.tn # K
    ttq = matmul.MatmulTile512KMKNFP16.ttk

@dataclass(frozen=True)
class ConvDwTile1024FP16KC(ConvDwTile):
    vbits = 1024
    dtype = Dtype.FP16
    tc = matmul.MatmulTile1024KMKNFP16.tm # M
    tk = matmul.MatmulTile1024KMKNFP16.tk # N
    tq = matmul.MatmulTile1024KMKNFP16.tn # K
    ttq = matmul.MatmulTile1024KMKNFP16.ttk
