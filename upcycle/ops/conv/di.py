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


logger = logging.getLogger(__name__)

@dataclass(frozen=True)
class ConvDiTile(M.WorkItem):
    write : bool
    ns : Slice
    ss : tuple[Slice]
    ks : Slice
    cs : Slice

    vbits = None
    dtype = None
    tr_w = False
    to = 1
    tp = 1
    tk = None # K
    tn = None # N
    tc = None # M
    ttk = None

    def __post_init__(self):
        assert self.vbits is None or self.arch.vbits == self.vbits
        assert self.dtype is None or self.op.dtype == self.dtype
        assert self.tr_w is None or self.op.tr_w == self.tr_w

    @property
    def do(self): return self.inputs[1]

    @property
    def w(self): return self.inputs[0]

    @property
    def di(self): return self.outputs[0]

    @property
    def read_trace(self):
        pad = self.op.pad
        st = self.op.stride
        os = tuple(map(lambda s: Slice((s.start + pad) // st, (s.stop + pad) // st), self.ss))
        if self.op.d == 2:
            yield from self.do[self.ns, os[0], os[1], self.ks]
            yield from self.w[:, :, self.ks, self.cs]
        elif self.op.d == 3:
            yield from self.do[self.ns, os[0], os[1], os[2], self.ks]
            yield from self.w[:, :, :, self.ks, self.cs]
        else:
            raise NotImplementedError()

    @property
    def write_trace(self):
        if not self.write: return
        raise NotImplementedError()

    def intrinsic(self, mss, nss, kss):
        m, n, k = self.op.c, self.op.so[-1], self.op.k
        assert len(mss) <= self.tc
        assert len(nss) <= self.tn
        assert len(kss) <= self.tk

        num_loads = M.nloads(
            self.arch, self.dtype, mss, m, kss, k, transpose=not self.tr_w)

        num_loads += M.nloads(
            self.arch, self.dtype, kss, k, nss, n,
            transpose=True, contig=self.op.stride == 1)

        exec_cyc = len(nss) * cld(len(kss), self.ttk)
        return num_loads, exec_cyc

    def _small_gemm(self, n):
        ns = Slice(0, n)

        num_loads = 0
        exec_cyc = 0
        for css in self.cs.subslice(self.tc):
            for nss in ns.subslice(self.tn):
                for kss in self.ks.subslice(self.tk):
                    inner_loads, inner_cyc = self.intrinsic(css, nss, kss)
                    num_loads += inner_loads
                    exec_cyc += inner_cyc

        return num_loads, exec_cyc

    @functools.cached_property
    def _gemms(self):
        st = self.op.stride
        gemms = [0 for _ in range(np.prod([len(s) for s in self.ss]))]
        i = 0
        pad = self.op.pad
        si = np.array(self.op.si)

        # N.B. I wrote this down as a diagram first but basically what we need
        # to do is for each pixel of dI we are computing, find the corresponding
        # _set_ of filter pixels it is multiplied with.
        for sx in itertools.product(*self.ss):
            sx = np.array(sx)
            if (sx < pad).any() or (sx >= (si + pad)).any(): continue
            offx = sx % st

            n = 0
            for sf in itertools.product(*[range(offx[i], self.op.sf[i], st) for i in range(self.op.d)]):
                sf = np.array(sf)
                sy = (sx - sf) // st
                if (sy < 0).any() or (sy >= self.op.so).any(): continue
                n += 1

            gemms[i] = n * len(self.ns)
            i += 1

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
class ConvDiTile256FP16KC(ConvDiTile):
    vbits = 256
    dtype = Dtype.FP16
    tc = matmul.MatmulTile256KMNKFP16.tm # M
    tk = matmul.MatmulTile256KMNKFP16.tk # K
    ttk = matmul.MatmulTile256KMNKFP16.ttk
    tn = matmul.MatmulTile256KMNKFP16.tn # N

@dataclass(frozen=True)
class ConvDiTile512FP16KC(ConvDiTile):
    vbits = 512
    dtype = Dtype.FP16
    tc = matmul.MatmulTile512KMNKFP16.tm # M
    tk = matmul.MatmulTile512KMNKFP16.tk # K
    ttk = matmul.MatmulTile512KMNKFP16.ttk
    tn = matmul.MatmulTile512KMNKFP16.tn # N

@dataclass(frozen=True)
class ConvDiTile1024FP16KC(ConvDiTile):
    vbits = 1024
    dtype = Dtype.FP16
    tc = matmul.MatmulTile1024KMNKFP16.tm # M
    tk = matmul.MatmulTile1024KMNKFP16.tk # K
    ttk = matmul.MatmulTile1024KMNKFP16.ttk
    tn = matmul.MatmulTile1024KMNKFP16.tn # N
