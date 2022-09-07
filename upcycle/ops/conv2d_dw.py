from dataclasses import dataclass
import numpy as np
import logging
import functools

from ..common import *
from .common import *

from . import matmul
from .conv2d import Conv2D, make_conv2d_tensors

logger = logging.getLogger(__name__)


@operator
@dataclass(frozen=True)
@register_backward(Conv2D)
class Conv2DDw(Conv2D):
    @staticmethod
    def from_forward(c : Conv2D):
        return Conv2DDw(c.dtype, False, c.n, c.h, c.w, c.c, c.p, c.q, c.k, c.r, c.s, c.stride, c.pad, c.tr_w)

@dataclass(frozen=True)
class Conv2DDwTile(M.WorkItem):
    write : bool

    ns : Slice
    ps : Slice
    qs : Slice

    rs : Slice
    ss : Slice
    ks : Slice
    cs : Slice

    tn = 1
    tp = 8
    tq = 16
    tc = 32
    tk = 4

    @property
    def i(self): return self.inputs[0]

    @property
    def do(self): return self.inputs[1]

    @property
    def dw(self): return self.outputs[0]

    @functools.cached_property
    def hs(self): return self.ps * self.op.stride

    @functools.cached_property
    def ws(self): return self.qs * self.op.stride

    @property
    def read_trace(self):
        yield from self.i[self.ns, self.hs, self.ws, self.cs]
        yield from self.do[self.ns, self.ps, self.qs, self.ks]

    @property
    def write_trace(self):
        if not self.write: return
        raise NotImplementedError()

    def _small_gemm(self, n):
        ns = Slice(0, n)
        # Each small gemm is basically a CxNxK matmul with KMKN layout. This
        # code should look similar to the FP16 matmul code.

        num_loads = 0
        exec_cyc = 0
        for css in self.cs.subslice(self.tc):
            for nss in ns.subslice(1):
                num_loads += M.nloads(
                    self.arch,
                    Dtype.FP16,
                    css, self.op.c,
                    nss, n,
                    transpose=True,
                    contig=False)

                for kss in self.ks.subslice(1):
                    num_loads += M.nloads(
                        self.arch,
                        Dtype.FP16,
                        nss, n,
                        kss, self.op.k,
                        contig=True)

                    exec_cyc += 1

        return num_loads, exec_cyc

    @functools.cached_property
    def _gemms(self):
        gemms = [0 for _ in range(len(self.hs) * len(self.ws))]
        i = 0

        for ri in self.rs.indices:
            for si in self.ss.indices:
                for ni in self.ns.indices:
                    for pi in self.ps.indices:
                        gemms[i] = len(self.qs)
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


@M.register_placement([OracleArch, BgroupArch, FbcastArch, HierArch], Conv2DDw)
def place_conv2d_dw_default(arch : Arch, conv : Conv2DDw, sim : M.SimBase):
    tdi, tw, tdo = make_conv2d_tensors(arch, conv)

    assert conv.dtype == Dtype.FP16
    assert conv.tr_w == False

    sim.map2d_place([
        [
            [
                Conv2DDwTile(
                    arch, conv, [tdo, tw], [tdi], False,
                    ns, ps, qs, br0, bs0, ks, cs)

                for ns in bn1.subslice(Conv2DDwTile.tn)
                for ps in Slice(0, conv.p).subslice(Conv2DDwTile.tp)
                for qs in Slice(0, conv.q).subslice(Conv2DDwTile.tq)
                for cs in Slice(0, conv.c).subslice(Conv2DDwTile.tc)
                for ks in bk1.subslice(Conv2DDwTile.tk)
            ]
            for bn1 in bn0.blkslice(4)
            for bs0 in Slice(0, conv.s).subslice(1)
            for bk1 in bk0.blkslice(16)
        ]
        for bn0 in Slice(0, conv.n).blkslice(4)
        for br0 in Slice(0, conv.r).subslice(1)
        for bk0 in Slice(0, conv.k).blkslice(16)
    ])

