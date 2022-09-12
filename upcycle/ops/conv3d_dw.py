from dataclasses import dataclass
import numpy as np
import logging
import functools

from ..common import *
from .common import *

from . import matmul
from .conv3d import Conv3D, make_conv3d_tensors
from .reduce import Reduce

logger = logging.getLogger(__name__)


@operator
@dataclass(frozen=True)
@register_backward(Conv3D)
class Conv3DDw(Conv3D):
    @staticmethod
    def from_forward(c : Conv3D):
        return Conv3DDw(c.dtype, False, c.n, c.h, c.d, c.w, c.c, c.p, c.q, c.o, c.k, c.r, c.s, c.t, c.stride, c.pad, c.tr_w)

@dataclass(frozen=True)
class Conv3DDwTile(M.WorkItem):
    write : bool

    ns : Slice
    ps : Slice
    qs : Slice
    os : Slice

    rs : Slice
    ss : Slice
    ts : Slice
    ks : Slice
    cs : Slice

    tn = 1
    tp = 1
    tq = 4
    to = 8
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

    @functools.cached_property
    def ds(self): return self.os * self.op.stride

    @property
    def read_trace(self):
        yield from self.i[self.ns, self.hs, self.ws, self.ds, self.cs]
        yield from self.do[self.ns, self.ps, self.qs, self.os, self.ks]

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
                        for qi in self.qs.indices:
                            gemms[i] = len(self.os)
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


@M.register_placement([OracleArch, BgroupArch, FbcastArch, HierArch], Conv3DDw)
def place_conv3d_dw_default(arch : Arch, conv : Conv3DDw, sim : M.SimBase):
    ti, tdw, tdo = make_conv3d_tensors(arch, conv)

    assert conv.dtype == Dtype.FP16
    assert conv.tr_w == False

    sim.map2d_place([
        [
            [
                Conv3DDwTile(
                    arch, conv, [ti, tdo], [tdw], False,
                    ns, ps, qs, os, br0, bs0, bt0, ks, cs)

                for bt0 in Slice(0, conv.t).subslice(Conv3DDwTile.to)
                for ns in bn1.subslice(Conv3DDwTile.tn)
                for ps in Slice(0, conv.p).subslice(Conv3DDwTile.tp)
                for qs in Slice(0, conv.q).subslice(Conv3DDwTile.tq)
                for os in Slice(0, conv.o).subslice(Conv3DDwTile.to)
                for cs in Slice(0, conv.c).subslice(Conv3DDwTile.tc)
                for ks in bk1.subslice(Conv3DDwTile.tk)
            ]
            for bn1 in bn0.blkslice(4)
            for bs0 in Slice(0, conv.s).subslice(1)
            for bk1 in bk0.blkslice(16)
        ]
        for bn0 in Slice(0, conv.n).blkslice(4)
        for br0 in Slice(0, conv.r).subslice(1)
        for bk0 in Slice(0, conv.k).blkslice(16)
    ])

    sim.barrier()

    M.place_op(
        arch,
        Reduce(conv.dtype, False, 16, len(tdw)),
        sim,
        check_flops=False)
