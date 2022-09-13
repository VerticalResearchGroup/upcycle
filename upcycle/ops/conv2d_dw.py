from dataclasses import dataclass
import numpy as np
import logging
import functools

from ..common import *
from .common import *

from . import matmul
from .conv2d import Conv2D
from .reduce import Reduce

logger = logging.getLogger(__name__)


@operator
@dataclass(frozen=True)
@register_backward(Conv2D)
class Conv2DDw(Conv2D):
    @staticmethod
    def from_forward(c : Conv2D):
        return Conv2DDw(c.dtype, False, c.n, c.h, c.w, c.c, c.p, c.q, c.k, c.r, c.s, c.stride, c.pad, c.tr_w)

    def make_tensors(self, arch : Arch):
        [ti, tdw], [tdo] = super().make_tensors(arch)
        return [ti, tdo], [tdw]

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
    tp = 2
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

def choose_tile(op : Conv2DDw):
    assert op.dtype == Dtype.FP16
    assert op.tr_w == False
    return Conv2DDwTile

@M.register_placement([OracleArch, BgroupArch, FbcastArch, HierArch], Conv2DDw)
def place_conv2d_dw_default(arch : Arch, conv : Conv2DDw, sim : M.SimBase):
    ins, outs = conv.make_tensors(arch)
    tile = choose_tile(conv)
    rnblk, cnblk = blk2d(conv.n)

    sim.map2d_place([
        [
            [
                tile(
                    arch, conv, ins, outs, False,
                    ns, ps, qs, br0, bs0, ks, cs)

                for ns in bn1.subslice(tile.tn)
                for ps in Slice(0, conv.p).subslice(tile.tp)
                for qs in Slice(0, conv.q).subslice(tile.tq)
                for cs in Slice(0, conv.c).subslice(tile.tc)
                for ks in bk1.subslice(tile.tk)
            ]
            for bn1 in bn0.blkslice(cnblk)
            for bs0 in Slice(0, conv.s).subslice(1)
            for bk1 in bk0.blkslice(16)
        ]
        for bn0 in Slice(0, conv.n).blkslice(rnblk)
        for br0 in Slice(0, conv.r).subslice(1)
        for bk0 in Slice(0, conv.k).blkslice(16)
    ])

    sim.barrier()

    M.place_op(
        arch,
        Reduce(conv.dtype, False, rnblk * cnblk, len(outs[0])),
        sim,
        check_flops=False)


@M.register_placement(
    [OracleArch, BgroupArch, FbcastArch, HierArch],
    Conv2DDw(None, None, None, None, None, None, None, None, None, 1, 1, None, None, None))
def place_conv2ddw_1x1(arch : Arch, conv : Conv2DDw, sim : M.SimBase):
    # We observe in this case the convolution degenerates into a large matmul.
    mm = matmul.MatmulDb.from_forward(matmul.Matmul(
        conv.dtype,
        conv.train,
        1,
        conv.n * conv.p * conv.q,
        conv.k,
        conv.c,
        False,
        not conv.tr_w))

    assert mm.flops == conv.flops
    return M.place_op(arch, mm, sim, False)
