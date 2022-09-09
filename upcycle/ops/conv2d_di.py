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
class Conv2DDi(Conv2D):
    @staticmethod
    def from_forward(c : Conv2D):
        return Conv2DDi(c.dtype, False, c.n, c.h, c.w, c.c, c.p, c.q, c.k, c.r, c.s, c.stride, c.pad, c.tr_w)

@dataclass(frozen=True)
class Conv2DDiTile(M.WorkItem):
    write : bool
    ns : Slice
    hs : Slice
    ws : Slice
    ks : Slice
    cs : Slice

    tc = 32
    tk = 2
    ttk = 1
    tn = 1

    @property
    def do(self): return self.inputs[0]

    @property
    def w(self): return self.inputs[1]

    @property
    def di(self): return self.outputs[0]

    @functools.cached_property
    def ps(self):
        return Slice(
            (self.hs.start + self.op.pad) // self.op.stride,
            self.hs.stop // self.op.stride)

    @functools.cached_property
    def qs(self):
        return Slice(
            (self.ws.start + self.op.pad) // self.op.stride,
            self.ws.stop // self.op.stride)

    @property
    def read_trace(self):
        yield from self.do[self.ns, self.ps, self.qs, self.ks]

        if not self.op.tr_w: yield from self.w[:, :, self.ks, self.cs]
        else: yield from self.w[:, :, self.cs, self.ks]

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
            for nss in ns.subslice(self.tn):
                for kss in self.ks.subslice(self.tk):
                    num_loads += M.nloads(
                        self.arch,
                        Dtype.FP16,
                        css, self.op.c,
                        self.ks, self.op.k,
                        transpose=True)

                    for ksss in kss.subslice(self.ttk):
                        num_loads += M.nloads(
                            self.arch,
                            Dtype.FP16,
                            ksss, self.op.k,
                            nss, n,
                            contig=False)

                        exec_cyc += len(nss)

        return num_loads, exec_cyc

    @functools.cached_property
    def _gemms(self):
        st = self.op.stride
        gemms = [0 for _ in range(len(self.hs) * len(self.ws))]
        i = 0
        pad = self.op.pad

        # N.B. I wrote this down as a diagram first but basically what we need
        # to do is for each pixel of dI we are computing, find the corresponding
        # _set_ of filter pixels it is multiplied with.
        for n in self.ns.indices:
            for h in self.hs.indices:
                if h < pad or h >= self.op.h + pad: continue
                oh = h % st
                for w in self.ws.indices:
                    if w < pad or w >= self.op.w + pad: continue
                    ow = w % st
                    n = 0
                    for r in range(oh, self.op.r, st):
                        for s in range(ow, self.op.s, st):
                            p = (h - r) // st
                            q = (w - s) // st
                            if p < 0 or p >= self.op.p: continue
                            if q < 0 or q >= self.op.q: continue
                            n += 1

                    gemms[i] = n
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

@M.register_placement([OracleArch, BgroupArch, FbcastArch, HierArch], Conv2DDi)
def place_conv2d_di_default(arch : Arch, conv : Conv2DDi, sim : M.SimBase):
    tdi, tw, tdo = make_conv2d_tensors(arch, conv)

    assert conv.dtype == Dtype.FP16
    assert conv.tr_w == False
    pad = conv.pad

    sim.map2d_place([
        [
            [
                Conv2DDiTile(
                    arch, conv, [tdo, tw], [tdi], False,
                    ns, hs, ws, ks, cs)

                for ns in bn1.subslice(1)
                for hs in bh0.subslice(conv.stride)
                for ws in bw0.subslice(conv.stride)
                for cs in Slice(0, conv.c).subslice(Conv2DDiTile.tc * 2)
                for ks in Slice(0, conv.k).subslice(Conv2DDiTile.tk * 4)
            ]
            for bn1 in bn0.blkslice(2)
            for bw0 in Slice(pad, conv.w + pad).blkslice(arch.ncols // 2)
        ]
        for bn0 in Slice(0, conv.n).blkslice(2)
        for bh0 in Slice(pad, conv.h + pad).blkslice(arch.nrows // 2)
    ])

@M.register_placement(
    [OracleArch, BgroupArch, FbcastArch, HierArch],
    Conv2DDi(None, None, None, None, None, None, None, None, None, None, None, 1, None, None))
def place_convdi_stride1(arch : Arch, conv : Conv2DDi, sim : M.SimBase):
    # N.B. Based on the 2018 paper from Intel, when stride=1, we can reuse the
    # forward prop algorithm for backprop. The weight tensor needs to be rotated
    # spatially by 180 degrees and the input / output channels are transposed.
    #
    # See: Georganas, et al. "Anatomy of High-Performance Deep Learning
    # Convolutions on SIMD Architectures."
    new_conv = Conv2D(
        conv.dtype, True,
        conv.n,
        conv.p, conv.q, conv.k,
        conv.h, conv.w, conv.c,
        conv.r, conv.s, conv.stride,
        conv.pad, not conv.tr_w, True)

    assert new_conv.flops == conv.flops
    return M.place_op(arch, new_conv, sim, False)

@M.register_placement(
    [OracleArch, BgroupArch, FbcastArch, HierArch],
    Conv2DDi(None, None, None, None, None, None, None, None, None, 1, 1, None, None, None))
def place_conv2ddi_1x1(arch : Arch, conv : Conv2DDi, sim : M.SimBase):
    # We observe in this case the convolution degenerates into a large matmul.
    mm = matmul.MatmulDa.from_forward(matmul.Matmul(
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
