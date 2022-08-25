from dataclasses import dataclass
import numpy as np
import logging
import functools

from ..common import *
from .common import *

from . import matmul

logger = logging.getLogger(__name__)

@operator
@dataclass(frozen=True)
class Conv2D(Operator):
    """2D Convolution Operator."""
    # Batch
    n : int
    # Input: (n, h, w, c)
    h : int
    w : int
    c : int

    # Output: (n, p, q, k)
    p : int
    q : int
    k : int

    # Filter: (r, s, k, c)
    r : int
    s : int

    stride : int
    pad : int

    # Whether or not the weight tensor is assumed to be transposed in C and K
    # dimensions. This essentially determines the layout of the small matrix
    # multiplies that occur.
    tr_w : bool = False

    # Whether or not the weight is assumed to be rotated in R and S dimensions.
    # I.e. when this is True, the layout of W is (-r, -s, k, c).
    #
    # N.B. This is functionally unused since the algorithm doesn't care what
    # spatial direction the weight is accessed in, but I still include it here
    # for clarity.
    rot_w : bool = False

    @property
    def flops(self):
        return self.n * self.p * self.q * self.k * self.r * self.s * self.c * 2

    def __repr__(self):
        return f'Conv2D(n={self.n}, i={self.h}x{self.w}x{self.c} w={self.r}x{self.s}x{self.k}x{self.c} o={self.p}x{self.q}x{self.k} by {self.stride})'

@dataclass(frozen=True)
class Conv2DTile(M.WorkItem):
    write : bool
    ni : int
    ps : Slice
    qs : Slice
    cs : Slice
    ks : Slice

    @property
    def i(self): return self.inputs[0]

    @property
    def w(self): return self.inputs[1]

    @property
    def o(self): return self.outputs[0]

    @functools.cached_property
    def flops(self):
        return \
            len(self.ps) * \
            len(self.qs) * \
            len(self.ks) * \
            len(self.cs) * \
            self.op.r * self.op.s * 2

    @property
    def read_trace(self):
        st = self.op.stride
        yield from self.i[self.ni, self.ps * st, self.qs * st, self.cs]

        if not self.op.tr_w: yield from self.w[:, :, self.ks, self.cs]
        else: yield from self.w[:, :, self.cs, self.ks]

    @functools.cached_property
    def write_trace(self):
        if not self.write: return
        raise NotImplementedError()

    def nloads_a(self, css, kss): raise NotImplementedError()

    def nloads_b(self, kss, qss): raise NotImplementedError()

    @functools.cached_property
    def exec_lat(self):
        num_loads = 0
        exec_cyc = 0
        rstride = 4 if self.op.dtype is Dtype.I8 else 2
        for _ in self.ps.subslice(self.tp):
            for qss in self.qs.subslice(self.tq):
                for br in range(self.op.r * self.op.s):
                    for kss in self.ks.subslice(self.tk):
                        for css in self.cs.subslice(self.tc):
                            num_loads += self.nloads_a(css, kss)
                            num_loads += self.nloads_b(kss, qss)
                            exec_cyc += len(qss) * cld(len(css), rstride)

        return max(num_loads / self.arch.l1.rports, exec_cyc)

@dataclass(frozen=True)
class Conv2DTileI8(Conv2DTile):
    tk = 16 # M
    tp = 1
    tq = 4  # N
    tc = 64 # K

    def __post_init__(self):
        assert self.op.dtype == Dtype.I8
        assert not self.op.tr_w

    def nloads_a(self, css, kss):
        return M.nloads(self.arch, Dtype.I8, kss, self.op.k, css, self.op.c)

    def nloads_b(self, kss, qss):
        return M.nloads(
            self.arch,
            Dtype.I8,
            kss, self.op.k,
            qss, self.op.q,
            transpose=True,
            contig=self.op.stride == 1)

@dataclass(frozen=True)
class Conv2DTileFP16(Conv2DTile):
    tk = 16 # M
    tp = 1
    tq = 4  # N
    tc = 32 # K

    def __post_init__(self):
        assert self.op.dtype == Dtype.FP16
        assert not self.op.tr_w

    def nloads_a(self, css, kss):
        return M.nloads(self.arch, Dtype.FP16, kss, self.op.k, css, self.op.c)

    def nloads_b(self, kss, qss):
        return M.nloads(
            self.arch,
            Dtype.FP16,
            kss, self.op.k,
            qss, self.op.q,
            transpose=True,
            contig=self.op.stride == 1)

@dataclass(frozen=True)
class Conv2DTileI8TW(Conv2DTile):
    tk = 16 # M
    tp = 1
    tq = 4  # N
    tc = 4  # K

    def __post_init__(self):
        assert self.op.dtype == Dtype.I8
        assert self.op.tr_w

    def nloads_a(self, css, kss):
        return M.nloads(
            self.arch,
            Dtype.I8,
            kss, self.op.k,
            css, self.op.c,
            transpose=True)

    def nloads_b(self, kss, qss):
        return M.nloads(
            self.arch,
            Dtype.I8,
            kss, self.op.k,
            qss, self.op.q,
            transpose=True,
            contig=self.op.stride == 1)

@dataclass(frozen=True)
class Conv2DTileFP16TW(Conv2DTile):
    tk = 16 # M
    tp = 1
    tq = 4  # N
    tc = 2  # K

    def __post_init__(self):
        assert self.op.dtype == Dtype.FP16
        assert self.op.tr_w

    def nloads_a(self, css, kss):
        return M.nloads(
            self.arch,
            Dtype.I8,
            kss, self.op.k,
            css, self.op.c,
            transpose=True)

    def nloads_b(self, kss, qss):
        return M.nloads(
            self.arch,
            Dtype.I8,
            kss, self.op.k,
            qss, self.op.q,
            transpose=True,
            contig=self.op.stride == 1)

def make_conv2d_tensors(arch : Arch, conv : Conv2D):
    ti = M.Tensor(
        arch,
        1,
        conv.dtype,
        (conv.n, conv.h + 2 * conv.pad, conv.w + 2 * conv.pad, conv.c))

    tw = M.Tensor(
        arch,
        2,
        conv.dtype,
        (conv.r, conv.s, conv.k, conv.c) if not conv.tr_w \
            else (conv.r, conv.s, conv.c, conv.k))

    to = M.Tensor(arch, 3, conv.dtype, (conv.n, conv.p, conv.q, conv.k))
    return ti, tw, to

@deprecated
def place_conv_pq_spatial(arch : Arch, conv : Conv2D, sim : M.SimBase):
    ti, tw, to = make_conv2d_tensors(arch, conv)

    tile = {
        (Dtype.I8, False): Conv2DTileI8,
        (Dtype.I8, True): Conv2DTileI8TW,
        (Dtype.FP16, False): Conv2DTileFP16,
        (Dtype.FP16, True): Conv2DTileFP16TW,
    }[(conv.dtype, conv.tr_w)]

    kblk = tile.tk
    kgrp = conv.k // kblk

    pgrp = conv.p / arch.nrows
    qgrp = conv.q / (arch.ncols / kgrp)

    for n in range(0, conv.n):
        for k in range(0, conv.k, kblk):
            for p in range(conv.p):
                row = int(p / pgrp)
                for q in range(conv.q):
                    col = (int(q / qgrp) + (k // kblk) * (arch.ncols // kgrp))
                    sim.flatmap_place([
                        [
                            tile(
                                arch, conv, [ti, tw], [to], False,
                                n,
                                Slice.blk(p, conv.p, 1),
                                Slice.blk(q, conv.q, 1),
                                Slice.blk(bc, conv.c, tile.tc),
                                Slice.blk(k, conv.k, kblk))
                            for bc in range(0, conv.c, tile.tc)
                        ]
                    ], bbox=(row, row + 1, col, col + 1))

@M.register_placement('default', [OracleArch, BgroupArch, FbcastArch, HierArch], Conv2D)
def place_conv2d_default(arch : Arch, conv : Conv2D, sim : M.SimBase):
    ti, tw, to = make_conv2d_tensors(arch, conv)

    tile = {
        (Dtype.I8, False): Conv2DTileI8,
        (Dtype.I8, True): Conv2DTileI8TW,
        (Dtype.FP16, False): Conv2DTileFP16,
        (Dtype.FP16, True): Conv2DTileFP16TW,
    }[(conv.dtype, conv.tr_w)]

    sim.flatmap_place([
        [
            tile(arch, conv, [ti, tw], [to], False, ni, bp0, bq0, bc1, bk0)
            for bc1 in bc0.subslice(tile.tc)
        ]
        for ni in Slice(0, conv.n).indices
        for bp0 in Slice(0, conv.p).subslice(tile.tp)
        for bq0 in Slice(0, conv.q).subslice(tile.tq)
        for bk0 in Slice(0, conv.k).subslice(tile.tk)
        for bc0 in Slice(0, conv.c).blkslice(1)
    ])

@M.register_placement('pg', [CoarseOracle], Conv2D)
def place_conv2d_coarse_arch(arch : CoarseOracle, conv : Conv2D, sim : M.SimBase):
    ti, tw, to = make_conv2d_tensors(arch, conv)

    tile = {
        (Dtype.I8, False): Conv2DTileI8,
        (Dtype.I8, True): Conv2DTileI8TW,
        (Dtype.FP16, False): Conv2DTileFP16,
        (Dtype.FP16, True): Conv2DTileFP16TW,
    }[(conv.dtype, conv.tr_w)]

    sim.map2d_place([
        [
            [
                tile(arch, conv, [ti, tw], [to], False, ni, bp0, bq0, bc1, bk0)
                for bc1 in bc0.subslice(tile.tc)
            ]
            for bk0 in Slice(0, conv.k).subslice(tile.tk)
            for bq0 in Slice(0, conv.q).subslice(tile.tq)
            for bc0 in Slice(0, conv.c).blkslice(1)
            for ni in Slice(0, conv.n).indices
        ]
        for bp0 in Slice(0, conv.p).blkslice(arch.nrows)
    ])


@M.register_placement('pg', [OracleArch, BgroupArch, FbcastArch, HierArch], Conv2D)
def place_conv2d_profiled(arch : Arch, conv : Conv2D, sim : M.SimBase):
    return profiled_placement(arch, conv, sim, place_conv2d_default)

@placement_profile(
    [OracleArch, BgroupArch, FbcastArch, HierArch],
    Conv2D(None, None, None, None, None, None, None, None, None, 1, 1, None, None, None))
def place_conv_1x1(arch : Arch, conv : Conv2D, sim : M.SimBase):
    # We observe in this case the convolution degenerates into a large matmul.
    mm = matmul.Matmul(
        conv.dtype, conv.train, 1, conv.n * conv.p * conv.q, conv.k, conv.c,
        False, not conv.tr_w)

    assert mm.flops == conv.flops
    return M.place_op('pg', arch, mm, sim, False)

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
    ni : int
    hs : Slice
    ws : Slice
    ks : Slice
    cs : Slice

    tc = 32
    tk = 32

    @property
    def do(self): return self.inputs[0]

    @property
    def w(self): return self.inputs[1]

    @property
    def di(self): return self.outputs[0]

    @property
    def read_trace(self):
        pad = self.op.pad
        st = self.op.stride
        ps = Slice((self.hs.start + pad) // st, self.hs.stop // st, 1)
        qs = Slice((self.ws.start + pad) // st, self.ws.stop // st, 1)
        yield from self.do[self.ni, ps, qs, self.ks]

        if not self.op.tr_w: yield from self.w[:, :, self.ks, self.cs]
        else: yield from self.w[:, :, self.cs, self.ks]

    @property
    def write_trace(self):
        if not self.write: return
        raise NotImplementedError()

    def _small_gemm(self, n):
        ns = Slice(0, n)
        # Each small gemm is basically a Cx1xK matmul with KMKN layout. This
        # code should look similar to the FP16 matmul code.

        num_loads = 0
        exec_cyc = 0
        for css in self.cs.subslice(self.tc):
            for kss in self.ks.subslice(1):
                num_loads += M.nloads(
                    self.arch,
                    Dtype.FP16,
                    css, self.op.c,
                    self.ks, self.op.k,
                    transpose=True)

                for nss in ns.subslice(1):
                    num_loads += M.nloads(
                        self.arch,
                        Dtype.FP16,
                        kss, self.op.k,
                        nss, n,
                        contig=False)

                    exec_cyc += 1

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

@M.register_placement('default', [OracleArch, BgroupArch, FbcastArch, HierArch], Conv2DDi)
def place_conv2d_di_default(arch : Arch, conv : Conv2DDi, sim : M.SimBase):
    tdi, tw, tdo = make_conv2d_tensors(arch, conv)

    tile = {
        (Dtype.FP16, False): Conv2DDiTile,
    }[(conv.dtype, conv.tr_w)]

    pad = conv.pad

    sim.flatmap_place([
        [
            tile(
                arch, conv, [tdo, tw], [tdi], False,
                ni,
                Slice.blk(bh, conv.h + pad, conv.stride),
                Slice.blk(bw, conv.w + pad, conv.stride),
                Slice.blk(bk, conv.k, tile.tk),
                Slice.blk(bc, conv.c, tile.tc))
            for ni in bn.indices
            for bk in range(0, conv.k, tile.tk)
        ]
        for bn in Slice(0, conv.n).subslice(4)
        for bh in range(0, conv.h + pad, conv.stride)
        for bw in range(0, conv.w + pad, conv.stride)
        for bc in range(0, conv.c, tile.tc)
    ])



@M.register_placement('pg', [OracleArch, BgroupArch, FbcastArch, HierArch], Conv2DDi)
def place_conv2ddi_profiled(arch : Arch, conv : Conv2D, sim : M.SimBase):
    return profiled_placement(arch, conv, sim, place_conv2d_di_default)


@placement_profile(
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
    return M.place_op('pg', arch, new_conv, sim, False)

# @operator
# @dataclass(frozen=True)
# @register_backward(Conv2D, weight_update=True)
# class Conv2DDw(Conv2D):
#     @staticmethod
#     def from_forward(c : Conv2D):
#         return Conv2DDw(c.dtype, False, c.n, c.h, c.w, c.c, c.p, c.q, c.k, c.r, c.s, c.stride, c.pad, c.tr_w)

