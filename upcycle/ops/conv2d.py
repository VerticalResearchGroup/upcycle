from dataclasses import dataclass
import numpy as np
import logging

from ..common import *
from .common import *

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
    tr_w : bool

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

    @property
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

    @property
    def write_trace(self):
        if not self.write: return
        raise NotImplementedError()

@dataclass(frozen=True)
class Conv2DTileI8(Conv2DTile):
    tk = 16 # M
    tp = 1
    tq = 4  # N
    tc = 64 # K

    def __post_init__(self):
        assert self.op.dtype == Dtype.I8
        assert not self.op.tr_w

    @property
    def exec_lat(self):
        num_loads = 0
        exec_cyc = 0
        for _ in self.ps.subslice(self.tp):
            for qss in self.qs.subslice(self.tq):
                for br in range(self.op.r * self.op.s):
                    for kss in self.ks.subslice(self.tk):
                        for css in self.cs.subslice(self.tc):
                            num_loads += M.nloads(
                                self.arch, Dtype.I8, kss, self.op.k, css, self.op.c)

                            num_loads += M.nloads(
                                self.arch,
                                Dtype.I8,
                                kss, self.op.k,
                                qss, self.op.q,
                                transpose=True,
                                contig=self.op.stride == 1)

                            exec_cyc += len(qss) * cld(len(css), 4)

        return max(num_loads / self.arch.l1_rports, exec_cyc)

@dataclass(frozen=True)
class Conv2DTileFP16(Conv2DTile):
    tk = 16 # M
    tp = 1
    tq = 4  # N
    tc = 32 # K

    def __post_init__(self):
        assert self.op.dtype == Dtype.FP16
        assert not self.op.tr_w

    @property
    def exec_lat(self):
        num_loads = 0
        exec_cyc = 0
        for _ in self.ps.subslice(self.tp):
            for qss in self.qs.subslice(self.tq):
                for br in range(self.op.r * self.op.s):
                    for kss in self.ks.subslice(self.tk):
                        for css in self.cs.subslice(self.tc):
                            num_loads += M.nloads(
                                self.arch, Dtype.FP16, kss, self.op.k, css, self.op.c)

                            num_loads += M.nloads(
                                self.arch,
                                Dtype.FP16,
                                kss, self.op.k,
                                qss, self.op.q,
                                transpose=True,
                                contig=self.op.stride == 1)

                            exec_cyc += len(qss) * cld(len(css), 2)

        return max(num_loads / self.arch.l1_rports, exec_cyc)

@dataclass(frozen=True)
class Conv2DTileI8TW(Conv2DTile):
    tk = 16 # M
    tp = 1
    tq = 4  # N
    tc = 4  # K

    def __post_init__(self):
        assert self.op.dtype == Dtype.I8
        assert self.op.tr_w

    @property
    def exec_lat(self):
        num_loads = 0
        exec_cyc = 0
        for _ in self.ps.subslice(self.tp):
            for qss in self.qs.subslice(self.tq):
                for br in range(self.op.r * self.op.s):
                    for kss in self.ks.subslice(self.tk):
                        for css in self.cs.subslice(self.tc):
                            num_loads += M.nloads(
                                self.arch,
                                Dtype.I8,
                                kss, self.op.k,
                                css, self.op.c,
                                transpose=True)

                            num_loads += M.nloads(
                                self.arch,
                                Dtype.I8,
                                kss, self.op.k,
                                qss, self.op.q,
                                transpose=True,
                                contig=self.op.stride == 1)

                            exec_cyc += len(qss) * cld(len(css), 4)

        return max(num_loads / self.arch.l1_rports, exec_cyc)

@dataclass(frozen=True)
class Conv2DTileFP16TW(Conv2DTile):
    tk = 16 # M
    tp = 1
    tq = 4  # N
    tc = 2  # K

    def __post_init__(self):
        assert self.op.dtype == Dtype.FP16
        assert self.op.tr_w

    @property
    def exec_lat(self):
        num_loads = 0
        exec_cyc = 0
        for _ in self.ps.subslice(self.tp):
            for qss in self.qs.subslice(self.tq):
                for br in range(self.op.r * self.op.s):
                    for kss in self.ks.subslice(self.tk):
                        for css in self.cs.subslice(self.tc):
                            num_loads += M.nloads(
                                self.arch,
                                Dtype.I8,
                                kss, self.op.k,
                                css, self.op.c,
                                transpose=True)

                            num_loads += M.nloads(
                                self.arch,
                                Dtype.I8,
                                kss, self.op.k,
                                qss, self.op.q,
                                transpose=True,
                                contig=self.op.stride == 1)

                            exec_cyc += len(qss) * cld(len(css), 2)

        return max(num_loads / self.arch.l1_rports, exec_cyc)

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

@M.register_placement('default', [OracleArch, BgroupArch, FbcastArch], Conv2D)
def place_conv2d_default(arch : Arch, conv : Conv2D, sim : M.SimBase):
    ti, tw, to = make_conv2d_tensors(arch, conv)
    npixels = conv.n * conv.p * conv.q

    tile = {
        (Dtype.I8, False): Conv2DTileI8,
        (Dtype.I8, True): Conv2DTileI8TW,
        (Dtype.FP16, False): Conv2DTileFP16,
        (Dtype.FP16, True): Conv2DTileFP16TW,
    }[(conv.dtype, conv.tr_w)]

    off = 0

    for ni in range(0, conv.n):
        off += sim.flatmap_place([
            [
                tile(
                    arch, conv, [ti, tw], [to], False,
                    ni,
                    Slice.blk(bp, conv.p, tile.tp),
                    Slice.blk(bq, conv.q, tile.tq),
                    Slice.blk(bc, conv.c, tile.tc),
                    Slice.blk(bk, conv.k, tile.tk))
                for bc in range(0, conv.c, tile.tc)
            ]
            for bp in range(0, conv.p, tile.tp)
            for bq in range(0, conv.q, tile.tq)
            for bk in range(0, conv.k, tile.tk)
        ], offset=off, bbox=None, randomize=False)

@M.register_placement('pg', [OracleArch, BgroupArch, FbcastArch], Conv2D)
def place_conv2d_profiled(arch : Arch, conv : Conv2D, sim : M.SimBase):
    return profiled_placement(arch, conv, sim, place_conv2d_default)

@operator
@dataclass(frozen=True)
@register_backward(Conv2D)
class Conv2DDi(Conv2D):
    @staticmethod
    def from_forward(c : Conv2D):
        return Conv2DDi(c.dtype, False, c.n, c.h, c.w, c.c, c.p, c.q, c.k, c.r, c.s, c.stride, c.pad, c.tr_w)

@M.register_placement('default', [OracleArch, BgroupArch, FbcastArch], Conv2DDi)
def place_conv2d_di_default(arch : Arch, conv : Conv2DDi, sim : M.SimBase):
    tdi, tw, tdo = make_conv2d_tensors(arch, conv)

    conv = Conv2DDi(conv.dtype, True, conv.n, conv.h, conv.w, conv.c, conv.p, conv.q, conv.k, conv.r, conv.s, conv.stride, conv.pad, not conv.tr_w)

    tile = {
        (Dtype.FP16, False): Conv2DTileFP16,
        (Dtype.FP16, True): Conv2DTileFP16TW,
    }[(conv.dtype, conv.tr_w)]

    off = 0

    for ni in range(0, conv.n):
        off += sim.flatmap_place([
            [
                tile(
                    arch, conv, [tdo, tw], [tdi], False,
                    ni,
                    Slice.blk(bp, conv.p, tile.tp),
                    Slice.blk(bq, conv.q, tile.tq),
                    Slice.blk(bk, conv.k, tile.tk),
                    Slice.blk(bc, conv.c, tile.tc))
                for bk in range(0, conv.k, tile.tk)
            ]
            for bp in range(0, conv.p, tile.tp)
            for bq in range(0, conv.q, tile.tq)
            for bc in range(0, conv.c, tile.tc)
        ], offset=off, bbox=None, randomize=False)



@M.register_placement('pg', [OracleArch, BgroupArch, FbcastArch], Conv2DDi)
def place_conv2ddi_profiled(arch : Arch, conv : Conv2D, sim : M.SimBase):
    return profiled_placement(arch, conv, sim, place_conv2d_di_default)


# @operator
# @dataclass(frozen=True)
# @register_backward(Conv2D, weight_update=True)
# class Conv2DDw(Conv2D):
#     @staticmethod
#     def from_forward(c : Conv2D):
#         return Conv2DDw(c.dtype, False, c.n, c.h, c.w, c.c, c.p, c.q, c.k, c.r, c.s, c.stride, c.pad, c.tr_w)

