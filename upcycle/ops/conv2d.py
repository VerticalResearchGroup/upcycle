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

@operator
@dataclass(frozen=True)
@register_backward(Conv2D)
class Conv2DBwd(Conv2D):
    @property
    def flops(self):
        flops = 0
        for hi in range(self.h):
            for wi in range(self.w):
                ho = hi % self.stride
                wo = wi % self.stride

                hr = np.ceil((self.r - ho) / self.stride)
                wr = np.ceil((self.s - wo) / self.stride)

                flops += hr * wr * self.c * self.k * 2

        return flops * self.n + super().flops
        # return super().flops

    @staticmethod
    def from_forward(c : Conv2D):
        return Conv2DBwd(c.dtype, False, c.n, c.h, c.w, c.c, c.p, c.q, c.k, c.r, c.s, c.stride, c.pad, c.tr_w)

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

def place_conv_pq_spatial2(arch : Arch, conv : Conv2D, sim : M.SimBase):
    ti, tw, to = make_conv2d_tensors(arch, conv)

    tile = {
        (Dtype.I8, False): Conv2DTileI8,
        (Dtype.I8, True): Conv2DTileI8TW,
    }[(conv.dtype, conv.tr_w)]

    kblk = tile.tk

    qk = conv.q * conv.k // kblk
    qkgrp = qk // arch.ncols

    pgrp = conv.p / arch.nrows

    for n in range(0, conv.n):
        for p in range(conv.p):
            row = int(p / pgrp)
            for q in range(conv.q):
                for k in range(0, conv.k, kblk):
                    col = int((q * conv.k // kblk + k // kblk) / qkgrp)
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
                    ], bbox=(row, col))

def blk_k(dtype : Dtype, k : int):
    kblk_in = 16 if dtype == Dtype.I8 else 8

    kblk_left = int(np.ceil(k / kblk_in))

    if kblk_left < 4: kblk_col = 1
    elif kblk_left < 8: kblk_col = 4
    else: kblk_col = 8

    kblk_left = int(np.ceil(kblk_left / kblk_col))

    if kblk_left < 4: kblk_row = 1
    elif kblk_left < 8: kblk_row = 4
    else: kblk_row = 8

    kblk_left = int(np.ceil(kblk_left / kblk_row))

    return kblk_col, kblk_row, kblk_in, kblk_left

@deprecated
def place_conv_pkqk_spatial_t(arch : Arch, conv : Conv2D, sim : M.SimBase):
    ti, tw, to = make_conv2d_tensors(arch, conv)

    tile = {
        (Dtype.I8, False): Conv2DTileI8,
        (Dtype.I8, True): Conv2DTileI8TW,
    }[(conv.dtype, conv.tr_w)]

    assert False

    kblk_col, kblk_row, kblk_in, kblk_out = blk_k(conv.dtype, conv.k)
    assert kblk_col * kblk_row * kblk_in * kblk_out == conv.k

    logger.debug(f'+ kblk_col={kblk_col}, kblk_row={kblk_row}, kblk_in={kblk_in}')

    pblk = max(1, int(np.ceil(conv.p / (arch.ncols / kblk_col))))
    qblk = max(1, int(np.ceil(conv.q / (arch.nrows / kblk_row))))

    logger.debug(f'+ pblk={pblk}, qblk={qblk}')

    for ni in range(0, conv.n):
        for p in range(0, conv.p, pblk):
            for q in range(0, conv.q, qblk):
                ki = 0
                for _ in range(0, kblk_out):
                    for kbi in range(kblk_col * kblk_row):
                        col = (p // pblk) * kblk_col + (kbi % kblk_col)
                        row = (q // qblk) * kblk_row + (kbi // kblk_col)

                        assert conv.k > ki

                        sim.flatmap_place([[
                            tile(
                                arch, conv.dtype,
                                conv, ti, tw, to, False, ni,
                                Slice.blk(p, conv.p, pblk),
                                Slice.blk(q, conv.q, qblk),
                                Slice.blk(bc, conv.c, 16),
                                Slice.blk(ki, conv.k, kblk_in))
                            for bc in range(0, conv.c, 16)
                        ]], bbox=(row, col))

                        ki += kblk_in

@M.register_placement('flatmap', [OracleArch, BgroupArch, FbcastArch], Conv2D)
def place_conv2d_flatmap(arch : Arch, conv : Conv2D, sim : M.SimBase):
    ti, tw, to = make_conv2d_tensors(arch, conv)
    npixels = conv.n * conv.p * conv.q

    tile = {
        (Dtype.I8, False): Conv2DTileI8,
        (Dtype.I8, True): Conv2DTileI8TW,
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
    return profiled_placement(arch, conv, sim, place_conv2d_flatmap)

@dataclass(frozen=True)
class Conv2DDiTile(M.WorkItem):
    write : bool
    ni : int
    hs : Slice
    ws : Slice
    cs : Slice
    ks : Slice

    @property
    def do(self): return self.inputs[0]

    @property
    def w(self): return self.inputs[1]

    @property
    def di(self): return self.outputs[0]

    @property
    def flops(self):
        flops = 0
        for hi in self.hs.indices:
            for wi in self.ws.indices:
                ho = hi % self.op.stride
                wo = wi % self.op.stride

                hr = np.ceil((self.op.r - ho) / self.op.stride)
                wr = np.ceil((self.op.s - wo) / self.op.stride)

                flops += hr * wr * len(self.cs) * len(self.ks) * 2

        return flops


    @property
    def read_trace(self):
        st = self.op.stride
        yield from self.di[self.ni, self.hs, self.ws, self.cs]

        for hi in range(min(len(self.hs), st)):
            for wi in range(min(len(self.ws), st)):
                ho = hi % self.op.stride
                wo = wi % self.op.stride
                if not self.op.tr_w: yield from self.w[ho::st, wo::st, self.ks, self.cs]
                else: yield from self.w[ho::st, wo::st, self.cs, self.ks]

        yield from self.do[self.ni, self.hs / st, self.ws / st, self.ks]


    @property
    def write_trace(self):
        if not self.write: return
        raise NotImplementedError()

@dataclass(frozen=True)
class Conv2DDwTile(M.WorkItem):
    write : bool
    ni : int

    ps : Slice
    qs : Slice

    rs : Slice
    ss : Slice

    cs : Slice
    ks : Slice

    @property
    def i(self): return self.inputs[0]

    @property
    def do(self): return self.inputs[1]

    @property
    def dw(self): return self.outputs[0]

    @property
    def flops(self):
        return len(self.ps) * len(self.qs) * len(self.rs) * len(self.ss) * len(self.cs) * len(self.ks) * 2

    @property
    def read_trace(self):
        st = self.op.stride

        hs = self.ps * st
        hs = Slice(hs.start, hs.stop, st)

        ws = self.qs * st
        ws = Slice(ws.start, ws.stop, st)

        for ri in self.rs.indices:
            for si in self.ss.indices:
                yield from self.i[self.ni, hs + ri, ws + si, self.cs]
        yield from self.do[self.ni, self.ps, self.qs, self.ks]


    @property
    def write_trace(self):
        if not self.write: return
        raise NotImplementedError()

@M.register_placement('flatmap', [OracleArch, BgroupArch, FbcastArch], Conv2DBwd)
def place_conv2d_bwd_flatmap(arch : Arch, conv : Conv2D, sim : M.SimBase):
    n = conv.n
    h = conv.h
    w = conv.w
    c = conv.c
    p = conv.p
    q = conv.q
    k = conv.k
    r = conv.r
    s = conv.s
    st = conv.stride
    pad = conv.pad

    ti = M.Tensor(arch, 1, conv.dtype, (n, h + pad * 2, w + pad * 2, c))
    tw = M.Tensor(arch, 2, conv.dtype, (r, s, k, c) if not conv.tr_w else (r, s, c, k))
    # to = M.Tensor(arch, 3, conv.dtype, (n, p, q, k))
    tdi = M.Tensor(arch, 4, conv.dtype, (n, h + pad * 2, w + pad * 2, c))
    tdw = M.Tensor(arch, 5, conv.dtype, (r, s, k, c) if not conv.tr_w else (r, s, c, k))
    tdo = M.Tensor(arch, 6, conv.dtype, (n, p, q, k))

    hblk = st
    wblk = st
    cols_per_didw = arch.ncols // 2
    nblk = 1

    off = sim.flatmap_place([
        [
            Conv2DDiTile(
                arch, conv, [tdo, tw], [tdi], False,
                ni,
                Slice.blk(bh, conv.h + pad, hblk),
                Slice.blk(bw, conv.w + pad, wblk),
                Slice.blk(bc, conv.c, 16),
                Slice.blk(bk, conv.k, 16))
            for bk in range(0, conv.k, 16)
        ]
        for ni in range(0, n, nblk)
        for bh in range(pad, conv.h + pad, hblk)
        for bw in range(pad, conv.w + pad, wblk)
        for bc in range(0, conv.c, 16)
    ], bbox=None, randomize=False)

    sim.flatmap_place([
        [
            Conv2DDwTile(
                arch, conv, [ti, tdo], [tdw], False,
                ni,
                Slice.blk(bp, conv.p, 16),
                Slice.blk(bq, conv.q, 16),
                Slice.blk(br, conv.r, 1),
                Slice.blk(bs, conv.s, 1),
                Slice.blk(bc, conv.c, 1),
                Slice.blk(bk, conv.k, 32))
            for ni in range(0, conv.n, 1)
            for bp in range(0, conv.p, 16)
            for bq in range(0, conv.q, 16)
        ]
        for bc in range(0, conv.c, 1)
        for br in range(0, conv.r, 1)
        for bs in range(0, conv.s, 1)
        for bk in range(0, conv.k, 32)
    ], offset=off, randomize=False)

@M.register_placement('pg', [OracleArch, BgroupArch, FbcastArch], Conv2DBwd)
def place_conv2d_profiled(arch : Arch, conv : Conv2DBwd, sim : M.SimBase):
    return profiled_placement(arch, conv, sim, place_conv2d_bwd_flatmap)
