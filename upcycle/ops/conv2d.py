from dataclasses import dataclass
import numpy as np
import logging

from ..common import *
from .common import *

logger = logging.getLogger(__name__)

@dataclass(frozen=True)
class Conv2D(Operator):
    n : int
    h : int
    w : int
    c : int
    p : int
    q : int
    k : int
    r : int
    s : int
    stride : int
    pad : int
    tr_w : bool

    @property
    def flops(self):
        return self.n * self.p * self.q * self.k * self.r * self.s * self.c * 2

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

        return flops * self.n

    @staticmethod
    def from_forward(c : Conv2D):
        return Conv2DBwd(c.dtype, False, c.n, c.h, c.w, c.c, c.p, c.q, c.k, c.r, c.s, c.stride, c.tr_w)

@dataclass(frozen=True)
class Conv2DTile(M.WorkItemPerfectCompute):
    conv : Conv2D
    i : M.Tensor
    w : M.Tensor
    o : M.Tensor
    write : bool
    ni : int
    ps : Slice
    qs : Slice
    cs : Slice
    ks : Slice
    tr_w : bool


    @property
    def flops(self):
        return \
            len(self.ps) * \
            len(self.qs) * \
            len(self.ks) * \
            len(self.cs) * \
            self.conv.r * self.conv.s * 2

    @property
    def read_trace(self):
        st = self.conv.stride
        yield from self.i[self.ni, self.ps * st, self.qs * st, self.cs]

        if not self.tr_w:
            yield from self.w[:, :, self.ks, self.cs]
        else:
            yield from self.w[:, :, self.cs, self.ks]

    @property
    def write_trace(self):
        if not self.write: return
        raise NotImplementedError()

@M.register_placement('flatmap', FlatMeshArch, Conv2D)
@M.register_placement('flatmap', BgroupArch, Conv2D)
@M.register_placement('flatmap', OracleArch, Conv2D)
def place_conv2d_flatmap(arch : Arch, conv : Conv2D):
    n = conv.n
    h = conv.h
    w = conv.w
    c = conv.c
    p = conv.p
    q = conv.q
    k = conv.k
    r = conv.r
    s = conv.s
    pad = conv.pad

    ti = M.Tensor(1, conv.dtype, (n, h + 2 * pad, w + 2 * pad, c))
    tw = M.Tensor(2, conv.dtype, (r, s, k, c) if not conv.tr_w else (r, s, c, k))
    to = M.Tensor(3, conv.dtype, (n, p, q, k))

    npixels = n * p * q
    if npixels > arch.ntiles:
        kblk = 128
    elif npixels > arch.ntiles // 4:
        kblk = 64
    else:
        kblk = 8

    qblk = int(max(1, 32 / kblk))
    nblk = int(max(1, arch.ncols // n))

    wl = M.WorkList.from_arch(arch, [ti, tw, to])
    for ni, col in enumerate(range(0, arch.ncols, nblk)):
        ncols = min(nblk, arch.ncols - col)
        wl.flatmap_place([
            [
                Conv2DTile(
                    arch, conv.dtype,
                    conv, ti, tw, to, False, ni,
                    Slice.blk(bp, conv.p, 1),
                    Slice.blk(bq, conv.q, qblk),
                    Slice.blk(bc, conv.c, 16),
                    Slice.blk(bk, conv.k, kblk),
                    conv.tr_w)
                for bc in range(0, conv.c, 16)
            ]
            for bp in range(0, conv.p, 1)
            for bq in range(0, conv.q, qblk)
            for bk in range(0, conv.k, kblk)
        ], bbox=(0, arch.nrows, col, col + ncols), randomize=False)

    return wl


@dataclass(frozen=True)
class Conv2DDiTile(M.WorkItemPerfectCompute):
    conv : Conv2DBwd
    di : M.Tensor
    w : M.Tensor
    do : M.Tensor
    write : bool
    ni : int
    hs : Slice
    ws : Slice
    cs : Slice
    ks : Slice
    tr_w : bool


    @property
    def flops(self):
        flops = 0
        for hi in self.hs.indices:
            for wi in self.ws.indices:
                ho = hi % self.conv.stride
                wo = wi % self.conv.stride

                hr = np.ceil((self.conv.r - ho) / self.conv.stride)
                wr = np.ceil((self.conv.s - wo) / self.conv.stride)

                flops += hr * wr * len(self.cs) * len(self.ks) * 2

        return flops


    @property
    def read_trace(self):
        st = self.conv.stride

        yield from self.di[self.ni, self.hs, self.ws, self.cs]

        for hi in range(min(len(self.hs), st)):
            for wi in range(min(len(self.ws), st)):
                ho = hi % self.conv.stride
                wo = wi % self.conv.stride
                if not self.tr_w: yield from self.w[ho::st, wo::st, self.ks, self.cs]
                else: yield from self.w[ho::st, wo::st, self.cs, self.ks]

        yield from self.do[self.ni, self.hs / st, self.ws / st, self.ks]


    @property
    def write_trace(self):
        if not self.write: return
        raise NotImplementedError()

@M.register_placement('flatmap', FlatMeshArch, Conv2DBwd)
@M.register_placement('flatmap', BgroupArch, Conv2DBwd)
@M.register_placement('flatmap', OracleArch, Conv2DBwd)
def place_conv2d_bwd_flatmap(arch : Arch, conv : Conv2D):
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

    ti = M.Tensor(1, conv.dtype, (n, h + pad * 2, w + pad * 2, c))
    tw = M.Tensor(2, conv.dtype, (r, s, k, c) if not conv.tr_w else (r, s, c, k))
    to = M.Tensor(3, conv.dtype, (n, p, q, k))
    tdi = M.Tensor(4, conv.dtype, (n, h + pad * 2, w + pad * 2, c))
    tdw = M.Tensor(5, conv.dtype, (r, s, k, c) if not conv.tr_w else (r, s, c, k))
    tdo = M.Tensor(6, conv.dtype, (n, p, q, k))

    hblk = st
    wblk = st
    nblk = int(max(1, arch.ncols // n))

    wl = M.WorkList.from_arch(arch, [ti, tw, to, tdi, tdw, tdo])
    for ni, col in enumerate(range(0, arch.ncols, nblk)):
        ncols = min(nblk, arch.ncols - col)
        wl.flatmap_place([
            [
                Conv2DDiTile(
                    arch, conv.dtype,
                    conv, tdi, tw, tdo, False, ni,
                    Slice.blk(bh, conv.h + pad, hblk),
                    Slice.blk(bw, conv.w + pad, wblk),
                    Slice.blk(bc, conv.c, 16),
                    Slice.blk(bk, conv.k, 16),
                    conv.tr_w)
                for bk in range(0, conv.k, 16)
            ]
            for bh in range(pad, conv.h + pad, hblk)
            for bw in range(pad, conv.w + pad, wblk)
            for bc in range(0, conv.c, 16)
        ], bbox=(0, arch.nrows, col, col + ncols), randomize=False)

    return wl