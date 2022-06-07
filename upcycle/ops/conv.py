from dataclasses import dataclass
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

    @property
    def flops(self):
        return self.n * self.p * self.q * self.k * self.r * self.s * self.c * 2

@dataclass(frozen=True)
@register_backward(Conv2D)
class Conv2DBwd(Conv2D):
    @property
    def flops(self): raise NotImplementedError()

    @staticmethod
    def from_forward(c : Conv2D):
        return Conv2DBwd(c.dtype, False, c.n, c.h, c.w, c.c, c.p, c.q, c.k, c.r, c.s, c.stride)

@dataclass(frozen=True)
class Conv2DTile(M.WorkItemPerfectCompute):
    conv : Conv2D
    i : M.Tensor
    w : M.Tensor
    o : M.Tensor
    write : bool
    ni : int
    ps : slice
    qs : slice
    cs : slice
    ks : slice


    @property
    def flops(self):
        return \
            slice_len(self.ps, self.conv.p) * \
            slice_len(self.qs, self.conv.q) * \
            slice_len(self.ks, self.conv.k) * \
            slice_len(self.cs, self.conv.c) * \
            self.conv.r * self.conv.s * 2

    @property
    def read_trace(self):
        st = self.conv.stride
        yield from self.i[self.ni, slice_mul(self.ps, st), slice_mul(self.qs, st), self.cs]
        yield from self.w[:, :, self.ks, self.cs]

    @property
    def write_trace(self):
        if not self.write: return
        raise NotImplementedError()


@M.register_placement('naive', FlatMeshArch, Conv2D)
@M.register_placement('naive', OracleArch, Conv2D)
def place_conv2d_naive(arch : Arch, conv : Conv2D):
    n = conv.n
    h = conv.h
    w = conv.w
    c = conv.c
    p = conv.p
    q = conv.q
    k = conv.k
    r = conv.r
    s = conv.s

    i = M.Tensor(1, conv.dtype, (n, h, w, c))
    w = M.Tensor(2, conv.dtype, (r, s, k, c))
    o = M.Tensor(3, conv.dtype, (n, p, q, k))

    wl = M.WorkList.from_arch(arch, [i, w, o])
    wl.contract2d_place([
        [
            [
                Conv2DTile(
                    arch, conv.dtype,
                    conv, i, w, o, False, ni,
                    slice_blk(pi, conv.p, 1),
                    slice_blk(qi, conv.q, 1),
                    slice_blk(bc, conv.c, 16),
                    slice_blk(bk, conv.k, 16))
                for bc in range(0, conv.c, 16)
            ]
            for pi in range(0, conv.p, 1)
            for bk in range(bbk, min(bbk + 64, conv.k), 16)
        ]
        for ni in range(conv.n)
        for qi in range(0, conv.q, 1)
        for bbk in range(0, conv.k, 64)
    ])

    return wl

@M.register_placement('flatmap', FlatMeshArch, Conv2D)
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

    i = M.Tensor(1, conv.dtype, (n, h, w, c))
    w = M.Tensor(2, conv.dtype, (r, s, k, c))
    o = M.Tensor(3, conv.dtype, (n, p, q, k))

    npixels = n * p * q
    if npixels > arch.ntiles:
        kblk = 128
    elif npixels > arch.ntiles // 4:
        kblk = 64
    else:
        kblk = 8

    qblk = int(max(1, 32 / kblk))
    nblk = int(max(1, arch.ncols // n))

    wl = M.WorkList.from_arch(arch, [i, w, o])
    for ni, col in enumerate(range(0, arch.ncols, nblk)):
        ncols = min(nblk, arch.ncols - col)
        wl.flatmap_place([
            [
                Conv2DTile(
                    arch, conv.dtype,
                    conv, i, w, o, False, ni,
                    slice_blk(bp, conv.p, 1),
                    slice_blk(bq, conv.q, qblk),
                    slice_blk(bc, conv.c, 16),
                    slice_blk(bk, conv.k, kblk))
                for bc in range(0, conv.c, 16)
            ]
            for bp in range(0, conv.p, 1)
            for bq in range(0, conv.q, qblk)
            for bk in range(0, conv.k, kblk)
        ], bbox=(0, arch.nrows, col, col + ncols), randomize=False)

    return wl


