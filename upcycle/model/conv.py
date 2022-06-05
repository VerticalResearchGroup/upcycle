import numpy as np
from dataclasses import dataclass
import itertools

from ..common import *
from .common import *

from .. import ops


@dataclass(frozen=True)
class Conv2DTile(WorkItemPerfectCompute):
    conv : ops.Conv2D
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
        n = self.conv.n
        h = self.conv.h
        w = self.conv.w
        c = self.conv.c
        k = self.conv.k
        r = self.conv.r
        s = self.conv.s
        st = self.conv.stride

        yield from Tensor(1, self.dtype, (n, h, w, c))[self.ni, slice_mul(self.ps, st), slice_mul(self.qs, st), self.cs]
        yield from Tensor(2, self.dtype, (r, s, k, c))[:, :, self.ks, self.cs]


    @property
    def write_trace(self):
        if not self.write: return
        raise NotImplementedError()



@register_placement('naive', FlatMeshArch, ops.Conv2D)
@register_placement('naive', OracleArch, ops.Conv2D)
def place_conv2d_naive(arch : Arch, conv : ops.Conv2D):
    qblk = conv.p * conv.q // arch.ntiles
    tiles = [
        [
            Conv2DTile(
                arch, conv.dtype,
                conv, False, ni,
                slice_blk(pi, conv.p, 1),
                slice_blk(bq, conv.q, qblk),
                slice_blk(bc, conv.c, 16),
                slice_blk(bk, conv.k, 64))
            for bc in range(0, conv.c, 16)
        ]
        for bk in range(0, conv.k, 64)
        for bq in range(0, conv.q, qblk)
        for pi in range(0, conv.p, 1)
        for ni in range(conv.n)
    ]

    gwl = GlobalWorkList.from_arch(arch)
    wi_per_tile = len(tiles) // arch.ntiles
    off = 0

    for ti in range(arch.ntiles):
        ntiles = wi_per_tile
        if ti < (len(tiles) % arch.ntiles): ntiles += 1
        gwl.tiles[ti] = list(itertools.chain(*tiles[off : off + ntiles]))
        off += ntiles

    return gwl
