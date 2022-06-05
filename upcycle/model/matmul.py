import numpy as np
from dataclasses import dataclass
import itertools

from ..common import *
from .common import *

from .. import ops


@dataclass(frozen=True)
class MatmulTile(WorkItemPerfectCompute):
    mm : ops.Matmul
    li : int
    mo : int
    no : int
    ko : int
    tm : int
    tn : int
    tk : int
    tr_a : bool
    tr_b : bool

    @property
    def flops(self): return self.tm * self.tn * self.tk * 2

    @property
    def read_trace(self):
        l = self.mm.l
        m = self.mm.m
        n = self.mm.n
        k = self.mm.k
        mstart, mstop = self.mo, self.mo + self.tm
        nstart, nstop = self.no, self.no + self.tn
        kstart, kstop = self.ko, self.ko + self.tk
        if not self.tr_a:
            yield from Tensor(1, self.dtype, (l, m, k))[self.li, mstart:mstop, kstart:kstop]
        else:
            yield from Tensor(1, self.dtype, (l, k, m))[self.li, kstart:kstop, mstart:mstop]

        if not self.tr_b:
            yield from Tensor(2, self.dtype, (l, k, n))[self.li, kstart:kstop, nstart:nstop]
        else:
            yield from Tensor(2, self.dtype, (l, n, k))[self.li, nstart:nstop, kstart:kstop]

    @property
    def write_trace(self):
        l = self.mm.l
        m = self.mm.m
        n = self.mm.n
        mstart, mstop = self.mo, self.mo + self.tm
        nstart, nstop = self.no, self.no + self.tn
        yield from Tensor(3, self.dtype, (l, m, n))[self.li, mstart:mstop, nstart:nstop]


@register_placement('naive', FlatMeshArch, ops.Matmul)
@register_placement('naive', FlatMeshArch, ops.Linear)
@register_placement('naive', OracleArch, ops.Matmul)
@register_placement('naive', OracleArch, ops.Linear)
def place_matmul_naive(arch : Arch, mm : ops.Matmul):
    tiles = [
        [
            MatmulTile(
                arch, mm.dtype,
                mm, li, bm, bn, bk,
                min(mm.m - bm, 16), min(mm.n - bn, 8), min(mm.k - bk, 64),
                mm.tr_a, mm.tr_b)
            for bk in range(0, mm.k, 64)
        ]
        for bm in range(0, mm.m, 16)
        for bn in range(0, mm.n, 8)
        for li in range(mm.l)
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
