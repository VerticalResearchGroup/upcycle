import numpy as np
from dataclasses import dataclass
import itertools

from ..common import *
from .common import *

from .. import ops


@dataclass(frozen=True)
class MatmulTile(WorkItemPerfectCompute):
    mm : ops.Matmul
    write : bool
    li : int
    ms : slice
    ns : slice
    ks : slice
    tr_a : bool
    tr_b : bool

    @property
    def flops(self):
        return \
            slice_len(self.ms, self.mm.m) * \
            slice_len(self.ns, self.mm.n) * \
            slice_len(self.ks, self.mm.k) * 2

    @property
    def read_trace(self):
        l = self.mm.l
        m = self.mm.m
        n = self.mm.n
        k = self.mm.k
        if not self.tr_a:
            yield from Tensor(1, self.dtype, (l, m, k))[self.li, self.ms, self.ks]
        else:
            yield from Tensor(1, self.dtype, (l, k, m))[self.li, self.ks, self.ms]

        if not self.tr_b:
            yield from Tensor(2, self.dtype, (l, k, n))[self.li, self.ks, self.ns]
        else:
            yield from Tensor(2, self.dtype, (l, n, k))[self.li, self.ns, self.ks]

    @property
    def write_trace(self):
        if not self.write: return
        l = self.mm.l
        m = self.mm.m
        n = self.mm.n
        yield from Tensor(3, self.dtype, (l, m, n))[self.li, self.ms, self.ns]


@register_placement('naive', FlatMeshArch, ops.Matmul)
@register_placement('naive', FlatMeshArch, ops.Linear)
@register_placement('naive', OracleArch, ops.Matmul)
@register_placement('naive', OracleArch, ops.Linear)
def place_matmul_naive(arch : Arch, mm : ops.Matmul):
    tiles = [
        [
            MatmulTile(
                arch, mm.dtype,
                mm, False, li,
                slice_blk(bm, mm.m, 16),
                slice_blk(bn, mm.n, 8),
                slice_blk(bk, mm.k, 64),
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

@register_placement('naive', FlatMeshArch, ops.MatmulBwd)
@register_placement('naive', FlatMeshArch, ops.LinearBwd)
@register_placement('naive', OracleArch, ops.MatmulBwd)
@register_placement('naive', OracleArch, ops.LinearBwd)
def place_matmul_bwd_naive(arch : Arch, mm : ops.MatmulBwd):
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