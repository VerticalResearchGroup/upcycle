import numpy as np
from dataclasses import dataclass
import itertools

from ..common import *
from .common import *

from .. import ops


@dataclass(frozen=True)
class MatmulTile(WorkItem):
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
    def exec_lat(self): return 0

    @property
    def flops(self): return self.tm * self.tn * self.mm.k * 2

    @property
    def read_trace(self):
        m = self.mm.m
        n = self.mm.k
        k = self.mm.k
        if not self.tr_a:
            yield AffineTile(1, self.mo * k + self.ko, k, self.tk, self.tm)
        else:
            yield AffineTile(1, self.ko * m + self.mo, m, self.tm, self.tk)

        if self.tr_b:
            yield AffineTile(2, self.no * k + self.ko, k, self.tk, self.tn)
        else:
            yield AffineTile(2, self.ko * n + self.no, n, self.tn, self.tk)

@register_placement('naive', ops.Matmul)
@register_placement('naive', ops.Linear)
def place_matmul_naive(arch : Arch, mm : ops.Matmul):
    tiles = [
        [
            MatmulTile(
                mm, li, bm, bn, bk,
                min(mm.m - bm, 16), min(mm.n - bn, 4), min(mm.k - bk, 64),
                mm.tr_a, mm.tr_b)
            for bk in range(0, mm.k, 64)
        ]
        for bm in range(0, mm.m, 16)
        for bn in range(0, mm.n, 4)
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
