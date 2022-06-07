from dataclasses import dataclass

from ..common import *
from .common import *


@dataclass(frozen=True)
class Matmul(Operator):
    l : int
    m : int
    n : int
    k : int
    tr_a : bool
    tr_b : bool

    @property
    def flops(self): return self.l * self.m * self.n * self.k * 2

@dataclass(frozen=True)
@register_backward(Matmul)
class MatmulBwd(Matmul):
    @staticmethod
    def from_forward(mm : Matmul):
        return MatmulBwd(mm.dtype, False, mm.l, mm.m, mm.n, mm.k, mm.tr_a, mm.tr_b)

@dataclass(frozen=True)
class Linear(Matmul): pass

@dataclass(frozen=True)
@register_backward(Linear)
class LinearBwd(MatmulBwd): pass


@dataclass(frozen=True)
class MatmulTile(M.WorkItemPerfectCompute):
    mm : Matmul
    a : M.Tensor
    b : M.Tensor
    c : M.Tensor
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
        if not self.tr_a: yield from self.a[self.li, self.ms, self.ks]
        else: yield from self.a[self.li, self.ks, self.ms]

        if not self.tr_b: yield from self.b[self.li, self.ks, self.ns]
        else: yield from self.b[self.li, self.ns, self.ks]

    @property
    def write_trace(self):
        if not self.write: return
        yield from self.c[self.li, self.ms, self.ns]


@M.register_placement('naive', FlatMeshArch, Matmul)
@M.register_placement('naive', FlatMeshArch, Linear)
@M.register_placement('naive', OracleArch, Matmul)
@M.register_placement('naive', OracleArch, Linear)
def place_matmul_naive(arch : Arch, mm : Matmul):
    l = mm.l
    m = mm.m
    n = mm.n
    k = mm.k

    a = M.Tensor(1, mm.dtype, (l, m, k) if not mm.tr_a else (l, k, m))
    b = M.Tensor(2, mm.dtype, (l, k, n) if not mm.tr_b else (l, n, k))
    c = M.Tensor(3, mm.dtype, (l, m, n))

    wl = M.WorkList.from_arch(arch, [a, b, c])
    wl.contract2d_place([
        [
            [
                MatmulTile(
                    arch, mm.dtype,
                    mm, a, b, c, False, li,
                    slice_blk(bm, mm.m, 16),
                    slice_blk(bn, mm.n, 8),
                    slice_blk(bk, mm.k, 64),
                    mm.tr_a, mm.tr_b)
                for bk in range(0, mm.k, 64)
            ]
            for bm in range(0, mm.m, 16)
        ]
        for bn in range(0, mm.n, 8)
        for li in range(mm.l)
    ])

    return wl

@M.register_placement('naive', FlatMeshArch, MatmulBwd)
@M.register_placement('naive', FlatMeshArch, LinearBwd)
@M.register_placement('naive', OracleArch, MatmulBwd)
@M.register_placement('naive', OracleArch, LinearBwd)
def place_matmul_bwd_naive(arch : Arch, mm : MatmulBwd):
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

    gwl = M.GlobalWorkList.from_arch(arch)
    wi_per_tile = len(tiles) // arch.ntiles
    off = 0

    for ti in range(arch.ntiles):
        ntiles = wi_per_tile
        if ti < (len(tiles) % arch.ntiles): ntiles += 1
        gwl.tiles[ti] = list(itertools.chain(*tiles[off : off + ntiles]))
        off += ntiles

    return gwl
