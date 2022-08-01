from dataclasses import dataclass
import logging

from ..common import *
from .common import *

logger = logging.getLogger(__name__)

@operator
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

@operator
@dataclass(frozen=True)
@register_backward(Matmul)
class MatmulBwd(Matmul):
    @staticmethod
    def from_forward(mm : Matmul):
        return MatmulBwd(mm.dtype, False, mm.l, mm.m, mm.n, mm.k, mm.tr_a, mm.tr_b)

    @property
    def da(self) -> Matmul:
        return Matmul(self.dtype, False, self.l, self.m, self.k, self.n, self.tr_a, not self.tr_b)

    @property
    def db(self) -> Matmul:
        return Matmul(self.dtype, False, self.l, self.k, self.n, self.m, not self.tr_a, self.tr_b)

    @property
    def flops(self): return super().flops * 2

@operator
@dataclass(frozen=True)
class Linear(Matmul): pass

@operator
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

    @property
    def flops(self):
        return \
            len(self.ms) * \
            len(self.ns) * \
            len(self.ks) * 2

    @property
    def read_trace(self):
        if not self.mm.tr_a: yield from self.a[self.li, self.ms, self.ks]
        else: yield from self.a[self.li, self.ks, self.ms]

        if not self.mm.tr_b: yield from self.b[self.li, self.ks, self.ns]
        else: yield from self.b[self.li, self.ns, self.ks]

    @property
    def write_trace(self):
        if not self.write: return
        yield from self.c[self.li, self.ms, self.ns]


def flatmap_matmul(arch : Arch, mm : Matmul, sim : M.SimBase, a, b, c, bbox=None, offset=0):
    return sim.flatmap_place([
        [
            MatmulTile(
                arch, mm.dtype,
                mm, a, b, c, False, li,
                Slice.blk(bm, mm.m, 16),
                Slice.blk(bn, mm.n, 8),
                Slice.blk(bk, mm.k, 64))
            for bk in range(0, mm.k, 64)
        ]
        for bm in range(0, mm.m, 16)
        for bn in range(0, mm.n, 8)
        for li in range(mm.l)
    ], offset=offset, bbox=bbox)

@M.register_placement('flatmap', [OracleArch, BgroupArch, FbcastArch], [Matmul, Linear])
def place_matmul_flatmap(arch : Arch, mm : Matmul, sim : M.SimBase):
    l = mm.l
    m = mm.m
    n = mm.n
    k = mm.k

    a = M.Tensor(arch, 1, mm.dtype, (l, m, k) if not mm.tr_a else (l, k, m))
    b = M.Tensor(arch, 2, mm.dtype, (l, k, n) if not mm.tr_b else (l, n, k))
    c = M.Tensor(arch, 3, mm.dtype, (l, m, n))

    flatmap_matmul(arch, mm, sim, a, b, c)


@M.register_placement('pg', [OracleArch, BgroupArch, FbcastArch], [Matmul, Linear])
def place_matmul_profiled(arch : Arch, mm : Matmul, sim : M.SimBase):
    return profiled_placement(arch, mm, sim, place_matmul_flatmap)

@M.register_placement('flatmap', [OracleArch, BgroupArch, FbcastArch], [MatmulBwd, LinearBwd])
def place_matmul_bwd_flatmap(arch : Arch, mm : MatmulBwd, sim : M.SimBase):
    l = mm.l
    m = mm.m
    n = mm.n
    k = mm.k

    a = M.Tensor(arch, 1, mm.dtype, (l, m, k) if not mm.tr_a else (l, k, m))
    b = M.Tensor(arch, 2, mm.dtype, (l, k, n) if not mm.tr_b else (l, n, k))
    # c = M.Tensor(arch, 3, mm.dtype, (l, m, n))
    da = M.Tensor(arch, 4, mm.dtype, (l, m, k) if not mm.tr_a else (l, k, m))
    db = M.Tensor(arch, 5, mm.dtype, (l, k, n) if not mm.tr_b else (l, n, k))
    dc = M.Tensor(arch, 6, mm.dtype, (l, m, n))

    off = flatmap_matmul(arch, mm.da, sim, dc, b, da)
    flatmap_matmul(arch, mm.db, sim, a, dc, db, offset=off)

@M.register_placement('pg', [OracleArch, BgroupArch, FbcastArch], [MatmulBwd, LinearBwd])
def place_matmul_bwd_profiled(arch : Arch, mm : Matmul, sim : M.SimBase):
    return profiled_placement(arch, mm, sim, place_matmul_bwd_flatmap)
