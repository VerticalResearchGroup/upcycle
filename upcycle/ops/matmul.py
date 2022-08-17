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
class MatmulTile(M.WorkItem):
    write : bool
    li : int
    ms : Slice
    ns : Slice
    ks : Slice

    @property
    def a(self): return self.inputs[0]

    @property
    def b(self): return self.inputs[1]

    @property
    def c(self): return self.outputs[0]

    @property
    def flops(self):
        return \
            len(self.ms) * \
            len(self.ns) * \
            len(self.ks) * 2

    @property
    def write_trace(self):
        if self.write: yield from self.c[self.li, self.ms, self.ns]


@dataclass(frozen=True)
class MatmulTileMKKNI8(MatmulTile):
    tm = 16
    tn = 4
    tk = 64
    ttk = 4

    def __post_init__(self):
        assert self.op.dtype == Dtype.I8
        assert not self.op.tr_a
        assert not self.op.tr_b

    @property
    def exec_lat(self):
        num_loads = 0
        exec_cyc = 0
        for mss in self.ms.subslice(self.tm):
            for nss in self.ns.subslice(self.tn):
                for kss in self.ks.subslice(self.tk):
                    # For each tk chunk, we issue a single VLD4T for A
                    num_loads += M.nloads(
                        self.arch, Dtype.I8, mss, self.op.m, kss, self.op.k)

                    # We further subtile k by another factor (ttk) and issue a
                    # VLB4X4 from B for each ttk chunk
                    for ksss in kss.subslice(self.ttk):
                        num_loads += M.nloads(
                            self.arch, Dtype.I8, ksss, self.op.k, nss, self.op.n)

                        # The number of vector FMAs issued is the product of
                        # m and n slice sizes.
                        exec_cyc += len(nss)

        return max(num_loads / self.arch.l1.rports, exec_cyc)

    @property
    def read_trace(self):
        yield from self.a[self.li, self.ms, self.ks]
        yield from self.b[self.li, self.ks, self.ns]

@dataclass(frozen=True)
class MatmulTileMKKNFP16(MatmulTile):
    tm = 16
    tn = 4
    tk = 32
    ttk = 2

    def __post_init__(self):
        assert self.op.dtype == Dtype.FP16
        assert not self.op.tr_a
        assert not self.op.tr_b

    @property
    def exec_lat(self):
        num_loads = 0
        exec_cyc = 0
        for mss in self.ms.subslice(self.tm):
            for nss in self.ns.subslice(self.tn):
                for kss in self.ks.subslice(self.tk):
                    # For each tk chunk, we issue a single VLD4T for A
                    num_loads += M.nloads(
                        self.arch, Dtype.FP16, mss, self.op.m, kss, self.op.k)

                    # We further subtile k by another factor (ttk) and issue a
                    # VLB4X4 from B for each ttk chunk
                    for ksss in kss.subslice(self.ttk):
                        num_loads += M.nloads(
                            self.arch, Dtype.FP16, ksss, self.op.k, nss, self.op.n)

                        # The number of vector FMAs issued is the product of
                        # m and n slice sizes.
                        exec_cyc += len(nss)

        return max(num_loads / self.arch.l1.rports, exec_cyc)

    @property
    def read_trace(self):
        yield from self.a[self.li, self.ms, self.ks]
        yield from self.b[self.li, self.ks, self.ns]

@dataclass(frozen=True)
class MatmulTileMKNKI8(MatmulTile):
    tm = 16
    tn = 4
    tk = 64

    def __post_init__(self):
        assert self.op.dtype == Dtype.I8
        assert not self.op.tr_a
        assert self.op.tr_b

    @property
    def exec_lat(self):
        num_loads = 0
        exec_cyc = 0
        for mss in self.ms.subslice(self.tm):
            for nss in self.ns.subslice(self.tn):
                for kss in self.ks.subslice(self.tk):
                    # For each tk chunk, we issue a single VLD4T for A
                    num_loads += M.nloads(
                        self.arch, Dtype.I8, mss, self.op.m, kss, self.op.k)

                    # For MKNK, B is contig. in K, so we just load as many lines
                    # as are needed to cover the kslice we are operating on.
                    num_loads += M.nloads(
                        self.arch,
                        Dtype.I8,
                        kss, self.op.k,
                        nss, self.op.n,
                        transpose=True)

                    exec_cyc += len(nss) * cld(len(kss), 4)

        return max(num_loads / self.arch.l1.rports, exec_cyc)

    @property
    def read_trace(self):
        yield from self.a[self.li, self.ms, self.ks]
        yield from self.b[self.li, self.ns, self.ks]


@dataclass(frozen=True)
class MatmulTileMKNKFP16(MatmulTile):
    tm = 16
    tn = 4
    tk = 32

    def __post_init__(self):
        assert self.op.dtype == Dtype.FP16
        assert not self.op.tr_a
        assert self.op.tr_b

    @property
    def exec_lat(self):
        num_loads = 0
        exec_cyc = 0
        for mss in self.ms.subslice(self.tm):
            for nss in self.ns.subslice(self.tn):
                for kss in self.ks.subslice(self.tk):
                    # For each tk chunk, we issue a single VLD4T for A
                    num_loads += M.nloads(
                        self.arch, Dtype.FP16, mss, self.op.m, kss, self.op.k)

                    # For MKNK, B is contig. in K, so we just load as many lines
                    # as are needed to cover the kslice we are operating on.
                    num_loads += M.nloads(
                        self.arch,
                        Dtype.FP16,
                        kss, self.op.k,
                        nss, self.op.n,
                        transpose=True)

                    exec_cyc += len(nss) * cld(len(kss), 2)

        return max(num_loads / self.arch.l1.rports, exec_cyc)

    @property
    def read_trace(self):
        yield from self.a[self.li, self.ms, self.ks]
        yield from self.b[self.li, self.ns, self.ks]

@dataclass(frozen=True)
class MatmulTileKMKNFP16(MatmulTile):
    tm = 16
    tn = 1
    tk = 2
    ttk = 1

    def __post_init__(self):
        assert self.op.dtype == Dtype.FP16
        assert self.op.tr_a
        assert not self.op.tr_b

    @property
    def exec_lat(self):
        num_loads = 0
        exec_cyc = 0
        for mss in self.ms.subslice(self.tm):
            for nss in self.ns.subslice(self.tn):
                for kss in self.ks.subslice(self.tk):
                    # for A=KM, we can just do normal loads into VRF
                    num_loads += M.nloads(
                        self.arch,
                        Dtype.FP16,
                        mss, self.op.m,
                        kss, self.op.k,
                        transpose=True)

                    # We further subtile k by another factor (ttk) and issue a
                    # VLB4X4 from B for each ttk chunk
                    for ksss in kss.subslice(self.ttk):
                        num_loads += M.nloads(
                            self.arch, Dtype.FP16, ksss, self.op.k, nss, self.op.n)

                        # The number of vector FMAs issued is the product of
                        # m and n slice sizes.
                        exec_cyc += len(nss)

        return max(num_loads / self.arch.l1.rports, exec_cyc)

    @property
    def read_trace(self):
        yield from self.a[self.li, self.ks, self.ms]
        yield from self.b[self.li, self.ks, self.ns]

@dataclass(frozen=True)
class MatmulTileKMNKFP16(MatmulTile):
    tm = 16
    tn = 4
    tk = 4
    ttk = 1

    def __post_init__(self):
        assert self.op.dtype == Dtype.FP16
        assert self.op.tr_a
        assert self.op.tr_b

    @property
    def exec_lat(self):
        num_loads = 0
        exec_cyc = 0
        for mss in self.ms.subslice(self.tm):
            for nss in self.ns.subslice(self.tn):
                for kss in self.ks.subslice(self.tk):
                    # for A=KM, we can just do normal loads into VRF
                    num_loads += M.nloads(
                        self.arch,
                        Dtype.FP16,
                        mss, self.op.m,
                        kss, self.op.k,
                        transpose=True)

                    # We further subtile k by another factor (ttk) and issue a
                    # VLB4X4 from B for each ttk chunk
                    for ksss in kss.subslice(self.ttk):
                        num_loads += M.nloads(
                            self.arch,
                            Dtype.FP16,
                            ksss, self.op.k,
                            nss, self.op.n,
                            transpose=True)

                        # The number of vector FMAs issued is the product of
                        # m and n slice sizes.
                        exec_cyc += len(nss)

        return max(num_loads / self.arch.l1.rports, exec_cyc)

    @property
    def read_trace(self):
        yield from self.a[self.li, self.ks, self.ms]
        yield from self.b[self.li, self.ks, self.ns]

def flatmap_matmul(arch : Arch, mm : Matmul, sim : M.SimBase, a, b, c, bbox=None, offset=0):
    tile = {
        (False, False, Dtype.I8): MatmulTileMKKNI8,
        (False, True, Dtype.I8): MatmulTileMKNKI8,
        (False, False, Dtype.FP16): MatmulTileMKKNFP16,
        (False, True, Dtype.FP16): MatmulTileMKNKFP16,
        (True, False, Dtype.FP16): MatmulTileKMKNFP16,
        (True, True, Dtype.FP16): MatmulTileKMNKFP16,
    }[(mm.tr_a, mm.tr_b, mm.dtype)]

    sim.flatmap_place([
        [
            tile(arch, mm, [a, b], [c], False, li, bm1, bn1, bk1)
            for bm1 in bm0.subslice(tile.tm * 2)
            for bk1 in bk0.subslice(tile.tk * 2)
            for bn1 in bn0.subslice(tile.tn * 2)
        ]
        for bm0 in Slice(0, mm.m).blkslice(64)
        for bn0 in Slice(0, mm.n).blkslice(32)
        for bk0 in Slice(0, mm.k).blkslice(1)
        for li in Slice(0, mm.l).indices
    ])

@M.register_placement('flatmap', [OracleArch, BgroupArch, FbcastArch, HierArch], [Matmul, Linear])
def place_matmul_flatmap(arch : Arch, mm : Matmul, sim : M.SimBase):
    l = mm.l
    m = mm.m
    n = mm.n
    k = mm.k

    a = M.Tensor(arch, 1, mm.dtype, (l, m, k) if not mm.tr_a else (l, k, m))
    b = M.Tensor(arch, 2, mm.dtype, (l, k, n) if not mm.tr_b else (l, n, k))
    c = M.Tensor(arch, 3, mm.dtype, (l, m, n))

    flatmap_matmul(arch, mm, sim, a, b, c)


@M.register_placement('pg', [OracleArch, BgroupArch, FbcastArch, HierArch], [Matmul, Linear])
def place_matmul_profiled(arch : Arch, mm : Matmul, sim : M.SimBase):
    return profiled_placement(arch, mm, sim, place_matmul_flatmap)

@M.register_placement('flatmap', [OracleArch, BgroupArch, FbcastArch, HierArch], [MatmulBwd, LinearBwd])
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

@M.register_placement('pg', [OracleArch, BgroupArch, FbcastArch, HierArch], [MatmulBwd, LinearBwd])
def place_matmul_bwd_profiled(arch : Arch, mm : Matmul, sim : M.SimBase):
    return profiled_placement(arch, mm, sim, place_matmul_bwd_flatmap)
