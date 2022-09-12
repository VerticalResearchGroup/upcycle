from dataclasses import dataclass
import numpy as np
import logging

from ..common import *
from ..arch import *
from .. import model as M
from .. import ops

logger = logging.getLogger(__name__)

@M.register_placement(
    [OracleArch, BgroupArch, FbcastArch, HierArch, CoarseOracle],
    ops.Linear(None, None, 1, 1, 4096, 1264, False, True))
@M.register_placement(
    [OracleArch, BgroupArch, FbcastArch, HierArch, CoarseOracle],
    ops.Linear(None, None, 1, 1, 4096, 2048, False, True))
@M.register_placement(
    [OracleArch, BgroupArch, FbcastArch, HierArch, CoarseOracle],
    ops.Linear(None, None, 1, 1, 4096, 3072, False, True))
def rnnt_hu_mm(arch : Arch, mm : ops.Matmul, sim : M.SimBase):
    l = mm.l
    m = mm.m
    n = mm.n
    k = mm.k

    # MKNK

    a = M.Tensor(arch, 1, mm.dtype, (l, m, k))
    b = M.Tensor(arch, 2, mm.dtype, (l, n, k))
    c = M.Tensor(arch, 3, mm.dtype, (l, m, n))

    tile = {
        # (False, False, Dtype.I8): ops.matmul.MatmulTileMKKNI8,
        (False, True, Dtype.I8): ops.matmul.MatmulTileMKNKI8,
        # (False, False, Dtype.FP16): ops.matmul.MatmulTileMKKNFP16,
        (False, True, Dtype.FP16): ops.matmul.MatmulTileMKNKFP16,
        # (True, False, Dtype.FP16): ops.matmul.MatmulTileKMKNFP16,
        # (True, True, Dtype.FP16): ops.matmul.MatmulTileKMNKFP16,
    }[(mm.tr_a, mm.tr_b, mm.dtype)]

    sim.map2d_place([
        [
            [
                tile(arch, mm, [a, b], [c], False, li, bm1, bn2, bk2)
                for li in Slice(0, mm.l).indices
                for bm1 in Slice(0, mm.m).blkslice(1)
                for bn2 in bn1.subslice(tile.tn * 4)
                for bk2 in bk1.subslice(tile.tk)
            ]
            for bn1 in bn0.blkslice(8)
            for bk1 in bk0.blkslice(8)
        ]
        for bn0 in Slice(0, mm.n).blkslice(8)
        for bk0 in Slice(0, mm.k).blkslice(4)
    ])

    sim.barrier()

    M.place_op(
        arch,
        ops.Reduce(mm.dtype, False, 32, len(c)),
        sim,
        check_flops=False)

@M.register_placement(
    [OracleArch, BgroupArch, FbcastArch, HierArch, CoarseOracle],
    ops.Linear(None, None, 1, Slice(2, 65536), 4096, 1264, False, True))
@M.register_placement(
    [OracleArch, BgroupArch, FbcastArch, HierArch, CoarseOracle],
    ops.Linear(None, None, 1, Slice(2, 65536), 4096, 2048, False, True))
@M.register_placement(
    [OracleArch, BgroupArch, FbcastArch, HierArch, CoarseOracle],
    ops.Linear(None, None, 1, Slice(2, 65536), 4096, 3072, False, True))
def rnnt_hu_mm(arch : Arch, mm : ops.Matmul, sim : M.SimBase):
    l = mm.l
    m = mm.m
    n = mm.n
    k = mm.k

    a = M.Tensor(arch, 1, mm.dtype, (l, m, k) if not mm.tr_a else (l, k, m))
    b = M.Tensor(arch, 2, mm.dtype, (l, k, n) if not mm.tr_b else (l, n, k))
    c = M.Tensor(arch, 3, mm.dtype, (l, m, n))

    tile = {
        (False, False, Dtype.I8): ops.matmul.MatmulTileMKKNI8,
        (False, True, Dtype.I8): ops.matmul.MatmulTileMKNKI8,
        (False, False, Dtype.FP16): ops.matmul.MatmulTileMKKNFP16,
        (False, True, Dtype.FP16): ops.matmul.MatmulTileMKNKFP16,
        (True, False, Dtype.FP16): ops.matmul.MatmulTileKMKNFP16,
        (True, True, Dtype.FP16): ops.matmul.MatmulTileKMNKFP16,
    }[(mm.tr_a, mm.tr_b, mm.dtype)]

    sim.map2d_place([
        [
            [
                tile(arch, mm, [a, b], [c], False, li, bm1, bn1, bk2)
                for li in bl1.indices
                for bm1 in bm0.subslice(tile.tm * 4)
                for bn1 in bn0.subslice(tile.tn * 4)
                for bk2 in bk1.subslice(tile.tk)
            ]
            for bl1 in bl0.blkslice(1)
            for bn0 in Slice(0, mm.n).blkslice(16)
            for bk1 in bk0.blkslice(4)
        ]
        for bl0 in Slice(0, mm.l).blkslice(1)
        for bm0 in Slice(0, mm.m).blkslice(8)
        for bk0 in Slice(0, mm.k).blkslice(4)
    ])

    sim.barrier()

    M.place_op(
        arch,
        ops.Reduce(mm.dtype, False, 16, len(c)),
        sim,
        check_flops=False)
