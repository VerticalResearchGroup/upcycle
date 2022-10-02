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
    ins, outs = mm.make_tensors(arch)
    tile = ops.matmul.choose_tile(arch, mm)
    rkblk, ckblk = blk2d(mm.k)

    sim.map2d_place([
        [
            (
                tile(arch, mm, ins, outs, False, li, bm1, bn2, bk2)
                for li in Slice(0, mm.l).indices
                for bm1 in Slice(0, mm.m).blkslice(1)
                for bn2 in bn1.subslice(tile.tn * 4)
                for bk2 in bk1.subslice(tile.tk)
            )
            for bn1 in bn0.blkslice(8)
            for bk1 in bk0.blkslice(ckblk)
        ]
        for bn0 in Slice(0, mm.n).blkslice(8)
        for bk0 in Slice(0, mm.k).blkslice(rkblk)
    ])

    sim.barrier()

    M.place_op(
        arch,
        ops.Reduce(mm.dtype, False, rkblk * ckblk, len(outs[0])),
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
    ins, outs = mm.make_tensors(arch)
    tile = ops.matmul.choose_tile(arch, mm)
    rkblk, ckblk = blk2d(mm.k)

    sim.map2d_place([
        [
            (
                tile(arch, mm, ins, outs, False, li, bm1, bn1, bk2)
                for li in bl1.indices
                for bm1 in bm0.subslice(tile.tm * 4)
                for bn1 in bn0.subslice(tile.tn * 4)
                for bk2 in bk1.subslice(tile.tk)
            )
            for bl1 in bl0.blkslice(1)
            for bn0 in Slice(0, mm.n).blkslice(16)
            for bk1 in bk0.blkslice(ckblk)
        ]
        for bl0 in Slice(0, mm.l).blkslice(1)
        for bm0 in Slice(0, mm.m).blkslice(8)
        for bk0 in Slice(0, mm.k).blkslice(rkblk)
    ])

    sim.barrier()

    M.place_op(
        arch,
        ops.Reduce(mm.dtype, False, rkblk * ckblk, len(outs[0])),
        sim,
        check_flops=False)
