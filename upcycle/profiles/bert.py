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
    ops.Matmul(None, None, Slice(1, 1024), Slice(1, 1024), 178, 64, None, None))
@M.register_placement(
    [OracleArch, BgroupArch, FbcastArch, HierArch, CoarseOracle],
    ops.Matmul(None, None, Slice(1, 1024), Slice(1, 1024), 64, 178, None, None))
def bert_l171_l172(arch : Arch, mm : ops.Matmul, sim : M.SimBase):
    ins, outs = mm.make_tensors(arch)
    tile = ops.matmul.choose_tile(arch, mm)

    def inner_loop(bl, bm, bn):
        return (
            tile(arch, mm, ins, outs, False, li, bm1, bn1, bk1)
            for li in bl.indices
            for bm1 in bm.subslice(tile.tm * 4)
            for bn1 in bn.subslice(tile.tn * 4)
            for bk1 in Slice(0, mm.k).blkslice(1)
        )

    sim.map2d_place([
        [
            inner_loop(bl1, bm0, bn0)
            for bl1 in bl0.blkslice(4)
            for bn0 in Slice(0, mm.n).blkslice(16)
        ]
        for bl0 in Slice(0, mm.l).blkslice(4)
        for bm0 in Slice(0, mm.m).blkslice(8)
    ])

@M.register_placement(
    [OracleArch, BgroupArch, FbcastArch, HierArch, CoarseOracle],
    ops.Linear(None, None, 1, Slice(127, 513), 1024, 1024, False, True))
def bert_l0_b1(arch : Arch, mm : ops.Matmul, sim : M.SimBase):
    ins, outs = mm.make_tensors(arch)
    tile = ops.matmul.choose_tile(arch, mm)

    def inner_loop(bm, bn):
        return (
            tile(arch, mm, ins, outs, False, li, bm1, bn2, bk1)
            for li in Slice(0, mm.l).indices
            for bk1 in Slice(0, mm.k).subslice(tile.tk)
            for bm1 in bm.subslice(tile.tm * 2)
            for bn2 in bn.subslice(tile.tn * 2)
        )

    sim.map2d_place([
        [
            inner_loop(bm0, bn1)
            for bn1 in bn0.blkslice(64)
        ]
        for bm0 in Slice(0, mm.m).blkslice(16)
        for bn0 in Slice(0, mm.n).blkslice(2)
    ])

