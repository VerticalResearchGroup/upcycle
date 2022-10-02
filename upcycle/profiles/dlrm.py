from dataclasses import dataclass
import numpy as np
import logging

from ..common import *
from ..arch import *
from .. import model as M
from .. import ops

logger = logging.getLogger(__name__)

@M.register_placement( # Layer 17
    [OracleArch, BgroupArch, FbcastArch, HierArch],
    ops.Matmul(None, None, None, 27, 27, 128, None, None))
def place_dlrm_interaction(arch : Arch, mm : ops.Matmul, sim : M.SimBase):
    ins, outs = mm.make_tensors(arch)
    tile = ops.matmul.choose_tile(arch, mm)

    def inner_loop(bl):
        return (
            tile(arch, mm, ins, outs, False, li, bm1, bn1, bk1)
            for li in bl.indices
            for bm1 in Slice(0, mm.m).subslice(tile.tm)
            for bn1 in Slice(0, mm.n).subslice(tile.tn)
            for bk1 in Slice(0, mm.k).blkslice(1)
        )

    sim.map2d_place([
        [
            inner_loop(bl1)
            for bl1 in bl0.blkslice(64)
        ]
        for bl0 in Slice(0, mm.l).blkslice(32)
    ])
