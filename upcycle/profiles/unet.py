from dataclasses import dataclass
import numpy as np
import logging

from ..common import *
from ..arch import *
from .. import model as M
from .. import ops

logger = logging.getLogger(__name__)

@M.register_placement(
    [OracleArch, BgroupArch, FbcastArch, HierArch],
    ops.Conv(None, None, None, (128, 128, 128), None, (128, 128, 128), None, (3, 3, 3), 1, None, None, None))
@M.register_placement(
    [OracleArch, BgroupArch, FbcastArch, HierArch],
    ops.Conv(None, None, None, (32, 32, 32), None, (32, 32, 32), None, (3, 3, 3), 1, None, None, None))
@M.register_placement(
    [OracleArch, BgroupArch, FbcastArch, HierArch],
    ops.Conv(None, None, None, (8, 8, 8), None, (8, 8, 8), None, (3, 3, 3), 1, None, None, None))
def place_unet_conv3d_dw(arch : Arch, conv : ops.Conv, sim : M.SimBase):
    ins, outs = conv.make_tensors(arch)
    tile = ops.conv.choose_fwd_tile(arch, conv)

    logger.debug(f'tile = {tile}')

    def inner_loop(bo, bp, bq, bn, bk):
        return (
            tile(arch, conv, ins, outs, False, ns, (bo1, bp1, bq1), bc1, bk2)
            for bo1 in bo.subslice(tile.to * 2)
            for bp1 in bp.subslice(tile.tp * 2)
            for bq1 in bq.subslice(tile.tq * 2)
            for ns in bn.subslice(1)
            for bk2 in bk.subslice(tile.tk * 4)
            for bc1 in Slice(0, conv.c).subslice(tile.tc * 2)
        )

    sim.map2d_place([
        [
            inner_loop(bo0, bp0, bq0, bn0, bk1)
            for bk1 in bk0.blkslice(2)
            for bp0 in Slice(0, conv.so[1]).blkslice(4)
            for bq0 in Slice(0, conv.so[2]).blkslice(8)
            for bn0 in Slice(0, conv.n).blkslice(1)
        ]
        for bk0 in Slice(0, conv.k).blkslice(4)
        for bo0 in Slice(0, conv.so[0]).blkslice(8)
    ])
