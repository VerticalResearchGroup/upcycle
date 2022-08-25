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
    ops.Conv2D(None, None, None, 14, 14, Slice(256, 1024), 14, 14, Slice(256, 1024), 3, 3, 1, None, None))
@M.register_placement(
    [OracleArch, BgroupArch, FbcastArch, HierArch],
    ops.Conv2D(None, None, None, 14, 14, Slice(256, 1024), 7, 7, Slice(256, 1024), 3, 3, 2, None, None))
@M.register_placement(
    [OracleArch, BgroupArch, FbcastArch, HierArch],
    ops.Conv2D(None, None, None, 7, 7, Slice(256, 1024), 7, 7, Slice(256, 1024), 3, 3, 1, None, None))
def resnet50_small_spatial(arch : Arch, conv : ops.Conv2D, sim : M.SimBase):
    ti, tw, to = ops.conv2d.make_conv2d_tensors(arch, conv)

    tile = {
        (Dtype.I8, False): ops.conv2d.Conv2DTileI8,
        (Dtype.I8, True): ops.conv2d.Conv2DTileI8TW,
        (Dtype.FP16, False): ops.conv2d.Conv2DTileFP16,
        (Dtype.FP16, True): ops.conv2d.Conv2DTileFP16TW,
    }[(conv.dtype, conv.tr_w)]

    sim.map2d_place([
        [
            [
                tile(arch, conv, [ti, tw], [to], False, ni, bp1, bq1, bc1, bk2)
                for bp1 in bp0.subslice(tile.tp * 2)
                for bq1 in bq0.subslice(tile.tq * 2)
                for ni in bn0.indices
                for bk2 in bk1.subslice(tile.tk)
                for bc1 in Slice(0, conv.c).subslice(tile.tc)
            ]
            for bk1 in bk0.blkslice(4)
            for bq0 in Slice(0, conv.q).blkslice(cld(arch.ncols, 64))
            for bn0 in Slice(0, conv.n).blkslice(16)
        ]
        for bk0 in Slice(0, conv.k).blkslice(4)
        for bp0 in Slice(0, conv.p).blkslice(cld(arch.nrows, 4))
    ])


