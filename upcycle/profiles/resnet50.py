from dataclasses import dataclass
import numpy as np
import logging

from ..common import *
from .. import model as M
from .. import ops

logger = logging.getLogger(__name__)

def place_conv_pq_spatial(arch : Arch, conv : ops.Conv2D):
    ti, tw, to = ops.conv2d.make_conv2d_tensors(conv)
    wl = M.WorkList.from_arch(arch, [ti, tw, to])

    kgrp = conv.k // 16

    pgrp = conv.p / arch.nrows
    qgrp = conv.q / (arch.ncols / kgrp)

    for n in range(0, conv.n):
        for k in range(0, conv.k, 16):
            for p in range(conv.p):
                row = int(p / pgrp)
                for q in range(conv.q):
                    col = (int(q / qgrp) + (k // 16) * (arch.ncols // kgrp))
                    wl.flatmap_place([
                        [
                            ops.conv2d.Conv2DTile(
                                arch, conv.dtype,
                                conv, ti, tw, to, False, n,
                                Slice.blk(p, conv.p, 1),
                                Slice.blk(q, conv.q, 1),
                                Slice.blk(bc, conv.c, 16),
                                Slice.blk(k, conv.k, 16),
                                conv.tr_w)
                            for bc in range(0, conv.c, 16)
                        ]
                    ], bbox=(row, row + 1, col, col + 1))

    return wl


@ops.placement_profile(
    [OracleArch, BgroupArch],
    ops.Conv2D(None, None, None, 224, 224, 3, 112, 112, 64, 7, 7, 2, None, None))
def place_rn50_l1(arch : Arch, conv : ops.Conv2D):
    return place_conv_pq_spatial(arch, conv)

@ops.placement_profile(
    [OracleArch, BgroupArch],
    ops.Conv2D(None, None, None, 56, 56, 128, 28, 28, 128, 3, 3, 2, None, None))
def place_rn50_l12(arch : Arch, conv : ops.Conv2D):
    return place_conv_pq_spatial(arch, conv)

@ops.placement_profile(
    [OracleArch, BgroupArch],
    ops.Conv2D(None, None, None, 28, 28, 128, 28, 28, 128, 3, 3, 1, None, None))
def place_rn50_l16(arch : Arch, conv : ops.Conv2D):
    return place_conv_pq_spatial(arch, conv)

@ops.placement_profile(
    [OracleArch, BgroupArch],
    ops.Conv2D(None, None, Slice(2, 256), 28, 28, 512, 28, 28, 128, 1, 1, 1, None, None))
def place_rn50_l15b(arch : Arch, conv : ops.Conv2D):
    return place_conv_pq_spatial(arch, conv)

@ops.placement_profile(
    [OracleArch, BgroupArch],
    ops.Conv2D(None, None, Slice(2, 256), 14, 14, 1024, 14, 14, 256, 1, 1, 1, None, None))
def place_rn50_l37b(arch : Arch, conv : ops.Conv2D):
    return place_conv_pq_spatial(arch, conv)

@ops.placement_profile(
    [OracleArch, BgroupArch],
    ops.Conv2D(None, None, Slice(2, 256), 14, 14, 256, 14, 14, 256, 3, 3, 1, None, None))
def place_rn50_l38b(arch : Arch, conv : ops.Conv2D):
    return place_conv_pq_spatial(arch, conv)
