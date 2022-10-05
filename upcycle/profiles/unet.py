from dataclasses import dataclass
import numpy as np
import logging

from ..common import *
from ..arch import *
from .. import model as M
from .. import ops

logger = logging.getLogger(__name__)

@M.register_placement(
    [OracleArch, BgroupArch, FbcastArch, HierArch], [
        ops.Conv(None, None, None, (128, 128, 128), None, (128, 128, 128), None, (3, 3, 3), 1, None, None, None),
        ops.Conv(None, None, None, (32, 32, 32), None, (32, 32, 32), None, (3, 3, 3), 1, None, None, None),
        ops.Conv(None, None, None, (16, 16, 16), None, (16, 16, 16), None, (3, 3, 3), 1, None, None, None),
        ops.Conv(None, None, None, (16, 16, 16), None, (8, 8, 8), None, (3, 3, 3), 1, None, None, None),
        ops.Conv(None, None, None, (8, 8, 8), None, (8, 8, 8), None, (3, 3, 3), 1, None, None, None),
    ])
def place_unet_conv3d(arch : Arch, conv : ops.Conv, sim : M.SimBase):
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

# Layer 432/1287: dW[Conv3D(n=16, i=(32, 32, 32)x32+1 w=(3, 3, 3)x32x32 o=(32, 32, 32)x32 by 1)], 2097478.0 cyc (14.08 %) (AmI = 862.58),  33 TOP/s  (Efficiency: 10.55 %)
# Layer 434/1287: dW[Conv3D(n=16, i=(32, 32, 32)x64+1 w=(3, 3, 3)x32x64 o=(32, 32, 32)x32 by 1)], 4194634.0 cyc (14.72 %) (AmI = 862.58),  33 TOP/s  (Efficiency: 10.55 %)
@M.register_placement(
    [OracleArch, BgroupArch, FbcastArch, HierArch],
    [ops.ConvDw(None, None, None, (32, 32, 32), 32, (32, 32, 32), 32, (3, 3, 3), None, None, None, None),
    ops.ConvDw(None, None, None, (32, 32, 32), None, (32, 32, 32), None, (3, 3, 3), 1, None, None, None),
    ops.ConvDw(None, None, None, (32, 32, 32), 64, (32, 32, 32), 32, (3, 3, 3), None, None, None, None),
    ops.ConvDw(None, None, None, (32, 32, 32), 64, (32, 32, 32), 64, (3, 3, 3), None, None, None, None),
    ops.ConvDw(None, None, None, (32, 32, 32), 128, (32, 32, 32), 64, (3, 3, 3), None, None, None, None)])
def place_conv3d_dw_unet(arch : Arch, conv : ops.ConvDw, sim : M.SimBase):
    ins, outs = conv.make_tensors(arch)
    tile = ops.conv.choose_dw_tile(arch, conv)

    def inner_loop(br, bs, bt, bn, bk):
        return (
            tile(
                arch, conv, ins, outs, False,
                ns, (os, ps, qs), (br, bs, bt), ks, cs)

            for ns in bn.subslice(tile.tn)
            for ps in Slice(0, conv.so[0]).subslice(tile.to * 4)
            for qs in Slice(0, conv.so[1]).subslice(tile.tp * 2)
            for os in Slice(0, conv.so[2]).subslice(tile.tq * 4)
            for cs in Slice(0, conv.c).subslice(tile.tc * 4)
            for ks in bk.subslice(tile.tk * 32)
        )

    sim.map2d_place([
        [
            inner_loop(br0, bs1, bt1, bn1, bk1)
            for bn1 in bn0.blkslice(1)
            for bk1 in bk0.blkslice(2)
            for br0 in Slice(0, conv.sf[0]).blkslice(3)
            for bs1 in Slice(0, conv.sf[1]).blkslice(3)
            for bt1 in Slice(0, conv.sf[2]).blkslice(3)
        ]
        for bn0 in Slice(0, conv.n).blkslice(16)
        for bk0 in Slice(0, conv.k).blkslice(2)
    ])

    sim.barrier()

    M.place_op(
        arch,
        ops.Reduce(conv.dtype, False, min(16, conv.n), len(outs[0])),
        sim,
        check_flops=False)