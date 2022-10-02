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
    ops.ConvDw(None, None, None, (128, 128, 128), None, (128, 128, 128), None, (3, 3, 3), 1, None, None, None))
def place_unet_conv3d_dw(arch : Arch, conv : ops.ConvDw, sim : M.SimBase):
    ins, outs = conv.make_tensors(arch)
    tile = ops.conv.choose_dw_tile(arch, conv)

    rrblk = maxpow2(conv.sf[0])
    csblk = maxpow2(conv.sf[1])
    ctblk = maxpow2(conv.sf[2])

    rnblk = 4
    cnblk = 2

    logger.debug(f'rrblk={rrblk} csblk={csblk} ctblk={ctblk} rnblk={rnblk} cnblk={cnblk}')

    sim.map2d_place([
        [
            (
                tile(
                    arch, conv, ins, outs, False,
                    ns, (os, ps, qs), (br0, bs1, bt1), ks, cs)

                for ns in bn1.subslice(tile.tn)
                for ps in Slice(0, conv.so[0]).subslice(tile.tp * 2)
                for qs in Slice(0, conv.so[1]).subslice(tile.tq * 2)
                for os in Slice(0, conv.so[2]).subslice(tile.to * 2)
                for cs in bc0.subslice(tile.tc * 32)
                for ks in bk1.subslice(tile.tk * 32)
            )
            for bn1 in bn0.blkslice(cnblk)
            for bs1 in Slice(0, conv.sf[1]).blkslice(3)
            for bt1 in Slice(0, conv.sf[2]).blkslice(3)
            for bk1 in Slice(0, conv.k).blkslice(2)
        ]
        for bn0 in Slice(0, conv.n).blkslice(rnblk)
        for br0 in Slice(0, conv.sf[0]).blkslice(3)
        for bc0 in Slice(0, conv.c).blkslice(2)
    ])

    sim.barrier()

    M.place_op(
        arch,
        ops.Reduce(conv.dtype, False, rnblk * cnblk, len(outs[0])),
        sim,
        check_flops=False)
