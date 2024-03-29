from dataclasses import dataclass
import numpy as np
import logging

from ..common import *
from ..arch import *
from .. import model as M
from .. import ops

from .resnet50 import resnet50_small_spatial

logger = logging.getLogger(__name__)

@M.register_placement( # Layer 17
    [OracleArch, BgroupArch, FbcastArch, HierArch],
    ops.Conv(None, None, None, (150, 150), 256, (150, 150), 256, (3, 3), 1, None, None))
def place_ssdrn34_l17(arch : Arch, conv : ops.Conv, sim : M.SimBase):
    ins, outs = conv.make_tensors(arch)
    tile = ops.conv.choose_fwd_tile(arch, conv)

    def inner_loop(bn, bp, bq, bk):
        return (
            tile(arch, conv, ins, outs, False, ns, (bp1, bq1), bc1, bk2)
            for bp1 in bp.subslice(tile.tp)
            for bq1 in bq.subslice(tile.tq)
            for ns in bn.subslice(1)
            for bk2 in bk.subslice(tile.tk)
            for bc1 in Slice(0, conv.c).subslice(tile.tc)
        )

    sim.map2d_place([
        [
            inner_loop(bn0, bp0, bq0, bk1)
            for bk1 in bk0.blkslice(4)
            for bq0 in Slice(0, conv.so[1]).blkslice(4)
            for bn0 in Slice(0, conv.n).blkslice(4)
        ]
        for bk0 in Slice(0, conv.k).blkslice(4)
        for bp0 in Slice(0, conv.so[0]).blkslice(cld(arch.nrows, 4))
    ])

@M.register_placement( # Layer 17
    [OracleArch, BgroupArch, FbcastArch, HierArch],
    ops.Conv(None, None, 1, (150, 150), 256, (150, 150), 256, (3, 3), 1, None, None))
def place_ssdrn34_l17_b1(arch : Arch, conv : ops.Conv, sim : M.SimBase):
    ins, outs = conv.make_tensors(arch)
    tile = ops.conv.choose_fwd_tile(arch, conv)

    def inner_loop(bn, bp, bq, bk):
        return (
            tile(arch, conv, ins, outs, False, ns, (bp1, bq1), bc1, bk2)
            for bp1 in bp.subslice(tile.tp)
            for bq1 in bq.subslice(tile.tq)
            for ns in bn.subslice(1)
            for bk2 in bk.subslice(tile.tk)
            for bc1 in Slice(0, conv.c).subslice(tile.tc)
        )

    sim.map2d_place([
        [
            inner_loop(bn0, bp0, bq0, bk1)
            for bk1 in bk0.blkslice(4)
            for bq0 in Slice(0, conv.so[1]).blkslice(16)
            for bn0 in Slice(0, conv.n).blkslice(1)
        ]
        for bk0 in Slice(0, conv.k).blkslice(4)
        for bp0 in Slice(0, conv.so[0]).blkslice(cld(arch.nrows, 4))
    ])

@M.register_placement( # Layer 17
    [OracleArch, BgroupArch, FbcastArch, HierArch],
    ops.Conv(None, None, 16, (38, 38), 256, (38, 38), 256, (3, 3), 1, None, None, False))
def place_ssdrn34_300_l17_b16(arch : Arch, conv : ops.Conv, sim : M.SimBase):
    ins, outs = conv.make_tensors(arch)
    tile = ops.conv.choose_fwd_tile(arch, conv)

    def inner_loop(bn, bp, bq, bk):
        return (
            tile(arch, conv, ins, outs, False, ns, (bp1, bq1), bc1, bk2)
            for bp1 in bp.subslice(tile.tp)
            for bq1 in bq.subslice(tile.tq * 2)
            for ns in bn.subslice(1)
            for bk2 in bk.subslice(tile.tk)
            for bc1 in Slice(0, conv.c).subslice(tile.tc)
        )

    sim.map2d_place([
        [
            inner_loop(bn1, bp0, bq0, bk1)
            for bk1 in bk0.blkslice(2)
            for bn1 in bn0.blkslice(4)
            for bq0 in Slice(0, conv.so[1]).blkslice(8)
        ]
        for bk0 in Slice(0, conv.k).blkslice(2)
        for bn0 in Slice(0, conv.n).blkslice(4)
        for bp0 in Slice(0, conv.so[0]).blkslice(4)
    ])


@M.register_placement( # Layer 17
    [OracleArch, BgroupArch, FbcastArch, HierArch],
    ops.Conv(None, None, 16, (38, 38), 256, (38, 38), 256, (3, 3), 1, None, None, True))
def place_ssdrn34_300_l95_b16(arch : Arch, conv : ops.Conv, sim : M.SimBase):
    ins, outs = conv.make_tensors(arch)
    tile = ops.conv.choose_fwd_tile(arch, conv)

    def inner_loop(bn, bp, bq, bk):
        return (
            tile(arch, conv, ins, outs, False, ns, (bp1, bq1), bc1, bk2)
            for bp1 in bp.subslice(tile.tp)
            for bq1 in bq.subslice(tile.tq * 2)
            for ns in bn.subslice(1)
            for bk2 in bk.subslice(tile.tk * 2)
            for bc1 in Slice(0, conv.c).subslice(tile.tc * 2)
        )

    sim.map2d_place([
        [
            inner_loop(bn1, bp0, bq0, bk1)
            for bk1 in bk0.blkslice(2)
            for bn1 in bn0.blkslice(4)
            for bq0 in Slice(0, conv.so[1]).blkslice(8)
        ]
        for bk0 in Slice(0, conv.k).blkslice(2)
        for bn0 in Slice(0, conv.n).blkslice(4)
        for bp0 in Slice(0, conv.so[0]).blkslice(4)
    ])

@M.register_placement(
    [OracleArch, BgroupArch, FbcastArch, HierArch],
    ops.Conv(None, None, None, (3, 3), Slice(128, 1024 + 1), (5, 5), Slice(128, 1024 + 1), (3, 3), 1, None, None, None))
def place_ssdrn34_dw_small_spatial(arch : Arch, conv : ops.Conv, sim : M.SimBase):
    ins, outs = conv.make_tensors(arch)
    tile = ops.conv.choose_fwd_tile(arch, conv)

    def inner_loop(bp, bq, bn, bk):
        return (
            tile(arch, conv, ins, outs, False, ns, (bp1, bq1), bc1, bk2)
            for bp1 in bp.subslice(tile.tp)
            for bq1 in bq.subslice(tile.tq)
            for ns in bn.subslice(1)
            for bk2 in bk.subslice(tile.tk * 4)
            for bc1 in Slice(0, conv.c).subslice(tile.tc * 4)
        )

    sim.map2d_place([
        [
            inner_loop(bp0, bq0, bn1, bk1)
            for bk1 in bk0.blkslice(2)
            for bq0 in Slice(0, conv.so[1]).subslice(1)
            for bn1 in bn0.blkslice(8)
        ]
        for bk0 in Slice(0, conv.k).blkslice(2)
        for bp0 in Slice(0, conv.so[0]).subslice(1)
        for bn0 in Slice(0, conv.n).blkslice(2)
    ])

