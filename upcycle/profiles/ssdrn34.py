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
    ops.Conv2D(None, None, None, 150, 150, 256, 150, 150, 256, 3, 3, 1, None, None))
def place_ssdrn34_l17(arch : Arch, conv : ops.Conv2D, sim : M.SimBase):
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
                for bp1 in bp0.subslice(tile.tp)
                for bq1 in bq0.subslice(tile.tq)
                for ni in bn0.indices
                for bk2 in bk1.subslice(tile.tk)
                for bc1 in Slice(0, conv.c).subslice(tile.tc)
            ]
            for bk1 in bk0.blkslice(4)
            for bq0 in Slice(0, conv.q).blkslice(4)
            for bn0 in Slice(0, conv.n).blkslice(4)
        ]
        for bk0 in Slice(0, conv.k).blkslice(4)
        for bp0 in Slice(0, conv.p).blkslice(cld(arch.nrows, 4))
    ])

@M.register_placement( # Layer 17
    [OracleArch, BgroupArch, FbcastArch, HierArch],
    ops.Conv2D(None, None, 1, 150, 150, 256, 150, 150, 256, 3, 3, 1, None, None))
def place_ssdrn34_l17_b1(arch : Arch, conv : ops.Conv2D, sim : M.SimBase):
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
                for bp1 in bp0.subslice(tile.tp)
                for bq1 in bq0.subslice(tile.tq)
                for ni in bn0.indices
                for bk2 in bk1.subslice(tile.tk)
                for bc1 in Slice(0, conv.c).subslice(tile.tc)
            ]
            for bk1 in bk0.blkslice(4)
            for bq0 in Slice(0, conv.q).blkslice(16)
            for bn0 in Slice(0, conv.n).blkslice(1)
        ]
        for bk0 in Slice(0, conv.k).blkslice(4)
        for bp0 in Slice(0, conv.p).blkslice(cld(arch.nrows, 4))
    ])

@M.register_placement( # Layer 17
    [OracleArch, BgroupArch, FbcastArch, HierArch],
    ops.Conv2D(None, None, 16, 38, 38, 256, 38, 38, 256, 3, 3, 1, None, None, False))
def place_ssdrn34_300_l17_b16(arch : Arch, conv : ops.Conv2D, sim : M.SimBase):
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
                for bp1 in bp0.subslice(tile.tp)
                for bq1 in bq0.subslice(tile.tq * 2)
                for ni in bn1.indices
                for bk2 in bk1.subslice(tile.tk)
                for bc1 in Slice(0, conv.c).subslice(tile.tc)
            ]
            for bk1 in bk0.blkslice(2)
            for bn1 in bn0.blkslice(4)
            for bq0 in Slice(0, conv.q).blkslice(8)
        ]
        for bk0 in Slice(0, conv.k).blkslice(2)
        for bn0 in Slice(0, conv.n).blkslice(4)
        for bp0 in Slice(0, conv.p).blkslice(4)
    ])


@M.register_placement( # Layer 17
    [OracleArch, BgroupArch, FbcastArch, HierArch],
    ops.Conv2D(None, None, 16, 38, 38, 256, 38, 38, 256, 3, 3, 1, None, None, True))
def place_ssdrn34_300_l95_b16(arch : Arch, conv : ops.Conv2D, sim : M.SimBase):
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
                for bp1 in bp0.subslice(tile.tp)
                for bq1 in bq0.subslice(tile.tq * 2)
                for ni in bn1.indices
                for bk2 in bk1.subslice(tile.tk * 2)
                for bc1 in Slice(0, conv.c).subslice(tile.tc * 2)
            ]
            for bk1 in bk0.blkslice(2)
            for bn1 in bn0.blkslice(4)
            for bq0 in Slice(0, conv.q).blkslice(8)
        ]
        for bk0 in Slice(0, conv.k).blkslice(2)
        for bn0 in Slice(0, conv.n).blkslice(4)
        for bp0 in Slice(0, conv.p).blkslice(4)
    ])
