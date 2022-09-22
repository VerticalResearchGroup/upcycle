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
    ins, outs = conv.make_tensors(arch)
    tile = ops.conv2d.choose_tile(conv)

    sim.map2d_place([
        [
            [
                tile(arch, conv, ins, outs, False, ni, bp1, bq1, bc1, bk2)
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

@M.register_placement(
    [OracleArch, BgroupArch, FbcastArch, HierArch, CoarseOracle],
    ops.Matmul(None, None, 1, Slice(2**12, 2**32), Slice(1, 1024), Slice(1, 1024), False, False))
def place_convdi_matmul(arch : Arch, mm : ops.Matmul, sim : M.SimBase):
    ins, outs = mm.make_tensors(arch)
    tile = ops.matmul.choose_tile(mm)

    sim.map2d_place([
        [
            [
                tile(arch, mm, ins, outs, False, li, bm2, bn1, bk1)
                for li in Slice(0, mm.l).indices
                for bk1 in Slice(0, mm.k).subslice(tile.tk * 4)
                for bm2 in bm1.subslice(tile.tm * 4)
                for bn1 in Slice(0, mm.n).subslice(tile.tn * 4)
            ]
            for bm1 in bm0.blkslice(64)
        ]
        for bm0 in Slice(0, mm.m).blkslice(32)
    ])

@M.register_placement(
    [OracleArch, BgroupArch, FbcastArch, HierArch, CoarseOracle],
    ops.Matmul(None, None, 1, Slice(1, 1024), Slice(1, 1024), Slice(2**12, 2**32), True, True))
def place_convdw_matmul(arch : Arch, mm : ops.Matmul, sim : M.SimBase):
    ins, outs = mm.make_tensors(arch)
    tile = ops.matmul.choose_tile(mm)

    rkblk = 1
    while rkblk * cld(mm.m, tile.tm) < 32: rkblk <<= 1

    ckblk = 1
    while ckblk * cld(mm.n, tile.tn) < 64: ckblk <<= 1

    sim.map2d_place([
        [
            [
                tile(arch, mm, ins, outs, False, li, bm1, bn1, bk2)
                for li in Slice(0, mm.l).indices
                for bk2 in bk1.subslice(tile.tk * 4)
                for bm1 in bm0.subslice(tile.tm * 4)
                for bn1 in bn0.subslice(tile.tn * 4)
            ]
            for bk1 in bk0.blkslice(ckblk)
            for bn0 in Slice(0, mm.n).blkslice(cld(mm.n, tile.tn))
        ]
        for bk0 in Slice(0, mm.k).blkslice(rkblk)
        for bm0 in Slice(0, mm.m).blkslice(cld(mm.m, tile.tm))
    ])

    sim.barrier()

    M.place_op(
        arch,
        ops.Reduce(mm.dtype, False, rkblk * ckblk, len(outs[0])),
        sim,
        check_flops=False)

@M.register_placement(
    [OracleArch, BgroupArch, FbcastArch, HierArch],
    ops.Conv2DDw(None, None, None, None, None, None, None, None, None, 3, 3, None, None, None, None))
def place_conv2d_dw_3x3(arch : Arch, conv : ops.Conv2DDw, sim : M.SimBase):
    ins, outs = conv.make_tensors(arch)
    tile = ops.conv2d_dw.choose_tile(conv)
    rnblk, cnblk = blk2d(conv.n)

    sim.map2d_place([
        [
            [
                tile(
                    arch, conv, ins, outs, False,
                    ns, ps, qs, br0, bs0, ks, cs)

                for ns in bn1.subslice(tile.tn)
                for ps in Slice(0, conv.p).subslice(tile.tp)
                for qs in Slice(0, conv.q).subslice(tile.tq)
                for cs in Slice(0, conv.c).subslice(tile.tc)
                for ks in bk1.subslice(tile.tk)
            ]
            for bn1 in bn0.blkslice(4)
            for bs0 in Slice(0, conv.s).blkslice(3)
            for bk1 in bk0.blkslice(4)
        ]
        for bn0 in Slice(0, conv.n).blkslice(4)
        for br0 in Slice(0, conv.r).blkslice(3)
        for bk0 in Slice(0, conv.k).blkslice(2)
    ])

    sim.barrier()

    M.place_op(
        arch,
        ops.Reduce(conv.dtype, False, rnblk * cnblk, len(outs[0])),
        sim,
        check_flops=False)

@M.register_placement(
    [OracleArch, BgroupArch, FbcastArch, HierArch],
    ops.Conv2DDw(None, None, None, None, None, None, None, None, None, 7, 7, None, None, None, None))
def place_conv2d_dw_7x7(arch : Arch, conv : ops.Conv2DDw, sim : M.SimBase):
    ins, outs = conv.make_tensors(arch)
    tile = ops.conv2d_dw.choose_tile(conv)
    rnblk, cnblk = blk2d(conv.n)

    sim.map2d_place([
        [
            [
                tile(
                    arch, conv, ins, outs, False,
                    ns, ps, qs, br0, bs0, ks, cs)

                for ns in bn1.subslice(tile.tn)
                for ps in Slice(0, conv.p).subslice(tile.tp)
                for qs in Slice(0, conv.q).subslice(tile.tq)
                for cs in Slice(0, conv.c).subslice(tile.tc)
                for ks in bk1.subslice(tile.tk)
            ]
            for bn1 in bn0.blkslice(4)
            for bs0 in Slice(0, conv.s).blkslice(cnblk)
            for bk1 in bk0.blkslice(4)
        ]
        for bn0 in Slice(0, conv.n).blkslice(rnblk)
        for br0 in Slice(0, conv.r).blkslice(4)
        for bk0 in Slice(0, conv.k).blkslice(2)
    ])

    sim.barrier()

    M.place_op(
        arch,
        ops.Reduce(conv.dtype, False, rnblk * cnblk, len(outs[0])),
        sim,
        check_flops=False)
