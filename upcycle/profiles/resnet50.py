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
    ops.Conv(None, None, None, (14, 14), Slice(256, 1024), (14, 14), Slice(256, 1024), (3, 3), 1, None, None))
@M.register_placement(
    [OracleArch, BgroupArch, FbcastArch, HierArch],
    ops.Conv(None, None, None, (14, 14), Slice(256, 1024), (7, 7), Slice(256, 1024), (3, 3), 2, None, None))
@M.register_placement(
    [OracleArch, BgroupArch, FbcastArch, HierArch],
    ops.Conv(None, None, None, (7, 7), Slice(256, 1024), (7, 7), Slice(256, 1024), (3, 3), 1, None, None))
def resnet50_small_spatial(arch : Arch, conv : ops.Conv, sim : M.SimBase):
    ins, outs = conv.make_tensors(arch)
    tile = ops.conv.choose_fwd_tile(arch, conv)

    def inner_loop(bp, bq, bn, bk):
        return (
            tile(arch, conv, ins, outs, False, ns, (bp1, bq1), bc1, bk2)
            for bp1 in bp.subslice(tile.tp * 2)
            for bq1 in bq.subslice(tile.tq * 2)
            for ns in bn.subslice(1)
            for bk2 in bk.subslice(tile.tk)
            for bc1 in Slice(0, conv.c).subslice(tile.tc)
        )

    sim.map2d_place([
        [
            inner_loop(bp0, bq0, bn0, bk1)
            for bk1 in bk0.blkslice(4)
            for bq0 in Slice(0, conv.so[1]).blkslice(cld(arch.ncols, 64))
            for bn0 in Slice(0, conv.n).blkslice(16)
        ]
        for bk0 in Slice(0, conv.k).blkslice(4)
        for bp0 in Slice(0, conv.so[0]).blkslice(cld(arch.nrows, 4))
    ])

@M.register_placement(
    [OracleArch, BgroupArch, FbcastArch, HierArch, CoarseOracle],
    ops.Matmul(None, None, 1, Slice(2**12, 2**32), Slice(1, 1024), Slice(1, 1024), None, None))
def place_convdi_matmul(arch : Arch, mm : ops.Matmul, sim : M.SimBase):
    ins, outs = mm.make_tensors(arch)
    tile = ops.matmul.choose_tile(arch, mm)

    def inner_loop(bm):
        return (
            tile(arch, mm, ins, outs, False, li, bm2, bn1, bk1)
            for li in Slice(0, mm.l).indices
            for bk1 in Slice(0, mm.k).subslice(tile.tk)
            for bm2 in bm.subslice(tile.tm * 16)
            for bn1 in Slice(0, mm.n).subslice(tile.tn * 8)
        )

    sim.map2d_place([
        [
            inner_loop(bm1)
            for bm1 in bm0.blkslice(64)
        ]
        for bm0 in Slice(0, mm.m).blkslice(32)
    ])

@M.register_placement(
    [OracleArch, BgroupArch, FbcastArch, HierArch, CoarseOracle],
    ops.Matmul(None, None, 1, Slice(1, 1024), Slice(1, 1024), Slice(2**12, 2**32), True, True))
def place_convdw_matmul(arch : Arch, mm : ops.Matmul, sim : M.SimBase):
    ins, outs = mm.make_tensors(arch)
    tile = ops.matmul.choose_tile(arch, mm)

    rkblk = 1
    while rkblk * cld(mm.m, tile.tm) < 32: rkblk <<= 1

    ckblk = 1
    while ckblk * cld(mm.n, tile.tn) < 64: ckblk <<= 1

    def inner_loop(bm, bn, bk):
        return (
            tile(arch, mm, ins, outs, False, li, bm1, bn1, bk2)
            for li in Slice(0, mm.l).indices
            for bk2 in bk.subslice(tile.tk * 4)
            for bm1 in bm.subslice(tile.tm * 4)
            for bn1 in bn.subslice(tile.tn * 4)
        )

    sim.map2d_place([
        [
            inner_loop(bm0, bn0, bk1)
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
    ops.ConvDw(None, None, None, (None, None), None, (None, None), None, (3, 3), None, None, None, None))
def place_conv2d_dw_3x3(arch : Arch, conv : ops.ConvDw, sim : M.SimBase):
    ins, outs = conv.make_tensors(arch)
    tile = ops.conv.choose_dw_tile(arch, conv)
    rnblk, cnblk = blk2d(conv.n)

    def inner_loop(br, bs, bn, bk):
        return (
            tile(
                arch, conv, ins, outs, False,
                ns, (ps, qs), (br, bs), ks, cs)

            for ns in bn.subslice(tile.tn)
            for ps in Slice(0, conv.so[0]).subslice(tile.tp)
            for qs in Slice(0, conv.so[1]).subslice(tile.tq * 2)
            for cs in Slice(0, conv.c).subslice(tile.tc * 4)
            for ks in bk.subslice(tile.tk * 4)
        )

    sim.map2d_place([
        [
            inner_loop(br0, bs0, bn1, bk1)
            for bn1 in bn0.blkslice(cnblk)
            for bs0 in Slice(0, conv.sf[1]).blkslice(3)
            for bk1 in bk0.blkslice(4)
        ]
        for bn0 in Slice(0, conv.n).blkslice(rnblk)
        for br0 in Slice(0, conv.sf[0]).blkslice(3)
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
    ops.ConvDw(None, None, None, (None, None), None, (None, None), None, (7, 7), None, None, None, None))
def place_conv2d_dw_7x7(arch : Arch, conv : ops.ConvDw, sim : M.SimBase):
    ins, outs = conv.make_tensors(arch)
    tile = ops.conv.choose_dw_tile(arch, conv)
    rnblk, cnblk = blk2d(conv.n)

    def inner_loop(br, bs, bn, bk):
        return (
            tile(
                arch, conv, ins, outs, False,
                ns, (ps, qs), (br, bs), ks, cs)

            for ns in bn.subslice(tile.tn)
            for ps in Slice(0, conv.so[0]).subslice(tile.tp)
            for qs in Slice(0, conv.so[1]).subslice(tile.tq)
            for cs in Slice(0, conv.c).subslice(tile.tc)
            for ks in bk.subslice(tile.tk)
        )

    sim.map2d_place([
        [
            inner_loop(br0, bs0, bn1, bk1)
            for bn1 in bn0.blkslice(4)
            for bs0 in Slice(0, conv.sf[1]).blkslice(cnblk)
            for bk1 in bk0.blkslice(4)
        ]
        for bn0 in Slice(0, conv.n).blkslice(rnblk)
        for br0 in Slice(0, conv.sf[0]).blkslice(4)
        for bk0 in Slice(0, conv.k).blkslice(2)
    ])

    sim.barrier()

    M.place_op(
        arch,
        ops.Reduce(conv.dtype, False, rnblk * cnblk, len(outs[0])),
        sim,
        check_flops=False)
