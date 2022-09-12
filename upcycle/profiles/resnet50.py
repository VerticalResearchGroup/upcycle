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

@M.register_placement(
    [OracleArch, BgroupArch, FbcastArch, HierArch, CoarseOracle],
    ops.Matmul(None, None, 1, Slice(2**12, 2**32), Slice(1, 1024), Slice(1, 1024), False, False))
def place_convdi_matmul(arch : Arch, mm : ops.Matmul, sim : M.SimBase):
    l = mm.l
    m = mm.m
    n = mm.n
    k = mm.k

    a = M.Tensor(arch, 1, mm.dtype, (l, m, k) if not mm.tr_a else (l, k, m))
    b = M.Tensor(arch, 2, mm.dtype, (l, k, n) if not mm.tr_b else (l, n, k))
    c = M.Tensor(arch, 3, mm.dtype, (l, m, n))

    tile = ops.matmul.MatmulTileMKKNFP16
    assert not mm.tr_a and not mm.tr_b

    sim.map2d_place([
        [
            [
                tile(arch, mm, [a, b], [c], False, li, bm2, bn1, bk1)
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
    l = mm.l
    m = mm.m
    n = mm.n
    k = mm.k

    a = M.Tensor(arch, 1, mm.dtype, (l, m, k) if not mm.tr_a else (l, k, m))
    b = M.Tensor(arch, 2, mm.dtype, (l, k, n) if not mm.tr_b else (l, n, k))
    c = M.Tensor(arch, 3, mm.dtype, (l, m, n))

    tile = ops.matmul.MatmulTileKMNKFP16
    assert mm.tr_a and mm.tr_b

    sim.map2d_place([
        [
            [
                tile(arch, mm, [a, b], [c], False, li, bm1, bn1, bk2)
                for li in Slice(0, mm.l).indices
                for bk2 in bk1.subslice(tile.tk * 4)
                for bm1 in bm0.subslice(tile.tm * 4)
                for bn1 in bn0.subslice(tile.tn * 4)
            ]
            for bk1 in bk0.blkslice(8)
            for bn0 in Slice(0, mm.n).blkslice(8)
        ]
        for bk0 in Slice(0, mm.k).blkslice(8)
        for bm0 in Slice(0, mm.m).blkslice(4)
    ])

    sim.barrier()

    M.place_op(
        arch,
        ops.Reduce(mm.dtype, False, 64, len(c)),
        sim,
        check_flops=False)

@M.register_placement(
    [OracleArch, BgroupArch, FbcastArch, HierArch],
    ops.Conv2DDw(None, None, None, None, None, None, None, None, None, 3, 3, None, None, None, None))
def place_conv2d_dw_3x3(arch : Arch, conv : ops.Conv2DDw, sim : M.SimBase):
    ti, tdw, tdo = ops.conv2d.make_conv2d_tensors(arch, conv)
    tile = ops.conv2d_dw.Conv2DDwTile
    assert conv.dtype == Dtype.FP16
    assert conv.tr_w == False

    sim.map2d_place([
        [
            [
                tile(
                    arch, conv, [ti, tdo], [tdw], False,
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
        ops.Reduce(conv.dtype, False, 16, len(tdw)),
        sim,
        check_flops=False)

@M.register_placement(
    [OracleArch, BgroupArch, FbcastArch, HierArch],
    ops.Conv2DDw(None, None, None, None, None, None, None, None, None, 7, 7, None, None, None, None))
def place_conv2d_dw_7x7(arch : Arch, conv : ops.Conv2DDw, sim : M.SimBase):
    ti, tdw, tdo = ops.conv2d.make_conv2d_tensors(arch, conv)
    tile = ops.conv2d_dw.Conv2DDwTile
    assert conv.dtype == Dtype.FP16
    assert conv.tr_w == False

    sim.map2d_place([
        [
            [
                tile(
                    arch, conv, [ti, tdo], [tdw], False,
                    ns, ps, qs, br0, bs0, ks, cs)

                for ns in bn1.subslice(tile.tn)
                for ps in Slice(0, conv.p).subslice(tile.tp)
                for qs in Slice(0, conv.q).subslice(tile.tq)
                for cs in Slice(0, conv.c).subslice(tile.tc)
                for ks in bk1.subslice(tile.tk)
            ]
            for bn1 in bn0.blkslice(4)
            for bs0 in Slice(0, conv.s).blkslice(4)
            for bk1 in bk0.blkslice(4)
        ]
        for bn0 in Slice(0, conv.n).blkslice(4)
        for br0 in Slice(0, conv.r).blkslice(4)
        for bk0 in Slice(0, conv.k).blkslice(2)
    ])

    sim.barrier()

    M.place_op(
        arch,
        ops.Reduce(conv.dtype, False, 16, len(tdw)),
        sim,
        check_flops=False)
