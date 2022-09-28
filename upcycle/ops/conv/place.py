from dataclasses import dataclass
import numpy as np
import logging
import functools

from ...common import *
from ..common import *

from .fwd import *
from .di import *
from .dw import *

#
# Conv Fwd
#

def choose_fwd_tile(arch : Arch, op : Conv):
    assert op.d == 2

    return {
        (256, Dtype.I8,   False): ConvTile256I8KC,
        (256, Dtype.FP16, False): ConvTile256FP16KC,
        (256, Dtype.FP16, True):  ConvTile256FP16CK,

        (512, Dtype.I8,   False): ConvTile512I8KC,
        (512, Dtype.FP16, False): ConvTile512FP16KC,
        (512, Dtype.FP16, True):  ConvTile512FP16CK,

        (1024, Dtype.I8,   False): ConvTile1024I8KC,
        (1024, Dtype.FP16, False): ConvTile1024FP16KC,
        (1024, Dtype.FP16, True):  ConvTile1024FP16CK,
    }[(arch.vbits, op.dtype, op.tr_w)]

@M.register_placement(
    [OracleArch, BgroupArch, FbcastArch, HierArch],
    Conv(None, None, None, (None, None), None, (None, None), None, (None, None), None, None, None, None))
def place_conv2d_default(arch : Arch, conv : Conv, sim : M.SimBase):
    ins, outs = conv.make_tensors(arch)
    tile = choose_fwd_tile(arch, conv)
    logger.debug(f'Conv tile: {tile}')

    sim.flatmap_place([
        [
            tile(arch, conv, ins, outs, False, ns, (bp0, bq0), bc1, bk0)
            for bc1 in bc0.subslice(tile.tc * 2)
        ]
        for ns in Slice(0, conv.n).subslice(1)
        for bp0 in Slice(0, conv.so[0]).subslice(tile.tp)
        for bq0 in Slice(0, conv.so[1]).subslice(tile.tq)
        for bk0 in Slice(0, conv.k).subslice(tile.tk * 2)
        for bc0 in Slice(0, conv.c).blkslice(1)
    ])

@M.register_placement(
    [OracleArch, BgroupArch, FbcastArch, HierArch],
    Conv(None, None, None, (None, None, None), None, (None, None, None), None, (None, None, None), None, None, None, None))
def place_conv3d_default(arch : Arch, conv : Conv, sim : M.SimBase):
    ins, outs = conv.make_tensors(arch)
    tile = choose_fwd_tile(arch, conv)
    logger.debug(f'Conv tile: {tile}')

    sim.flatmap_place([
        [
            tile(arch, conv, ins, outs, False, ns, (bo0, bp0, bq0), bc1, bk0)
            for bc1 in bc0.subslice(tile.tc * 2)
        ]
        for ns in Slice(0, conv.n).subslice(1)
        for bo0 in Slice(0, conv.so[0]).subslice(tile.to)
        for bp0 in Slice(0, conv.so[1]).subslice(tile.tp)
        for bq0 in Slice(0, conv.so[2]).subslice(tile.tq * 2)
        for bk0 in Slice(0, conv.k).subslice(tile.tk * 2)
        for bc0 in Slice(0, conv.c).blkslice(1)
    ])

@M.register_placement(
    [OracleArch, BgroupArch, FbcastArch, HierArch],
    [ConvDi(None, None, None, None, None, None, None, (1, 1), None, None, None, None),
    ConvDi(None, None, None, None, None, None, None, (1, 1, 1), None, None, None, None)])
def place_conv_1x1x1(arch : Arch, conv : ConvDi, sim : M.SimBase):
    # We observe in this case the convolution degenerates into a large matmul.
    mm = matmul.Matmul(
        conv.dtype,
        conv.train,
        1,
        conv.n * conv.outspatial,
        conv.k,
        conv.c,
        False,
        not conv.tr_w)

    assert mm.flops == conv.flops
    return M.place_op(arch, mm, sim, False)

#
# Conv Di
#

def choose_di_tile(arch : Arch, op : ConvDi):
    return {
        (256, Dtype.FP16, False): ConvDiTile256FP16KC,
        (512, Dtype.FP16, False): ConvDiTile512FP16KC,
        (1024, Dtype.FP16, False): ConvDiTile1024FP16KC,
    }[(arch.vbits, op.dtype, op.tr_w)]

@M.register_placement(
    [OracleArch, BgroupArch, FbcastArch, HierArch],
    ConvDi(None, None, None, (None, None), None, (None, None), None, (None, None), None, None, None, None))
def place_conv2ddi_default(arch : Arch, conv : ConvDi, sim : M.SimBase):
    ins, outs = conv.make_tensors(arch)
    tile = choose_di_tile(conv)

    sim.map2d_place([
        [
            [
                tile(arch, conv, ins, outs, False, ns, (hs, ws), ks, cs)
                for ns in bn1.subslice(1)
                for hs in bh0.subslice(conv.stride)
                for ws in bw0.subslice(conv.stride)
                for cs in Slice(0, conv.c).subslice(tile.tc * 2)
                for ks in Slice(0, conv.k).subslice(tile.tk * 4)
            ]
            for bn1 in bn0.blkslice(2)
            for bw0 in Slice(conv.pad, conv.si[1] + conv.pad).blkslice(arch.ncols // 2)
        ]
        for bn0 in Slice(0, conv.n).blkslice(2)
        for bh0 in Slice(conv.pad, conv.si[0] + conv.pad).blkslice(arch.nrows // 2)
    ])

@M.register_placement(
    [OracleArch, BgroupArch, FbcastArch, HierArch],
    ConvDi(None, None, None, (None, None, None), None, (None, None, None), None, (None, None, None), None, None, None, None))
def place_conv3ddi_default(arch : Arch, conv : ConvDi, sim : M.SimBase):
    ins, outs = conv.make_tensors(arch)
    tile = choose_di_tile(conv)

    sim.map2d_place([
        [
            [
                tile(arch, conv, ins, outs, False, ns, (hs, ws, ds), ks, cs)
                for ns in bn1.subslice(1)
                for hs in bh0.subslice(conv.stride)
                for ws in bw0.subslice(conv.stride)
                for ds in Slice(0, conv.si[2]).subslice(conv.stride)
                for cs in Slice(0, conv.c).subslice(tile.tc * 2)
                for ks in Slice(0, conv.k).subslice(tile.tk * 4)
            ]
            for bn1 in bn0.blkslice(2)
            for bw0 in Slice(conv.pad, conv.si[1] + conv.pad).blkslice(arch.ncols // 2)
        ]
        for bn0 in Slice(0, conv.n).blkslice(2)
        for bh0 in Slice(conv.pad, conv.si[0] + conv.pad).blkslice(arch.nrows // 2)
    ])

@M.register_placement(
    [OracleArch, BgroupArch, FbcastArch, HierArch],
    ConvDi(None, None, None, None, None, None, None, None, 1, None, None, None))
def place_convdi_stride1(arch : Arch, conv : ConvDi, sim : M.SimBase):
    # N.B. Based on the 2018 paper from Intel, when stride=1, we can reuse the
    # forward prop algorithm for backprop. The weight tensor needs to be rotated
    # spatially by 180 degrees and the input / output channels are transposed.
    #
    # See: Georganas, et al. "Anatomy of High-Performance Deep Learning
    # Convolutions on SIMD Architectures."
    new_conv = Conv(
        conv.dtype, True,
        conv.n,
        conv.so, conv.k,
        conv.si, conv.c,
        conv.sf, conv.stride,
        conv.pad, not conv.tr_w, True)

    assert new_conv.flops == conv.flops
    return M.place_op(arch, new_conv, sim, False)

@M.register_placement(
    [OracleArch, BgroupArch, FbcastArch, HierArch],
    [ConvDi(None, None, None, None, None, None, None, (1, 1), None, None, None, None),
    ConvDi(None, None, None, None, None, None, None, (1, 1, 1), None, None, None, None)])
def place_convdi_1x1x1(arch : Arch, conv : ConvDi, sim : M.SimBase):
    # We observe in this case the convolution degenerates into a large matmul.
    mm = matmul.MatmulDa.from_forward(matmul.Matmul(
        conv.dtype,
        conv.train,
        1,
        conv.n * conv.outspatial,
        conv.k,
        conv.c,
        False,
        not conv.tr_w))

    assert mm.flops == conv.flops
    return M.place_op(arch, mm, sim, False)

#
# Conv Dw
#

def choose_dw_tile(arch : Arch, op : ConvDw):
    return {
        (256, Dtype.FP16, False): ConvDwTile256FP16KC,
        (512, Dtype.FP16, False): ConvDwTile512FP16KC,
        (1024, Dtype.FP16, False): ConvDwTile1024FP16KC,
    }[(arch.vbits, op.dtype, op.tr_w)]

@M.register_placement(
    [OracleArch, BgroupArch, FbcastArch, HierArch],
    ConvDw(None, None, None, (None, None), None, (None, None), None, (None, None), None, None, None, None))
def place_conv2d_dw_default(arch : Arch, conv : ConvDw, sim : M.SimBase):
    ins, outs = conv.make_tensors(arch)
    tile = choose_dw_tile(conv)
    rnblk, cnblk = blk2d(conv.n)

    sim.map2d_place([
        [
            [
                tile(
                    arch, conv, ins, outs, False,
                    ns, (ps, qs), (br0, bs0), ks, cs)

                for ns in bn1.subslice(tile.tn)
                for ps in Slice(0, conv.so[0]).subslice(tile.tp)
                for qs in Slice(0, conv.so[1]).subslice(tile.tq)
                for cs in Slice(0, conv.c).subslice(tile.tc)
                for ks in bk1.subslice(tile.tk)
            ]
            for bn1 in bn0.blkslice(cnblk)
            for bs0 in Slice(0, conv.sf[1]).subslice(1)
            for bk1 in bk0.blkslice(16)
        ]
        for bn0 in Slice(0, conv.n).blkslice(rnblk)
        for br0 in Slice(0, conv.sf[0]).subslice(1)
        for bk0 in Slice(0, conv.k).blkslice(16)
    ])

    sim.barrier()

    M.place_op(
        arch,
        Reduce(conv.dtype, False, rnblk * cnblk, len(outs[0])),
        sim,
        check_flops=False)

@M.register_placement(
    [OracleArch, BgroupArch, FbcastArch, HierArch],
    ConvDw(None, None, None, (None, None, None), None, (None, None, None), None, (None, None, None), None, None, None, None))
def place_conv3d_dw_default(arch : Arch, conv : ConvDw, sim : M.SimBase):
    ins, outs = conv.make_tensors(arch)
    tile = choose_dw_tile(conv)
    rnblk, cnblk = blk2d(conv.n)

    sim.map2d_place([
        [
            [
                tile(
                    arch, conv, ins, outs, False,
                    ns, (os, ps, qs), (br0, bs0, bt0), ks, cs)

                for ns in bn1.subslice(tile.tn)
                for os in Slice(0, conv.so[0]).subslice(tile.to)
                for ps in Slice(0, conv.so[1]).subslice(tile.tp)
                for qs in Slice(0, conv.so[2]).subslice(tile.tq)
                for cs in Slice(0, conv.c).subslice(tile.tc)
                for ks in bk1.subslice(tile.tk)
            ]
            for bn1 in bn0.blkslice(cnblk)
            for bt0 in Slice(0, conv.sf[2]).subslice(1)
            for bk1 in bk0.blkslice(16)
        ]
        for bn0 in Slice(0, conv.n).blkslice(rnblk)
        for br0 in Slice(0, conv.sf[0]).subslice(1)
        for bs0 in Slice(0, conv.sf[1]).subslice(1)
        for bk0 in Slice(0, conv.k).blkslice(8)
    ])

    sim.barrier()

    M.place_op(
        arch,
        Reduce(conv.dtype, False, rnblk * cnblk, len(outs[0])),
        sim,
        check_flops=False)

@M.register_placement(
    [OracleArch, BgroupArch, FbcastArch, HierArch],
    [ConvDw(None, None, None, None, None, None, None, (1, 1), None, None, None, None),
    ConvDw(None, None, None, None, None, None, None, (1, 1, 1), None, None, None, None)])
def place_convdw_1x1x1(arch : Arch, conv : ConvDi, sim : M.SimBase):
    # We observe in this case the convolution degenerates into a large matmul.
    mm = matmul.MatmulDb.from_forward(matmul.Matmul(
        conv.dtype,
        conv.train,
        1,
        conv.n * conv.outspatial,
        conv.k,
        conv.c,
        False,
        not conv.tr_w))

    assert mm.flops == conv.flops
    return M.place_op(arch, mm, sim, False)
