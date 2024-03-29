from dataclasses import dataclass
import numpy as np
import logging
import functools

from ...common import *
from ..common import *

from .fwd import *
from .di import *
from .dw import *

from .fwd_notp import *
from .di_notp import *
from .dw_notp import *

#
# Conv Fwd
#

def choose_fwd_tile(arch : Arch, op : Conv):
    return {
        (True, 256, Dtype.I8,   False): ConvTile256I8KC,
        (True, 256, Dtype.FP16, False): ConvTile256FP16KC,
        (True, 256, Dtype.FP16, True):  ConvTile256FP16CK,

        (True, 512, Dtype.I8,   False): ConvTile512I8KC,
        (True, 512, Dtype.FP16, False): ConvTile512FP16KC,
        (True, 512, Dtype.FP16, True):  ConvTile512FP16CK,

        (True, 1024, Dtype.I8,   False): ConvTile1024I8KC,
        (True, 1024, Dtype.FP16, False): ConvTile1024FP16KC,
        (True, 1024, Dtype.FP16, True):  ConvTile1024FP16CK,

        (False, 256, Dtype.I8,   False): ConvTile256I8KC_NT,
        (False, 256, Dtype.FP16, False): ConvTile256FP16KC_NT,
        (False, 256, Dtype.FP16, True):  ConvTile256FP16CK_NT,

        (False, 512, Dtype.I8,   False): ConvTile512I8KC_NT,
        (False, 512, Dtype.FP16, False): ConvTile512FP16KC_NT,
        (False, 512, Dtype.FP16, True):  ConvTile512FP16CK_NT,

        (False, 1024, Dtype.I8,   False): ConvTile1024I8KC_NT,
        (False, 1024, Dtype.FP16, False): ConvTile1024FP16KC_NT,
        (False, 1024, Dtype.FP16, True):  ConvTile1024FP16CK_NT,
    }[(arch.tpeng, arch.vbits, op.dtype, op.tr_w)]

@M.register_placement(
    [OracleArch, BgroupArch, FbcastArch, HierArch],
    Conv(None, None, None, (None, None), None, (None, None), None, (None, None), None, None, None, None))
def place_conv2d_default(arch : Arch, conv : Conv, sim : M.SimBase):
    ins, outs = conv.make_tensors(arch)
    tile = choose_fwd_tile(arch, conv)
    logger.debug(f'Conv tile: {tile}')

    def inner_loop(ns, ps, qs, ks):
        return (
            tile(arch, conv, ins, outs, False, ns, (ps, qs), bc1, ks)
            for bc1 in Slice(0, conv.c).subslice(tile.tc * 2)
        )

    sim.flatmap_place((
        inner_loop(ns, bp0, bq0, bk0)
        for ns in Slice(0, conv.n).subslice(1)
        for bp0 in Slice(0, conv.so[0]).subslice(tile.tp)
        for bq0 in Slice(0, conv.so[1]).subslice(tile.tq)
        for bk0 in Slice(0, conv.k).subslice(tile.tk * 2)
    ))

@M.register_placement(
    [OracleArch, BgroupArch, FbcastArch, HierArch],
    Conv(None, None, None, (None, None, None), None, (None, None, None), None, (None, None, None), None, None, None, None))
def place_conv3d_default(arch : Arch, conv : Conv, sim : M.SimBase):
    ins, outs = conv.make_tensors(arch)
    tile = choose_fwd_tile(arch, conv)
    logger.debug(f'Conv tile: {tile}')

    def inner_loop(os, ps, qs, ks):
        return (
            tile(arch, conv, ins, outs, False, ns, (os, ps, qs), bc1, ks)
            for ns in Slice(0, conv.n).subslice(1)
            for bc1 in Slice(0, conv.c).subslice(tile.tc)
        )

    sim.flatmap_place((
        inner_loop(bo0, bp0, bq0, bk0)
        for bo0 in Slice(0, conv.so[0]).subslice(tile.to)
        for bp0 in Slice(0, conv.so[1]).subslice(tile.tp)
        for bq0 in Slice(0, conv.so[2]).subslice(tile.tq)
        for bk0 in Slice(0, conv.k).subslice(tile.tk)
    ))

@M.register_placement(
    [OracleArch, BgroupArch, FbcastArch, HierArch],
    [Conv(None, None, None, None, None, None, None, (1, 1), None, None, None, None),
    Conv(None, None, None, None, None, None, None, (1, 1, 1), None, None, None, None)])
def place_conv_1x1x1(arch : Arch, conv : ConvDi, sim : M.SimBase):
    # We observe in this case the convolution degenerates into a large matmul.
    mm = matmul.Matmul(
        conv.dtype,
        conv.fwd,
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
        (True, 256, Dtype.FP16, False): ConvDiTile256FP16KC,
        (True, 512, Dtype.FP16, False): ConvDiTile512FP16KC,
        (True, 1024, Dtype.FP16, False): ConvDiTile1024FP16KC,

        (False, 256, Dtype.FP16, False): ConvDiTile256FP16KC_NT,
        (False, 512, Dtype.FP16, False): ConvDiTile512FP16KC_NT,
        (False, 1024, Dtype.FP16, False): ConvDiTile1024FP16KC_NT,
    }[(arch.tpeng, arch.vbits, op.dtype, op.tr_w)]

@M.register_placement(
    [OracleArch, BgroupArch, FbcastArch, HierArch],
    ConvDi(None, None, None, (None, None), None, (None, None), None, (None, None), None, None, None, None))
def place_conv2ddi_default(arch : Arch, conv : ConvDi, sim : M.SimBase):
    ins, outs = conv.make_tensors(arch)
    tile = choose_di_tile(arch, conv)

    def inner_loop(bn, bh, bw):
        return (
            tile(arch, conv, ins, outs, False, ns, (hs, ws), ks, cs)
            for ns in bn.subslice(1)
            for hs in bh.subslice(conv.stride)
            for ws in bw.subslice(conv.stride)
            for cs in Slice(0, conv.c).subslice(tile.tc * 2)
            for ks in Slice(0, conv.k).subslice(tile.tk * 4)
        )

    sim.map2d_place([
        [
            inner_loop(bn1, bh0, bw0)
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
    tile = choose_di_tile(arch, conv)

    def inner_loop(bn, bh, bw):
        return (
            tile(arch, conv, ins, outs, False, ns, (hs, ws, ds), ks, cs)
            for ns in bn.subslice(1)
            for hs in bh.subslice(conv.stride)
            for ws in bw.subslice(conv.stride)
            for ds in Slice(0, conv.si[2]).subslice(conv.stride)
            for cs in Slice(0, conv.c).subslice(tile.tc * 2)
            for ks in Slice(0, conv.k).subslice(tile.tk * 4)
        )

    sim.map2d_place([
        [
            inner_loop(bn1, bh0, bw0)
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
        int(np.floor((conv.si[0] - conv.so[0] + conv.sf[0] - 1) / 2)),
        not conv.tr_w, True)

    # N.B. In the case that the output is smaller than the input (filter size >
    # 1), we will end up computing more flops using the forward pass than
    # actually needed. This will make the flop counts mismatch.
    #
    # assert new_conv.flops == conv.flops, new_conv
    return M.place_op(arch, new_conv, sim, False)

@M.register_placement(
    [OracleArch, BgroupArch, FbcastArch, HierArch],
    [ConvDi(None, None, None, None, None, None, None, (1, 1), None, None, None, None),
    ConvDi(None, None, None, None, None, None, None, (1, 1, 1), None, None, None, None)])
def place_convdi_1x1x1(arch : Arch, conv : ConvDi, sim : M.SimBase):
    # We observe in this case the convolution degenerates into a large matmul.
    mm = matmul.MatmulDa.from_forward(matmul.Matmul(
        conv.dtype,
        conv.fwd,
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
        (True, 256, Dtype.FP16, False): ConvDwTile256FP16KC,
        (True, 512, Dtype.FP16, False): ConvDwTile512FP16KC,
        (True, 1024, Dtype.FP16, False): ConvDwTile1024FP16KC,

        (False, 256, Dtype.FP16, False): ConvDwTile256FP16KC_NT,
        (False, 512, Dtype.FP16, False): ConvDwTile512FP16KC_NT,
        (False, 1024, Dtype.FP16, False): ConvDwTile1024FP16KC_NT,
    }[(arch.tpeng, arch.vbits, op.dtype, op.tr_w)]

@M.register_placement(
    [OracleArch, BgroupArch, FbcastArch, HierArch],
    ConvDw(None, None, None, (None, None), None, (None, None), None, (None, None), None, None, None, None))
def place_conv2d_dw_default(arch : Arch, conv : ConvDw, sim : M.SimBase):
    ins, outs = conv.make_tensors(arch)
    tile = choose_dw_tile(arch, conv)
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
    tile = choose_dw_tile(arch, conv)

    rrblk = maxpow2(conv.sf[0])
    csblk = maxpow2(conv.sf[1])
    ctblk = maxpow2(conv.sf[2])

    rnblk = arch.nrows // rrblk // 2
    cnblk = arch.ncols // csblk // ctblk // 2

    logger.debug(f'rrblk={rrblk} csblk={csblk} ctblk={ctblk} rnblk={rnblk} cnblk={cnblk}')

    def inner_loop(br, bs, bt, bn, bk):
        return (
            tile(
                arch, conv, ins, outs, False,
                ns, (os, ps, qs), (br, bs, bt), ks, cs)

            for ns in bn.subslice(tile.tn)
            for ps in Slice(0, conv.so[0]).subslice(tile.tp * 2)
            for qs in Slice(0, conv.so[1]).subslice(tile.tq * 2)
            for os in Slice(0, conv.so[2]).subslice(tile.to * 2)
            for cs in Slice(0, conv.c).subslice(tile.tc * 32)
            for ks in bk.subslice(tile.tk * 32)
        )

    sim.map2d_place([
        [
            inner_loop(br0, bs1, bt1, bn1, bk1)
            for bn1 in bn0.blkslice(cnblk)
            for bs1 in Slice(0, conv.sf[1]).blkslice(csblk)
            for bt1 in Slice(0, conv.sf[2]).blkslice(ctblk)
            for bk1 in bk0.blkslice(2)
        ]
        for bn0 in Slice(0, conv.n).blkslice(rnblk)
        for br0 in Slice(0, conv.sf[0]).blkslice(rrblk)
        for bk0 in Slice(0, conv.k).blkslice(2)
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
        conv.fwd,
        1,
        conv.n * conv.outspatial,
        conv.k,
        conv.c,
        False,
        not conv.tr_w))

    assert mm.flops == conv.flops
    return M.place_op(arch, mm, sim, False)
