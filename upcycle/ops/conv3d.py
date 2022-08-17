from dataclasses import dataclass
import numpy as np
import logging

from ..common import *
from .common import *

logger = logging.getLogger(__name__)

@operator
@dataclass(frozen=True)
class Conv3D(Operator):
    n : int

    h : int
    w : int
    d : int
    c : int

    p : int
    q : int
    o : int
    k : int

    r : int
    s : int
    t : int
    stride : int
    pad : int
    tr_w : bool

    @property
    def flops(self):
        return self.n * self.p * self.q * self.o * self.k * self.r * self.s * self.t * self.c * 2

@operator
@dataclass(frozen=True)
@register_backward(Conv3D)
class Conv3DBwd(Conv3D):
    @property
    def flops(self): return super().flops * 2

    @staticmethod
    def from_forward(c : Conv3D):
        return Conv3DBwd(c.dtype, False, c.n, c.h, c.w, c.d, c.c, c.p, c.q, c.o, c.k, c.r, c.s, c.t, c.stride, c.pad, c.tr_w)


@dataclass(frozen=True)
class Conv3DTile(M.WorkItem):
    write : bool
    ni : int
    ps : Slice
    qs : Slice
    os : Slice
    cs : Slice
    ks : Slice

    @property
    def i(self): return self.inputs[0]

    @property
    def w(self): return self.inputs[1]

    @property
    def o(self): return self.outputs[0]

    @property
    def flops(self):
        return \
            len(self.ps) * \
            len(self.qs) * \
            len(self.os) * \
            len(self.ks) * \
            len(self.cs) * \
            self.op.r * self.op.s * self.op.t * 2

    @property
    def read_trace(self):
        st = self.op.stride
        yield from self.i[self.ni, self.ps * st, self.qs * st, self.os * st, self.cs]

        if not self.op.tr_w: yield from self.w[:, :, :, self.ks, self.cs]
        else: yield from self.w[:, :, :, self.cs, self.ks]

    @property
    def write_trace(self):
        if not self.write: return
        raise NotImplementedError()


    def nloads_a(self, css, kss): raise NotImplementedError()

    def nloads_b(self, kss, oss): raise NotImplementedError()

    @property
    def exec_lat(self):
        num_loads = 0
        exec_cyc = 0
        rstride = 4 if self.op.dtype is Dtype.I8 else 2
        for _ in self.ps.subslice(self.tp):
            for _ in self.qs.subslice(self.tq):
                for oss in self.os.subslice(self.to):
                    for br in range(self.op.r * self.op.s * self.op.t):
                        for kss in self.ks.subslice(self.tk):
                            for css in self.cs.subslice(self.tc):
                                num_loads += self.nloads_a(css, kss)
                                num_loads += self.nloads_b(kss, oss)
                                exec_cyc += len(oss) * cld(len(css), rstride)

        return max(num_loads / self.arch.l1.rports, exec_cyc)

@dataclass(frozen=True)
class Conv3DTileI8(Conv3DTile):
    tk = 16 # M
    tp = 1
    tq = 1
    to = 4  # N
    tc = 64 # K

    def __post_init__(self):
        assert self.op.dtype == Dtype.I8
        assert not self.op.tr_w

    def nloads_a(self, css, kss):
        return M.nloads(self.arch, Dtype.I8, kss, self.op.k, css, self.op.c)

    def nloads_b(self, kss, oss):
        return M.nloads(
            self.arch,
            Dtype.I8,
            kss, self.op.k,
            oss, self.op.o,
            transpose=True,
            contig=self.op.stride == 1)


@dataclass(frozen=True)
class Conv3DTileFP16(Conv3DTile):
    tk = 16 # M
    tp = 1
    tq = 1
    to = 4  # N
    tc = 32 # K

    def __post_init__(self):
        assert self.op.dtype == Dtype.FP16
        assert not self.op.tr_w

    def nloads_a(self, css, kss):
        return M.nloads(self.arch, Dtype.FP16, kss, self.op.k, css, self.op.c)

    def nloads_b(self, kss, oss):
        return M.nloads(
            self.arch,
            Dtype.FP16,
            kss, self.op.k,
            oss, self.op.o,
            transpose=True,
            contig=self.op.stride == 1)


@dataclass(frozen=True)
class Conv3DTileI8TW(Conv3DTile):
    tk = 16 # M
    tp = 1
    tq = 1
    to = 4  # N
    tc = 4  # K

    def __post_init__(self):
        assert self.op.dtype == Dtype.I8
        assert self.op.tr_w

    def nloads_a(self, css, kss):
        return M.nloads(
            self.arch,
            Dtype.I8,
            kss, self.op.k,
            css, self.op.c,
            transpose=True)

    def nloads_b(self, kss, oss):
        return M.nloads(
            self.arch,
            Dtype.I8,
            kss, self.op.k,
            oss, self.op.o,
            transpose=True,
            contig=self.op.stride == 1)

@dataclass(frozen=True)
class Conv3DTileFP16TW(Conv3DTile):
    tk = 16 # M
    tp = 1
    tq = 1
    to = 4  # N
    tc = 2  # K

    def __post_init__(self):
        assert self.op.dtype == Dtype.FP16
        assert self.op.tr_w

    def nloads_a(self, css, kss):
        return M.nloads()

    def nloads_b(self, kss, oss):
        return M.nloads()

    @property
    def exec_lat(self):
        num_loads = 0
        exec_cyc = 0
        for _ in self.ps.subslice(self.tp):
            for _ in self.qs.subslice(self.tq):
                for oss in self.os.subslice(self.to):
                    for br in range(self.op.r * self.op.s * self.op.t):
                        for kss in self.ks.subslice(self.tk):
                            for css in self.cs.subslice(self.tc):
                                num_loads += M.nloads(
                                    self.arch,
                                    Dtype.I8,
                                    kss, self.op.k,
                                    css, self.op.c,
                                    transpose=True)

                                num_loads += M.nloads(
                                    self.arch,
                                    Dtype.I8,
                                    kss, self.op.k,
                                    oss, self.op.o,
                                    transpose=True,
                                    contig=self.op.stride == 1)

                                exec_cyc += len(oss) * cld(len(css), 2)

        return max(num_loads / self.arch.l1.rports, exec_cyc)


def make_conv3d_tensors(arch : Arch, conv : Conv3D):
    ti = M.Tensor(
        arch,
        1,
        conv.dtype,
        (conv.n, conv.h + 2 * conv.pad, conv.w + 2 * conv.pad, conv.d + 2 * conv.pad, conv.c))

    tw = M.Tensor(
        arch,
        2,
        conv.dtype,
        (conv.r, conv.s, conv.t, conv.k, conv.c) if not conv.tr_w \
            else (conv.r, conv.s, conv.t, conv.c, conv.k))

    to = M.Tensor(arch, 3, conv.dtype, (conv.n, conv.p, conv.q, conv.o, conv.k))
    return ti, tw, to

@M.register_placement('default', [OracleArch, BgroupArch, FbcastArch], Conv3D)
def place_conv3d_default(arch : Arch, conv : Conv3D, sim : M.SimBase):
    ti, tw, to = make_conv3d_tensors(arch, conv)

    tile = {
        (Dtype.I8, False): Conv3DTileI8,
        (Dtype.I8, True): Conv3DTileI8TW,
        (Dtype.FP16, False): Conv3DTileFP16,
        (Dtype.FP16, True): Conv3DTileFP16TW,
    }[(conv.dtype, conv.tr_w)]

    sim.flatmap_place([
        [
            tile(arch, conv, [ti, tw], [to], False, ni, bp0, bq0, bo0, bc1, bk0)
            for bc1 in bc0.subslice(tile.tc)
        ]
        for ni in Slice(0, conv.n).indices
        for bp0 in Slice(0, conv.p).subslice(tile.tp)
        for bq0 in Slice(0, conv.q).subslice(tile.tq)
        for bo0 in Slice(0, conv.o).subslice(tile.to)
        for bk0 in Slice(0, conv.k).subslice(tile.tk)
        for bc0 in Slice(0, conv.c).blkslice(1)
    ])

@M.register_placement('pg', [OracleArch, BgroupArch, FbcastArch], Conv3D)
def place_conv3d_profiled(arch : Arch, conv : Conv3D, sim : M.SimBase):
    return profiled_placement(arch, conv, sim, place_conv3d_default)
