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
        # flops = 0
        # for hi in range(self.h):
        #     for wi in range(self.w):
        #         ho = hi % self.stride
        #         wo = wi % self.stride

        #         hr = np.ceil((self.r - ho) / self.stride)
        #         wr = np.ceil((self.s - wo) / self.stride)

        #         flops += hr * wr * self.c * self.k * 2

        # return flops * self.n

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
    ks : slice

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

@M.register_placement('flatmap', [OracleArch, BgroupArch, FbcastArch], Conv3D)
def place_conv3d_flatmap(arch : Arch, conv : Conv3D, sim : M.SimBase):
    ti, tw, to = make_conv3d_tensors(arch, conv)
    npixels = conv.n * conv.p * conv.q

    if npixels > arch.ntiles: kblk = 128
    elif npixels > arch.ntiles // 4: kblk = 64
    else: kblk = 8

    pblk = 4
    qblk = 4
    oblk = int(max(1, 128 / kblk))
    # nblk = int(max(1, arch.ncols // conv.n))
    off = 0

    for ni in range(0, conv.n):
        off += sim.flatmap_place([
            [
                Conv3DTile(
                    arch, conv, [ti, tw], [to], False,
                    ni,
                    Slice.blk(bp, conv.p, pblk),
                    Slice.blk(bq, conv.q, qblk),
                    Slice.blk(bo, conv.o, oblk),
                    Slice.blk(bc, conv.c, 16),
                    Slice.blk(bk, conv.k, kblk))
                for bc in range(0, conv.c, 16)
            ]
            for bp in range(0, conv.p, pblk)
            for bq in range(0, conv.q, qblk)
            for bo in range(0, conv.o, oblk)
            for bk in range(0, conv.k, kblk)
        ], offset=off, bbox=None, randomize=False)

@M.register_placement('pg', [OracleArch, BgroupArch, FbcastArch], Conv3D)
def place_conv3d_profiled(arch : Arch, conv : Conv3D, sim : M.SimBase):
    return profiled_placement(arch, conv, sim, place_conv3d_flatmap)
