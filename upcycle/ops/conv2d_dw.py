from dataclasses import dataclass
import numpy as np
import logging
import functools

from ..common import *
from .common import *

from . import matmul
from .conv2d import Conv2D, make_conv2d_tensors

logger = logging.getLogger(__name__)


@operator
@dataclass(frozen=True)
@register_backward(Conv2D)
class Conv2DDw(Conv2D):
    @staticmethod
    def from_forward(c : Conv2D):
        return Conv2DDw(c.dtype, False, c.n, c.h, c.w, c.c, c.p, c.q, c.k, c.r, c.s, c.stride, c.pad, c.tr_w)

@dataclass(frozen=True)
class Conv2DDwTile(M.WorkItem):
    write : bool
    # Serial
    ns : Slice
    hs : Slice
    ws : Slice

    # Parallizable
    rs : Slice
    ss : Slice
    ks : Slice
    cs : Slice

    @property
    def i(self): return self.inputs[0]

    @property
    def do(self): return self.inputs[1]

    @property
    def dw(self): return self.outputs[0]

    @functools.cached_property
    def ps(self):
        return Slice(
            (self.hs.start + self.conv.pad) // self.conv.stride,
            self.hs.stop // self.conv.stride)

    @functools.cached_property
    def qs(self):
        return Slice(
            (self.ws.start + self.conv.pad) // self.conv.stride,
            self.ws.stop // self.conv.stride)

    @property
    def read_trace(self):
        yield from self.do[self.ns, self.hs, self.ws, self.ks]
        yield from self.do[self.ns, self.ps, self.qs, self.ks]

    @property
    def write_trace(self):
        if not self.write: return
        raise NotImplementedError()

    # @functools.cached_property
    # def flops(self): return sum(self._gemms) * len(self.cs) * len(self.ks) * 2

    # @functools.cached_property
    # def exec_lat(self):
    #     num_loads = 0
    #     exec_cyc = 0
    #     for n in self._gemms:
    #         _num_loads, _exec_cyc = self._small_gemm(n)
    #         num_loads += _num_loads
    #         exec_cyc += _exec_cyc
    #     lat = max(num_loads / self.arch.l1.rports, exec_cyc)
    #     return lat

@M.register_placement([OracleArch, BgroupArch, FbcastArch, HierArch], Conv2DDw)
def place_conv2d_di_default(arch : Arch, conv : Conv2DDw, sim : M.SimBase):
    tdi, tw, tdo = make_conv2d_tensors(arch, conv)

    tile = {
        (Dtype.FP16, False): Conv2DDwTile,
    }[(conv.dtype, conv.tr_w)]

    pad = conv.pad

    sim.flatmap_place([
        [
            tile(
                arch, conv, [tdo, tw], [tdi], False,
                ni,
                Slice.blk(bh, conv.h + pad, conv.stride),
                Slice.blk(bw, conv.w + pad, conv.stride),
                Slice.blk(bk, conv.k, tile.tk),
                Slice.blk(bc, conv.c, tile.tc))
            for ni in bn.indices
            for bk in range(0, conv.k, tile.tk)
        ]
        for bn in Slice(0, conv.n).subslice(4)
        for bh in range(0, conv.h + pad, conv.stride)
        for bw in range(0, conv.w + pad, conv.stride)
        for bc in range(0, conv.c, tile.tc)
    ])


@M.register_placement(
    [OracleArch, BgroupArch, FbcastArch, HierArch],
    Conv2DDw(None, None, None, None, None, None, None, None, None, 1, 1, None, None, None))
def place_conv2ddw_1x1(arch : Arch, conv : Conv2DDw, sim : M.SimBase):
    # We observe in this case the convolution degenerates into a large matmul.
    mm = matmul.MatmulDa.from_forward(matmul.Matmul(
        conv.dtype,
        conv.train,
        1,
        conv.n * conv.p * conv.q,
        conv.k,
        conv.c,
        False,
        not conv.tr_w))

    assert mm.flops == conv.flops
    return M.place_op(arch, mm, sim, False)
