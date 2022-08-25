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
