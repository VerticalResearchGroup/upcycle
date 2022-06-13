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
    def flops(self):
        flops = 0
        for hi in range(self.h):
            for wi in range(self.w):
                ho = hi % self.stride
                wo = wi % self.stride

                hr = np.ceil((self.r - ho) / self.stride)
                wr = np.ceil((self.s - wo) / self.stride)

                flops += hr * wr * self.c * self.k * 2

        return flops * self.n

    @staticmethod
    def from_forward(c : Conv3D):
        return Conv3DBwd(c.dtype, False, c.n, c.h, c.w, c.d, c.c, c.p, c.q, c.o, c.k, c.r, c.s, c.t, c.stride, c.tr_w)

@dataclass(frozen=True)
class Conv2DTile(M.WorkItemPerfectCompute):
    conv : Conv3D
    i : M.Tensor
    w : M.Tensor
    o : M.Tensor
    write : bool
    ni : int
    ps : Slice
    qs : Slice
    cs : Slice
    ks : Slice
    tr_w : bool


    @property
    def flops(self):
        return \
            len(self.ps) * \
            len(self.qs) * \
            len(self.ks) * \
            len(self.cs) * \
            self.conv.r * self.conv.s * 2

    @property
    def read_trace(self):
        st = self.conv.stride
        yield from self.i[self.ni, self.ps * st, self.qs * st, self.cs]

        if not self.tr_w:
            yield from self.w[:, :, self.ks, self.cs]
        else:
            yield from self.w[:, :, self.cs, self.ks]

    @property
    def write_trace(self):
        if not self.write: return
        raise NotImplementedError()

def make_conv3d_tensors(conv : Conv3D):
    ti = M.Tensor(
        1,
        conv.dtype,
        (conv.n, conv.h + 2 * conv.pad, conv.w + 2 * conv.pad, conv.d + 2 * conv.pad, conv.c))

    tw = M.Tensor(
        2,
        conv.dtype,
        (conv.r, conv.s, conv.t, conv.k, conv.c) if not conv.tr_w \
            else (conv.r, conv.s, conv.t, conv.c, conv.k))

    to = M.Tensor(3, conv.dtype, (conv.n, conv.p, conv.q, conv.o, conv.k))
    return ti, tw, to
