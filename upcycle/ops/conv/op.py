from dataclasses import dataclass
import numpy as np
import logging
import functools

from ...common import *
from ..common import *

logger = logging.getLogger(__name__)

@operator
@dataclass(frozen=True)
class Conv(Operator):
    """Convolution Operator."""
    # Batch Size
    n : int

    # Input: (n, ..si.. , c)
    si : tuple[int]
    c : int

    # Output: (n, ..so.. , k)
    so : tuple[int]
    k : int

    # Filter: (..sf.. , k, c)
    sf : tuple[int]

    stride : int
    pad : int

    # Whether or not the weight tensor is assumed to be transposed in C and K
    # dimensions. This essentially determines the layout of the small matrix
    # multiplies that occur.
    tr_w : bool = False

    # Whether or not the weight is assumed to be rotated in R and S dimensions.
    # I.e. when this is True, the layout of W is (..-sf.. , k, c).
    #
    # N.B. This is functionally unused since the algorithm doesn't care what
    # spatial direction the weight is accessed in, but I still include it here
    # for clarity.
    rot_w : bool = False

    @functools.cached_property
    def inspatial(self): return np.prod(self.si)

    @functools.cached_property
    def outspatial(self): return np.prod(self.so)

    @functools.cached_property
    def filsize(self): return np.prod(self.sf)

    @functools.cached_property
    def d(self): return len(self.si)

    @property
    def flops(self):
        return self.n * self.outspatial * self.filsize * self.k * self.c * 2

    @property
    def total_read_bytes(self) -> int:
        return (self.n * self.inspatial * self.c + self.filsize * self.k * self.c) * Dtype.sizeof(self.dtype)

    @property
    def total_weight_bytes(self) -> int:
        return self.filsize * self.k * self.c * Dtype.sizeof(self.dtype)

    @property
    def total_write_bytes(self) -> int:
        return self.n * self.outspatial * self.k * Dtype.sizeof(self.dtype)

    def make_tensors(self, arch : Arch):
        ti = M.Tensor(arch, 1, self.dtype, (self.n, *tuple(d + 2 * self.pad for d in self.si), self.c))
        tw = M.Tensor(arch, 2, self.dtype, (*self.sf, self.k, self.c))
        to = M.Tensor(arch, 3, self.dtype, (self.n, *self.so, self.k))
        return [ti, tw], [to]

    def __repr__(self):
        r_str = 'R' if self.rot_w else ''
        t_str = '^T' if self.tr_w else ''
        return f'Conv{len(self.si) if self.si is not None else None}D[{self.dtype}](n={self.n}, i={self.si}x{self.c}+{self.pad} {r_str}w{t_str}={self.sf}x{self.k}x{self.c} o={self.so}x{self.k} by {self.stride})'

@operator
@dataclass(frozen=True)
@register_backward(Conv)
class ConvDi(Conv):
    @staticmethod
    def from_forward(c : Conv):
        return ConvDi(c.dtype, False, c.n, c.si, c.c, c.so, c.k, c.sf, c.stride, c.pad, c.tr_w)

    @property
    def total_read_bytes(self) -> int:
        return (self.n * self.outspatial * self.k + self.filsize * self.k * self.c) * Dtype.sizeof(self.dtype)

    @property
    def total_weight_bytes(self) -> int:
        return self.filsize * self.k * self.c * Dtype.sizeof(self.dtype)

    @property
    def total_write_bytes(self) -> int:
        return self.n * self.inspatial * self.c * Dtype.sizeof(self.dtype)

    def make_tensors(self, arch : Arch):
        [tdi, tw], [tdo] = super().make_tensors(arch)
        return [tw, tdo], [tdi]

    def __repr__(self): return f'dI[{super().__repr__()}]'

@operator
@dataclass(frozen=True)
@register_backward(Conv)
class ConvDw(Conv):
    @staticmethod
    def from_forward(c : Conv):
        return ConvDw(c.dtype, False, c.n, c.si, c.c, c.so, c.k, c.sf, c.stride, c.pad, c.tr_w)

    @property
    def total_read_bytes(self) -> int:
        return (self.n * self.inspatial * self.c + self.n * self.outspatial * self.k) * Dtype.sizeof(self.dtype)

    @property
    def total_weight_bytes(self) -> int:
        return self.n * self.inspatial * self.c * Dtype.sizeof(self.dtype)

    @property
    def total_write_bytes(self) -> int:
        return self.filsize * self.k * self.c * Dtype.sizeof(self.dtype)

    def make_tensors(self, arch : Arch):
        [ti, tdw], [tdo] = super().make_tensors(arch)
        return [tdo, ti], [tdw]

    def __repr__(self): return f'dW[{super().__repr__()}]'
