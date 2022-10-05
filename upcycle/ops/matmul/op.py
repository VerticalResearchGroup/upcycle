from dataclasses import dataclass
import logging

from ...common import *
from ..common import *

logger = logging.getLogger(__name__)

@operator
@dataclass(frozen=True)
class Matmul(Operator):
    l : int
    m : int
    n : int
    k : int
    tr_a : bool
    tr_b : bool

    @property
    def flops(self): return self.l * self.m * self.n * self.k * 2

    @property
    def total_read_bytes(self) -> int:
        return self.l * (self.m * self.k + self.n * self.k) * Dtype.sizeof(self.dtype)

    @property
    def total_weight_bytes(self) -> int:
        return self.l * self.n * self.k * Dtype.sizeof(self.dtype)

    @property
    def total_write_bytes(self) -> int:
        return self.l * self.m * self.n * Dtype.sizeof(self.dtype)

    def make_tensors(self, arch : Arch):
        l = self.l
        m = self.m
        n = self.n
        k = self.k
        a = M.Tensor(arch, 1, self.dtype, (l, m, k) if not self.tr_a else (l, k, m))
        b = M.Tensor(arch, 2, self.dtype, (l, k, n) if not self.tr_b else (l, n, k))
        c = M.Tensor(arch, 3, self.dtype, (l, m, n))
        return [a, b], [c]

    @staticmethod
    def layout_str(tr_a, tr_b):
        return {
            (False, False): 'MKKN',
            (False, True): 'MKNK',
            (True, False): 'KMKN',
            (True, True): 'KNMK',
            (None, False): '*KN',
            (None, True): '*NK',
            (False, None): 'MK*',
            (True, None): 'KM*',
            (None, None): '*',
        }[(tr_a, tr_b)]

    def __repr__(self):
        return f'Matmul[{self.dtype}][{self.layout_str(self.tr_a, self.tr_b)}]({self.l} {self.m}x{self.n}x{self.k})'

@operator
@dataclass(frozen=True)
@register_backward(Matmul)
class MatmulDa(Matmul):
    @staticmethod
    def from_forward(mm : Matmul):
        return Matmul(mm.dtype, False, mm.l, mm.m, mm.k, mm.n, mm.tr_a, not mm.tr_b)

@operator
@dataclass(frozen=True)
@register_backward(Matmul)
class MatmulDb(Matmul):
    @staticmethod
    def from_forward(mm : Matmul):
        return Matmul(mm.dtype, False, mm.l, mm.k, mm.n, mm.m, not mm.tr_a, mm.tr_b)

@operator
@dataclass(frozen=True)
class Linear(Matmul): pass

@operator
@dataclass(frozen=True)
@register_backward(Linear)
class LinearDi(MatmulDa): pass

@operator
@dataclass(frozen=True)
@register_backward(Linear)
class LinearDw(MatmulDb): pass


@dataclass(frozen=True)
class MatmulTile(M.WorkItem):
    write : bool
    li : int
    ms : Slice
    ns : Slice
    ks : Slice

    vbits = None
    dtype = None
    tr_a = None
    tr_b = None
    tm = None
    tn = None
    tk = None

    def __post_init__(self):
        assert self.arch.vbits == self.vbits
        assert self.op.dtype == self.dtype
        assert self.op.tr_a == self.tr_a
        assert self.op.tr_b == self.tr_b

    @property
    def a(self): return self.inputs[0]

    @property
    def b(self): return self.inputs[1]

    @property
    def c(self): return self.outputs[0]

    @property
    def flops(self):
        return \
            len(self.ms) * \
            len(self.ns) * \
            len(self.ks) * 2

    @property
    def read_trace(self):
        if not self.tr_a: yield from self.a[self.li, self.ms, self.ks]
        else: yield from self.a[self.li, self.ks, self.ms]

        if not self.tr_b: yield from self.b[self.li, self.ks, self.ns]
        else: yield from self.b[self.li, self.ns, self.ks]

    @property
    def write_trace(self):
        if self.write: yield from self.c[self.li, self.ms, self.ns]

    def intrinsic(self, mss, nss, kss): raise NotImplementedError()

    @property
    def exec_lat(self):
        num_loads = 0
        exec_cyc = 0
        for mss in self.ms.subslice(self.tm):
            for nss in self.ns.subslice(self.tn):
                for kss in self.ks.subslice(self.tk):
                    inner_loads, inner_cyc = self.intrinsic(mss, nss, kss)
                    num_loads += inner_loads
                    exec_cyc += inner_cyc

        return max(num_loads / self.arch.l1.rports, exec_cyc)

