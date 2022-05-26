from dataclasses import dataclass

from .common import *

backward_map = {}

def register_backward(for_class):
    def decorator(x):
        global backward_map
        if for_class not in backward_map:
            backward_map[for_class] = []

        backward_map[for_class].append(x)
        return x
    return decorator

@dataclass
class Operator:
    dtype : Dtype
    train : bool

    @property
    def flops(self): raise NotImplementedError()

@dataclass
class Matmul(Operator):
    l : int
    m : int
    n : int
    k : int
    tr_a : bool
    tr_b : bool

    @property
    def flops(self): return self.l * self.m * self.n * self.k * 2

@dataclass
@register_backward(Matmul)
class MatmulDa(Matmul):
    @staticmethod
    def from_forward(mm : Matmul):
        return MatmulDa(mm.dtype, False, mm.l, mm.m, mm.n, mm.k, mm.tr_a, mm.tr_b)

@dataclass
@register_backward(Matmul)
class MatmulDb(Matmul):
    @staticmethod
    def from_forward(mm : Matmul):
        return MatmulDb(mm.dtype, False, mm.l, mm.m, mm.n, mm.k, mm.tr_a, mm.tr_b)

@dataclass
class Conv2D(Operator):
    n : int
    h : int
    w : int
    c : int
    p : int
    q : int
    k : int
    r : int
    s : int
    stride : int

    @property
    def flops(self):
        return self.p * self.q * self.k * self.r * self.s * self.c * 2

@dataclass
@register_backward(Conv2D)
class Conv2DDi(Conv2D):
    @property
    def flops(self): raise NotImplementedError()

    @staticmethod
    def from_forward(c : Conv2D):
        return Conv2DDi(c.dtype, False, c.n, c.h, c.w, c.c, c.p, c.q, c.k, c.r, c.s, c.stride)

@dataclass
@register_backward(Conv2D)
class Conv2DDw(Conv2D):
    @property
    def flops(self): raise NotImplementedError()

    @staticmethod
    def from_forward(c : Conv2D):
        return Conv2DDw(c.dtype, False, c.n, c.h, c.w, c.c, c.p, c.q, c.k, c.r, c.s, c.stride)

@dataclass
class Lstm(Operator):
    n : int
    s : int
    d : int
    h : int

    @property
    def flops(self): raise NotImplementedError('LSTM needs this')
