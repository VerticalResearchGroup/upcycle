from dataclasses import dataclass

from ..common import *
from .common import *

from . import matmul

@dataclass(frozen=True)
class Lstm(Operator):
    n : int
    s : int
    d : int
    h : int

    @property
    def flops(self):
        return self.s * sum([
            matmul.Linear(self.dtype, False, 1, self.n, self.h * 4, self.d, False, False).flops, # Xt*W
            matmul.Linear(self.dtype, False, 1, self.n, self.h * 4, self.h, False, False).flops, # Ht*U
            # TODO: Other ops not counted yet (They seem to be insignificant though)
        ])
