from dataclasses import dataclass
import logging

from ..common import *
from .common import *


logger = logging.getLogger(__name__)

@operator
@dataclass(frozen=True)
class Embedding(Operator):
    # N.B. In this model we elide the fact that not all selected vectors get
    # summed to the same destination. This shouldn't make too much of a
    # difference since the embedding reduction is typically so few ops it won't
    # make a big difference.

    n : int
    l : int
    d : int
    tps : float # Tokens per sample as a fraction of total embedding table size
    op : str = 'sum'

    @property
    def min_llc_capacity(self) -> int: return 0

    @property
    def flops(self):
        # Assuming a summation, there's n vectors of size d that need to be
        # added element-wise.
        assert self.op == 'sum'
        return self.n * self.d


@operator
@dataclass(frozen=True)
@register_backward(Embedding)
class EmbeddingBwd(Embedding):
    @property
    def flops(self): return super().flops * 2

    @staticmethod
    def from_forward(l : Embedding):
        return EmbeddingBwd(l.dtype, False, l.n, l.l, l.d, l.op)
