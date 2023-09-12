from dataclasses import dataclass, field
from enum import IntEnum
import functools
import numpy as np
import random
import logging

logger = logging.getLogger(__name__)

# "Aggressive memory" is intended to refer to a set of behaviors that attempt to
# reduce RAM usage as much as possible. This is to mitigate the risk of running
# out of memory for certain NN layers which seem to eat up a lot of RAM.
_aggressive_mem = False

def enable_aggressive_mem():
    global _aggressive_mem
    _aggressive_mem = True

def disable_aggressive_mem():
    global _aggressive_mem
    _aggressive_mem = False

def aggressive_mem():
    global _aggressive_mem
    return _aggressive_mem

# Adapted from https://stackoverflow.com/questions/2536307/decorators-in-the-python-standard-lib-deprecated-specifically
def deprecated(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        logger.warning(f'Call to deprecated function {func.__name__} (in {func.__module__}).')
        return func(*args, **kwargs)
    return wrapper

def blkdiv(n, b):
    """Divide n into b blocks of equal size.

    If n is not divisible by b, some blocks will be smaller by 1.
    """
    nn = n // b
    for i in range(b):
        r = nn + (1 if i < n % b else 0)
        if r > 0: yield r

def blkdivn(n, b):
    """Divide n into b blocks of equal size.

    If n is not divisible by b, some blocks will be smaller by 1.
    """
    nn = n // b
    nfull = n % b
    nunderfull = b - nfull
    return nfull, nn + 1, nunderfull, nn


@dataclass(frozen=True, order=True)
class Slice:
    """A custom slice object.

    This class represents a 1D strided list of indices similar to the build-in
    Python slice class. This class adds some additional functionality beyond
    what Python slice's provide such as the ability to scale the range, or check
    inclusion. It is used in many places, such as providing a nice way to index
    into Tensors in a compact way.
    """
    start : int
    stop : int
    step : int = 1

    @staticmethod
    def from_pyslice(s : slice, n : int) -> 'Slice':
        return Slice(
            s.start if s.start is not None else 0,
            s.stop if s.stop is not None else n,
            s.step if s.step is not None else 1)

    @staticmethod
    def blk(start : int, n : int, blk : int):
        return Slice(start, start + min(n - start, blk), 1)

    def __mul__(self, c : int):
        return Slice(self.start * c, self.stop * c, self.step)

    def __add__(self, c : int):
        return Slice(self.start + c, self.stop + c, self.step)

    def _div(self, c : int):
        return Slice(self.start // c, int(np.ceil(self.stop / c)), self.step)

    def __div__(self, c : int): return self._div(c)
    def __truediv__(self, c : int): return self._div(c)
    def __floordiv__(self, c : int): return self._div(c)
    def __contains__(self, i : int): return self.start <= i < self.stop

    def __len__(self):
        return (self.stop - self.start) // self.step

    @property
    def indices(self): yield from range(self.start, self.stop, self.step)

    def __iter__(self): yield from self.indices

    def subslice(self, chunksize):
        assert self.step == 1
        for i in range(self.start, self.stop, chunksize):
            s = Slice(i, min(self.stop, i + chunksize), 1)
            if len(s) > 0: yield s

    def blkslice(self, nblks):
        i = self.start
        for n in blkdiv(len(self), nblks):
            s = Slice(i, i + n, 1)
            if len(s) > 0: yield s
            i += n

        assert i == self.stop

    @property
    def blocks(self):
        for i in range(self.start, self.stop, self.step):
            yield i // self.step, i, min(i + self.step, self.stop)

    def __repr__(self):
        if self.step > 1:
            return f'{self.start}:{self.stop}:{self.step}'
        else:
            return f'{self.start}:{self.stop}'


def cld(n : int, d : int):
    """Ceiling-divide."""
    return (n // d) + (1 if n % d > 0 else 0)

def flog2(n : int):
    return int(np.log2(n))

def maxpow2(n : int):
    return int(1 << flog2(n))

def blk2d(n : int, max=64):
    i = 1
    while i < n and i < max: i <<= 1

    return {
        1: (1, 1),
        2: (1, 2),
        4: (2, 2),
        8: (2, 4),
        16: (4, 4),
        32: (4, 8),
        64: (8, 8),
    }[i]

class Dtype(IntEnum):
    """Enum for the different data types supported by UPCYCLE."""
    I8 = 1
    FP16 = 2

    @staticmethod
    def sizeof(dt): return int(dt)

    @staticmethod
    def from_str(s):
        if s == 'I8': return Dtype.I8
        if s == 'FP16': return Dtype.FP16
        raise ValueError(f'Invalid dtype: {s}')

    def __repr__(self):
        if self == Dtype.I8: return 'Int8'
        elif self == Dtype.FP16: return 'Float16'
        else: raise ValueError(f'Invalid dtype: {self}')

    def __str__(self): return self.__repr__()

@dataclass(frozen=True)
class Operator:
    """Base class for an Operator."""

    # Data type of the operand tensors
    dtype : Dtype

    # Whether or not this operator is a forward op that must be differentiated
    # for backward-propagation.
    fwd : bool

    @property
    def flops(self) -> int: raise NotImplementedError()

    @property
    def total_read_bytes(self) -> int: raise NotImplementedError()

    @property
    def total_weight_bytes(self) -> int: raise NotImplementedError()

    @property
    def total_write_bytes(self) -> int: raise NotImplementedError()

    @property
    def ami(self) -> float: return self.flops / self.total_read_bytes

    @property
    def min_llc_capacity(self) -> int: raise NotImplementedError()

    def make_tensors(self, arch) -> tuple[list, list]: raise NotImplementedError()
