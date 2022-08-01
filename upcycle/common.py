from dataclasses import dataclass, field
from enum import IntEnum
import functools
import numpy as np
import random
import logging

logger = logging.getLogger(__name__)

# Adapted from https://stackoverflow.com/questions/2536307/decorators-in-the-python-standard-lib-deprecated-specifically
def deprecated(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        logger.warning(f'Call to deprecated function {func.__name__} (in {func.__module__}).')
        return func(*args, **kwargs)
    return wrapper

@dataclass(frozen=True, order=True)
class Slice:
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

    @property
    def blocks(self):
        for i in range(self.start, self.stop, self.step):
            yield i // self.step, i, min(i + self.step, self.stop)

def blkdiv(n, b):
    nn = n // b
    for i in range(b):
        r = nn + (1 if i < n % b else 0)
        yield r

class Dtype(IntEnum):
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
    dtype : Dtype
    train : bool

    @property
    def flops(self): raise NotImplementedError()
