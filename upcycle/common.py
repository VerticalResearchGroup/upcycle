from dataclasses import dataclass, field
from enum import Enum
import random

@dataclass(order=True)
class Arch:
    freq : float
    nrow : int
    ncol : int

    @property
    def ntiles(self): return self.nrow * self.ncol

    def tile_coords(self, tid):
        assert tid >= 0 and tid < self.ntiles
        return (tid // self.ncol), (tid % self.ncol)

    def addr_llc_coords(self, addr : int):
        line = addr >> 6
        tid = line & (self.ntiles - 1)
        return self.tile_coords(tid)

@dataclass(order=True)
class ArchRandomLlc(Arch):
    llc_addr_map : list[int] = field(default_factory=list)

    def __post_init__(self):
        self.llc_addr_map = list(range(self.ntiles))
        random.shuffle(self.llc_addr_map)

    def addr_llc_coords(self, addr : int):
        line = addr >> 6
        tid = line & (self.ntiles - 1)
        return self.tile_coords(self.llc_addr_map[tid])

class Dtype(Enum):
    I8 = 1
    FP16 = 2

