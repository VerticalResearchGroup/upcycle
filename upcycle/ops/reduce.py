from dataclasses import dataclass
import logging

from ..common import *
from .common import *

logger = logging.getLogger(__name__)

@operator
@dataclass(frozen=True)
class Reduce(Operator):
    n : int
    m : int

    @property
    def flops(self): return self.n * self.m

    @property
    def total_load_bytes(self): return self.n * self.m * Dtype.sizeof(self.dtype)

    def make_tensors(self, arch): return [], []


    def __repr__(self):
        return f'Reduce[{self.dtype}]({self.n}x{self.m})'

@dataclass(frozen=True)
class ReduceTile(M.WorkItem):
    ns : Slice
    ms : Slice

    @property
    def flops(self): return len(self.ms) * len(self.ns)

    @property
    def read_trace(self):
        return
        yield

    @property
    def write_trace(self):
        return
        yield

    @property
    def exec_lat(self):
        return len(self.ns) * cld(len(self.ms), self.arch.vlen(self.op.dtype))

@M.register_placement(
    [OracleArch, BgroupArch, FbcastArch, HierArch, CoarseOracle], Reduce)
def place_reduction_default(arch : Arch, r : Reduce, sim : M.SimBase):
    # The reuction operation assumes we are trying to sum a set of N tensors
    # of flattened size M. We further assume that adjacent indices in the outer
    # N dimension all map to the same LLC slice. This means that the reduction
    # can happen entirely inside each tile with absolutely no LLC <-> L1/L2
    # traffic.

    ins, outs = r.make_tensors(arch)

    def inner_loop(bm):
        return (
            ReduceTile(arch, r, ins, outs, ns, bm)
            for ns in Slice(0, r.n).blkslice(1)
        )

    sim.map2d_place([
        [
            inner_loop(bm1)
            for bm1 in bm0.blkslice(arch.ncols)
        ]
        for bm0 in Slice(0, r.m).blkslice(arch.nrows)
    ])
