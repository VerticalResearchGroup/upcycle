from dataclasses import dataclass
import logging

from ..common import *
from .common import *

from . import matmul

logger = logging.getLogger(__name__)

@operator
@dataclass(frozen=True)
class LstmCell(Operator):
    n : int
    d : int
    h : int
    tr_xh : bool
    tr_wu : bool

    @functools.cached_property
    def mm(self):
        return matmul.Linear(
            self.dtype,
            self.train,
            1,
            self.n,
            self.h * 4,
            self.d + self.h,
            self.tr_xh,
            self.tr_wu)

    @property
    def flops(self): return self.mm.flops

    @property
    def total_load_bytes(self): return self.mm.total_load_bytes

    def make_tensors(self, arch): return [], []

@dataclass(frozen=True)
class LstmCellBackend(M.WorkItem):
    @property
    def flops(self): return 0

    @property
    def read_trace(self):
        return
        yield

    @property
    def exec_lat(self): return 32

    @property
    def write_trace(self): raise NotImplementedError()

@M.register_placement([OracleArch, BgroupArch, FbcastArch, HierArch], LstmCell)
def place_lstm_default(arch : Arch, lstm : LstmCell, sim : M.SimBase):
    ins, outs = lstm.make_tensors(arch)

    M.place_op(arch, lstm.mm, sim, check_flops=False)
    sim.barrier()

    sim.map2d_place([
        [
            [ LstmCellBackend(arch, lstm, ins, outs) ]
            for col in range(arch.ncols)
        ]
        for row in range(arch.nrows)
    ])


