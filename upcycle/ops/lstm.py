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
    def total_read_bytes(self) -> int:
        return self.n * (self.d + self.h) * Dtype.sizeof(self.dtype) + \
            (self.d + self.h) * (self.d + self.h) * 4 * Dtype.sizeof(self.dtype)

    @property
    def total_weight_bytes(self) -> int:
        return (self.d + self.h) * (self.d + self.h) * 4 * Dtype.sizeof(self.dtype)

    @property
    def total_write_bytes(self) -> int:
        return self.n * (self.d + self.h) * Dtype.sizeof(self.dtype)

    def make_tensors(self, arch): return [], []

    def __repr__(self):
        return f'LstmCell[{self.dtype}][{self.mm.layout_str(self.tr_xh, self.tr_wu)}]({self.n}, {self.d}, {self.h})'

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

@operator
@dataclass(frozen=True)
@register_backward(LstmCell)
class LstmCellBwd(LstmCell):
    @staticmethod
    def from_forward(lstm : LstmCell):
        return LstmCell(lstm.dtype, False, lstm.n, lstm.d, lstm.h, lstm.tr_xh, lstm.tr_wu)

    @property
    def flops(self): return self.mm.flops * 2

    @property
    def total_read_bytes(self) -> int:
        return 2 * self.n * (self.d + self.h) * Dtype.sizeof(self.dtype) + \
            (self.d + self.h) * (self.d + self.h) * 4 * Dtype.sizeof(self.dtype)

    @property
    def total_weight_bytes(self) -> int:
        return self.n * (self.d + self.h) * Dtype.sizeof(self.dtype)

    @property
    def total_write_bytes(self) -> int:
        return self.total_read_bytes

    def __repr__(self):
        return f'LstmCellBwd[{self.dtype}][{self.mm.layout_str(self.tr_xh, self.tr_wu)}]({self.n}, {self.d}, {self.h})'

@M.register_placement([OracleArch, BgroupArch, FbcastArch, HierArch], LstmCellBwd)
def place_lstmbwd_default(arch : Arch, lstm : LstmCellBwd, sim : M.SimBase):
    ins, outs = lstm.make_tensors(arch)

    sim.map2d_place([
        [
            [
                LstmCellBackend(arch, lstm, ins, outs),
                LstmCellBackend(arch, lstm, ins, outs)
            ]
            for col in range(arch.ncols)
        ]
        for row in range(arch.nrows)
    ])

    sim.barrier()

    M.place_op(
        arch,
        matmul.MatmulDa.from_forward(lstm.mm),
        sim,
        check_flops=False)

    M.place_op(
        arch,
        matmul.MatmulDb.from_forward(lstm.mm),
        sim,
        check_flops=False)
