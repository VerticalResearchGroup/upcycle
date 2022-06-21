from dataclasses import dataclass
import logging

from ..common import *
from .common import *

from . import matmul

logger = logging.getLogger(__name__)

@operator
@dataclass(frozen=True)
class Lstm(Operator):
    n : int
    s : int
    d : int
    h : int
    tr_xh : bool
    tr_wu : bool

    @property
    def flops(self):
        return self.s * sum([
            matmul.Linear(self.dtype, False, 1, self.n, self.h * 4, self.d, False, False).flops, # Xt*W
            matmul.Linear(self.dtype, False, 1, self.n, self.h * 4, self.h, False, False).flops, # Ht*U
            # Other ops not counted but are insignificant
        ])


@operator
@dataclass(frozen=True)
@register_backward(Lstm)
class LstmBwd(Lstm):
    @property
    def flops(self): return super().flops * 2

    @staticmethod
    def from_forward(l : Lstm):
        return LstmBwd(l.dtype, False, l.n, l.s, l.d, l.h, l.tr_xh, l.tr_wu)

@dataclass(frozen=True)
class LstmMatmul(M.WorkItemPerfectCompute):
    lstm : Lstm
    xh : M.Tensor
    wu : M.Tensor
    o : M.Tensor
    write : bool

    si : int
    ns : Slice # M = Batch
    hs : Slice # N = 4H
    hds : Slice # K = H or D

    @property
    def flops(self):
        return len(self.ns) * len(self.hs) * len(self.hds) * 2

    @property
    def read_trace(self):
        if not self.lstm.tr_xh: yield from self.xh[self.si, self.ns, :]
        else: yield from self.xh[self.si, :, self.ns]

        if self.lstm.tr_wu: yield from self.wu[self.si, self.hs, :]
        else: yield from self.wu[self.si, :, self.hs]

    @property
    def write_trace(self):
        if not self.write: return
        yield from self.o[self.si, self.ns, self.hs]

@dataclass(frozen=True)
class LstmBackend(M.WorkItemPerfectCompute):
    lstm : Lstm
    txp : M.Tensor # S, N, 4H
    thp : M.Tensor # S, N, 4H
    tc : M.Tensor # S, N, H or S, H, N
    th : M.Tensor # S, N, H or S, H, N
    write : bool

    si : int
    ns : Slice
    hs : Slice

    @property
    def flops(self): return 0

    @property
    def read_trace(self):
        yield from self.txp[self.si, self.ns, self.hs + self.lstm.h * 0]
        yield from self.thp[self.si, self.ns, self.hs + self.lstm.h * 0]
        yield from self.txp[self.si, self.ns, self.hs + self.lstm.h * 1]
        yield from self.thp[self.si, self.ns, self.hs + self.lstm.h * 1]
        yield from self.txp[self.si, self.ns, self.hs + self.lstm.h * 2]
        yield from self.thp[self.si, self.ns, self.hs + self.lstm.h * 2]
        yield from self.txp[self.si, self.ns, self.hs + self.lstm.h * 3]
        yield from self.thp[self.si, self.ns, self.hs + self.lstm.h * 3]

    @property
    def write_trace(self): raise NotImplementedError()

def flatmap_lstm_mm(arch : Arch, lstm : Lstm, wl : M.WorkList, txh, twu, to, si, hd, offset=0, bbox=None):
    return wl.flatmap_place([
        [
            LstmMatmul(
                arch, lstm.dtype,
                lstm, txh, twu, to, False, si,
                Slice.blk(bm, lstm.n, 16),
                Slice.blk(bn, 4 * lstm.h, 8),
                Slice.blk(bk, hd, 64))
            for bk in range(0, hd, 64)
        ]
        for bm in range(0, lstm.n, 16)
        for bn in range(0, 4 * lstm.h, 8)
    ], offset=offset, bbox=bbox)

def flatmap_lstm_backend(arch : Arch, lstm : Lstm, wl : M.WorkList, txp, thp, tc, th, si, offset=0, bbox=None):
    return wl.flatmap_place([
        [
            LstmBackend(
                arch, lstm.dtype,
                lstm, txp, thp, tc, th, False, si,
                Slice.blk(bn, lstm.n, 1),
                Slice.blk(bh, lstm.h, 32))
        ]
        for bn in range(0, lstm.n, 1)
        for bh in range(0, lstm.h, 32)
    ], offset=offset, bbox=bbox)

@M.register_placement('flatmap', [OracleArch, BgroupArch], [Lstm])
def place_lstm_flatmap(arch : Arch, lstm : Lstm):
    logger.debug(f'=== Place LSTM ===')
    logger.debug(f'+ LSTM: {lstm}')

    n = lstm.n
    s = lstm.s
    d = lstm.d
    h = lstm.h

    tc = M.Tensor(arch, 1, lstm.dtype, (s, n, h) if not lstm.tr_xh else (s, h, n))
    tx = M.Tensor(arch, 2, lstm.dtype, (s, n, d) if not lstm.tr_xh else (s, d, n))
    th = M.Tensor(arch, 3, lstm.dtype, (s, n, h) if not lstm.tr_xh else (s, h, n))
    txp = M.Tensor(arch, 3, lstm.dtype, (s, n, 4 * h))
    thp = M.Tensor(arch, 3, lstm.dtype, (s, n, 4 * h))
    tw = M.Tensor(arch, 4, lstm.dtype, (s, h * 4, d) if lstm.tr_wu else (s, d, h * 4))
    tu = M.Tensor(arch, 5, lstm.dtype, (s, h * 4, h) if lstm.tr_wu else (s, h, h * 4))

    wl = M.WorkList.from_arch(arch, [tc, tx, th, txp, thp, tw, tu])

    cols_x = int(d / (d + h) * arch.ncols)
    off1 = 0
    off2 = 0

    for si in range(s):
        off1 += flatmap_lstm_mm(arch, lstm, wl, tx, tw, txp, si, d, offset=off1)
        off1 += flatmap_lstm_mm(arch, lstm, wl, th, tu, thp, si, h, offset=off1)
        off2 += flatmap_lstm_backend(arch, lstm, wl, txp, thp, tc, th, si, offset=off2)

    return wl

@M.register_placement('pg', [OracleArch, BgroupArch], [Lstm])
def place_lstm_profiled(arch : Arch, lstm : Lstm):
    return profiled_placement(arch, lstm, place_lstm_flatmap)
