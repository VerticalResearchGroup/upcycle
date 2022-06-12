from dataclasses import dataclass
import logging

from ..common import *
from .common import *

from . import matmul

logger = logging.getLogger(__name__)

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

@dataclass(frozen=True)
class LstmTile(M.WorkItemPerfectCompute):
    lstm : Lstm
    write : bool
    c : M.Tensor
    x : M.Tensor
    h : M.Tensor
    w : M.Tensor
    u : M.Tensor

    si : int
    ni : int
    hs : Slice

    @property
    def flops(self):
        return 4 * len(self.hs) * (self.lstm.h + self.lstm.d) * 2

    @property
    def read_trace(self):
        if not self.lstm.tr_xh:
            yield from self.c[self.si, self.ni, :]
            yield from self.x[self.si, self.ni, :]
            yield from self.h[self.si, self.ni, :]
        else:
            yield from self.c[self.si, :, self.ni]
            yield from self.x[self.si, :, self.ni]
            yield from self.h[self.si, :, self.ni]

        if not self.lstm.tr_wu:
            yield from self.w[:, self.hs + (self.lstm.h * 0)]
            yield from self.w[:, self.hs + (self.lstm.h * 1)]
            yield from self.w[:, self.hs + (self.lstm.h * 2)]
            yield from self.w[:, self.hs + (self.lstm.h * 3)]

            yield from self.u[:, self.hs + (self.lstm.h * 0)]
            yield from self.u[:, self.hs + (self.lstm.h * 1)]
            yield from self.u[:, self.hs + (self.lstm.h * 2)]
            yield from self.u[:, self.hs + (self.lstm.h * 3)]
        else:
            yield from self.w[self.hs + (self.lstm.h * 0), :]
            yield from self.w[self.hs + (self.lstm.h * 1), :]
            yield from self.w[self.hs + (self.lstm.h * 2), :]
            yield from self.w[self.hs + (self.lstm.h * 3), :]

            yield from self.u[self.hs + (self.lstm.h * 0), :]
            yield from self.u[self.hs + (self.lstm.h * 1), :]
            yield from self.u[self.hs + (self.lstm.h * 2), :]
            yield from self.u[self.hs + (self.lstm.h * 3), :]


    @property
    def write_trace(self):
        if not self.write: return
        yield from self.c[self.li, self.ms, self.ns]

@M.register_placement('flatmap', FlatMeshArch, Lstm)
@M.register_placement('flatmap', OracleArch, Lstm)
def place_lstm_flatmap(arch : Arch, lstm : Lstm):
    logger.debug(f'=== Place LSTM ===')
    logger.debug(f'+ LSTM: {lstm}')

    n = lstm.n
    s = lstm.s
    d = lstm.d
    h = lstm.h

    tc = M.Tensor(1, lstm.dtype, (s, n, h) if not lstm.tr_xh else (s, h, n))
    tx = M.Tensor(2, lstm.dtype, (s, n, d) if not lstm.tr_xh else (s, d, n))
    th = M.Tensor(3, lstm.dtype, (s, n, h) if not lstm.tr_xh else (s, h, n))
    tw = M.Tensor(4, lstm.dtype, (h * 4, d) if not lstm.tr_wu else (d, h * 4))
    tu = M.Tensor(5, lstm.dtype, (h * 4, h) if not lstm.tr_wu else (h, h * 4))

    wl = M.WorkList.from_arch(arch, [tc, tx, th, tw, tu])

    hblk = 128
    rows_per_batch = h / hblk
    col_height = arch.nrows
    while col_height > rows_per_batch: col_height //= 2
    batch_per_col = arch.nrows // col_height

    logger.debug(f'+ hblk={hblk}')

    for si in range(s):
        logger.debug(f'+ si={si}')
        for ni in range(n):
            col = int((ni // batch_per_col) % arch.ncols)
            row = int((ni % batch_per_col) * col_height)
            logger.debug(f'    + ni={ni}, bbox={(row, row + col_height, col, col + 1)}')
            wl.flatmap_place([
                [
                    LstmTile(
                        arch, lstm.dtype, lstm, False,
                        tc, tx, th, tw, tu,
                        si, ni, Slice.blk(hb, h, hblk))
                ]
                for hb in range(0, h, hblk)
            ], bbox=(row, row + col_height, col, col + 1))

    return wl
